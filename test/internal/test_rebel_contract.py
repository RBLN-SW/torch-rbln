# Owner(s): ["module: PrivateUse1"]

"""The rebel-compiler Python surface torch-rbln declares in ``rebel_contract``.

Three things are checked here:

  - the installed rebel still satisfies ``CONTRACT`` (``BROKEN`` is empty);
  - the verifier can tell the divergences apart, against a fabricated module, so a green run
    means "checked" rather than "read nothing";
  - ``rebel_contract`` is still the only module under ``torch_rbln/`` that imports rebel.

``DRIFTED`` is reported, never asserted: rebel lands interface changes before torch-rbln follows,
so a rebel parameter that gained a default must not fail this suite.
"""

import ast
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import rebel_contract
from torch_rbln._internal.rebel_contract import BROKEN, DRIFTED, Name


_REPO_ROOT = Path(__file__).resolve().parents[2]

_FAKE_MODULE = "torch_rbln_fake_rebel"

# What torch-rbln passes to the sync-runtime prepare methods; reused by the fabricated cases.
_PARAMS = ("device_inputs", "cpu_inputs")


@pytest.mark.test_set_ci
class TestContractHolds(TestCase):
    """The installed rebel against what torch-rbln declares it calls."""

    def test_nothing_broken(self):
        divergences = rebel_contract.broken()
        self.assertEqual(divergences, [], "\n".join(str(d) for d in divergences))

    def test_no_entry_is_broken(self):
        # Per entry as well as in aggregate, for a failure that names which one. DRIFTED is not
        # asserted: a rebel parameter that gained a default still works, and turning that red
        # here would make torch-rbln's suite the thing that blocks an additive rebel change.
        labels = [name.label for name in rebel_contract.CONTRACT]
        self.assertEqual(len(labels), len(set(labels)), "a duplicated entry hides a real divergence")
        for name in rebel_contract.CONTRACT:
            with self.subTest(name=name.label):
                divergence = rebel_contract._verify(name)
                self.assertNotEqual(getattr(divergence, "kind", None), BROKEN, str(divergence))

    def test_every_entry_resolves(self):
        # Guards the other way: an entry naming something that never existed would report BROKEN
        # once and otherwise look like coverage.
        for name in rebel_contract.CONTRACT:
            with self.subTest(name=name.label):
                self.assertIsNone(rebel_contract._resolution_error(name), name.label)

    def test_runtime_methods_are_what_the_hit_path_calls(self):
        self.assertEqual(rebel_contract.RUNTIME_METHODS, ("prepare_inputs", "prepare_outputs", "run"))


@pytest.mark.test_set_ci
class TestVerifierDetects(TestCase):
    """The verifier against a fabricated rebel, one divergence per case."""

    def _entry(self, fn=None, attr="entry", params=_PARAMS, **members):
        """Register a throwaway module and verify ``attr`` on it against ``params``."""
        module = types.ModuleType(_FAKE_MODULE)
        if fn is not None:
            module.entry = fn
        for name, value in members.items():
            setattr(module, name, value)
        sys.modules[_FAKE_MODULE] = module
        self.addCleanup(sys.modules.pop, _FAKE_MODULE, None)
        return rebel_contract._verify(Name(_FAKE_MODULE, attr, params))

    def test_unchanged_is_clean(self):
        def entry(device_inputs, cpu_inputs): ...

        self.assertIsNone(self._entry(entry))

    def test_renamed_parameter_is_broken(self):
        def entry(dev, cpu): ...

        self.assertEqual(self._entry(entry).kind, BROKEN)

    def test_reordered_parameters_are_broken(self):
        def entry(cpu_inputs, device_inputs): ...

        self.assertEqual(self._entry(entry).kind, BROKEN)

    def test_new_required_parameter_is_broken(self):
        def entry(device_inputs, cpu_inputs, stale): ...

        self.assertEqual(self._entry(entry).kind, BROKEN)

    def test_new_required_keyword_only_is_broken(self):
        def entry(device_inputs, cpu_inputs, *, stale): ...

        self.assertEqual(self._entry(entry).kind, BROKEN)

    def test_new_defaulted_parameter_is_drift_not_break(self):
        def entry(device_inputs, cpu_inputs, stale=()): ...

        divergence = self._entry(entry)
        self.assertEqual(divergence.kind, DRIFTED)
        self.assertIn("stale", divergence.detail)

    def test_missing_attribute_is_broken(self):
        self.assertEqual(self._entry().kind, BROKEN)

    def test_missing_module_is_broken(self):
        self.assertEqual(rebel_contract._verify(Name("torch_rbln_no_such_rebel_module")).kind, BROKEN)

    def test_leading_self_is_not_counted(self):
        class Runtime:
            def entry(self, device_inputs, cpu_inputs): ...

        self.assertIsNone(self._entry(attr="Runtime.entry", Runtime=Runtime))


@pytest.mark.test_set_ci
class TestSignatureReader(TestCase):
    """``inspect.signature`` raises on every pybind11 entry point; the docstring is the fallback."""

    def test_reads_a_real_pybind_signature(self):
        prepare_inputs = _sync_runtime_type().prepare_inputs
        signature = rebel_contract._read_signature(prepare_inputs)
        self.assertIsNotNone(signature)
        self.assertEqual(signature.positional[0], "self")
        self.assertIn("device_inputs", signature.positional)

    def test_overloaded_entry_point_reads_as_unknown_not_broken(self):
        # pybind renders an overload set as ``(*args, **kwargs)``, which names no parameters.
        # Reporting BROKEN off that would be a false alarm, so it reads as unknown instead.
        def entry(*args, **kwargs): ...

        entry.__doc__ = "entry(*args, **kwargs)\nOverloaded function."
        self.assertIsNone(rebel_contract._read_pybind_signature(entry))

    def test_unparseable_docstring_reads_as_unknown(self):
        def entry(): ...

        entry.__doc__ = "not a signature at all"
        self.assertIsNone(rebel_contract._read_pybind_signature(entry))


@pytest.mark.test_set_ci
class TestImportChokepoint(TestCase):
    """``rebel_contract`` is the only module under ``torch_rbln/`` that imports rebel.

    ruff's TID251 enforces this in lint; this asserts it from the tree, so the invariant holds
    where lint is not run. Parsed rather than grepped: a rebel import inside a string -- the
    subprocess scripts under test/ are full of them -- is not an import.
    """

    CONTRACT_MODULE = Path("torch_rbln/_internal/rebel_contract.py")

    def test_only_the_contract_module_imports_rebel(self):
        offenders = []
        sources = sorted((_REPO_ROOT / "torch_rbln").rglob("*.py"))
        self.assertTrue(sources, "found no sources to check")
        for path in sources:
            relative = path.relative_to(_REPO_ROOT)
            if relative == self.CONTRACT_MODULE:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.split(".")[0] == "rebel" for name in names):
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], "add the name to rebel_contract.CONTRACT and reach it through that module")

    def test_the_check_can_see_an_offender(self):
        # Without this, a walk that silently visited nothing would read as a clean tree.
        tree = ast.parse("from rebel.device_info import get_npu_name\n")
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual([m.split(".")[0] for m in modules if m], ["rebel"])

    def test_a_rebel_import_inside_a_string_is_not_an_import(self):
        tree = ast.parse('SCRIPT = """\nfrom rebel._C import get_npu_name\n"""\n')
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))], [])


@pytest.mark.test_set_ci
class TestConfigureTimeReport(TestCase):
    """``tools/check_rebel_contract.py``, whose output FindRebel.cmake matches line by line."""

    def _run(self, contract_source):
        contract = Path(self._tmp) / "torch_rbln" / "_internal"
        contract.mkdir(parents=True)
        (contract / "rebel_contract.py").write_text(contract_source)
        (Path(self._tmp) / "tools").mkdir()
        checker = Path(self._tmp) / "tools" / "check_rebel_contract.py"
        checker.write_text((_REPO_ROOT / "tools" / "check_rebel_contract.py").read_text())
        return subprocess.run([sys.executable, str(checker)], capture_output=True, text=True)

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)

    def test_clean_contract_prints_nothing(self):
        result = self._run("def verify():\n    return []\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")

    def test_each_divergence_is_one_matchable_line(self):
        result = self._run(
            "class D:\n"
            "    kind = 'DRIFTED'\n"
            "    name = 'rebel.x:y'\n"
            "    detail = 'grew\\ndefaulted'\n"
            "def verify():\n"
            "    return [D(), D()]\n"
        )
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2, result.stdout)
        for line in lines:
            self.assertTrue(line.startswith("DRIFTED=rebel.x:y: "), line)

    def test_a_broken_check_reports_and_still_exits_zero(self):
        # The build must not stop because the contract check itself could not run.
        result = self._run("raise RuntimeError('boom\\nsecond line')\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout.splitlines()), 1, result.stdout)
        self.assertTrue(result.stdout.startswith("ERROR="), result.stdout)


def _sync_runtime_type():
    """rebel's sync-runtime type, reached the way ``CONTRACT`` declares it."""
    import importlib

    return importlib.import_module("rebel._C").PyRblnSyncRuntime


if __name__ == "__main__":
    run_tests()
