# Owner(s): ["module: PrivateUse1"]

"""Tests for ``python -m torch_rbln.diagnose`` and the report it prints.

diagnose is what an ``import torch_rbln`` that fails points at, so it has to run exactly when
the runtime cannot be loaded, and say which ``rebel.v2``, which ``librebel_v2_rt.so`` and which ABI ids
it found.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln
from torch_rbln._internal import abi_check, env_diagnostic


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in pyproject.
pytestmark = pytest.mark.torch_rbln_only


_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(torch_rbln.__file__)))
_RUNTIME = "/tree/build/rebel/v2/librebel_v2_rt.so"


def _diagnostics(**overrides) -> dict:
    d = {
        "torch_rbln": {},
        "rbln_runtime": {
            "package": {"found": True, "origin": "/tree/rebel/python/rebel/v2/__init__.py"},
            "module": "/tree/rebel/python/rebel/v2/runtime/__init__.py",
            "path": _RUNTIME,
            "mapped": [_RUNTIME],
            "under_rebel_home": True,
            "error": None,
        },
        "abi": {
            "built_abi": "a" * 64,
            "built_include_dir": "/tree/rebel/v2/include",
            "runtime_abi": "a" * 64,
            "check_disabled": False,
            "verdict": "OK",
            "error": None,
        },
        "env": dict.fromkeys(env_diagnostic.ENV_VARS, ""),
        "gcc_versions": [],
        "python_executable": sys.executable,
    }
    d.update(overrides)
    return d


def _run_diagnose(pythonpath: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath))
    return subprocess.run(
        [sys.executable, "-m", "torch_rbln.diagnose"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tempfile.gettempdir(),
    )


@pytest.mark.test_set_ci
class TestFormatDiagnostics(TestCase):
    """The printed report, from a synthetic set of findings."""

    def test_the_runtime_and_both_ids_are_reported(self):
        text = env_diagnostic.format_diagnostics(_diagnostics())
        self.assertIn("rebel.v2 package: /tree/rebel/python/rebel/v2/__init__.py", text)
        self.assertIn(f"librebel_v2_rt.so: {_RUNTIME}", text)
        self.assertIn(f"built ABI:   {'a' * 64}", text)
        self.assertIn("headers:   /tree/rebel/v2/include", text)
        self.assertIn(f"runtime ABI: {'a' * 64}", text)
        self.assertIn("verdict: OK", text)

    def test_an_unimportable_rebel_v2_says_where_to_get_it(self):
        d = _diagnostics()
        d["rbln_runtime"] = {
            "package": {"found": False},
            "path": None,
            "mapped": [],
            "error": "No module named 'rebel.v2'",
        }
        text = env_diagnostic.format_diagnostics(d)
        self.assertIn("rebel.v2 package: not importable", text)
        self.assertIn("PYTHONPATH=$REBEL_HOME/rebel/python", text)
        self.assertIn("librebel_v2_rt.so: not mapped", text)

    def test_two_copies_and_a_runtime_outside_rebel_home_are_flagged(self):
        d = _diagnostics()
        d["rbln_runtime"].update(mapped=[_RUNTIME, "/other/librebel_v2_rt.so"], under_rebel_home=False)
        text = env_diagnostic.format_diagnostics(d)
        self.assertIn(">>> 2 copies mapped", text)
        self.assertIn(">>> The runtime is not from REBEL_HOME", text)

    def test_a_disabled_check_is_flagged(self):
        d = _diagnostics()
        d["abi"]["check_disabled"] = True
        self.assertIn("TORCH_RBLN_SKIP_ABI_CHECK", env_diagnostic.format_diagnostics(d))

    def test_every_verdict_has_a_readable_text(self):
        verdicts = [getattr(abi_check, name) for name in dir(abi_check) if name.startswith("VERDICT_")]
        missing = [
            v for v in verdicts if v != abi_check.VERDICT_SKIPPED_DISABLED and v not in env_diagnostic._VERDICT_TEXT
        ]
        self.assertEqual(missing, [])


@pytest.mark.test_set_ci
class TestDiagnoseModule(TestCase):
    """``python -m torch_rbln.diagnose`` in a fresh interpreter."""

    def test_runs_when_rebel_v2_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as stub_root:
            os.makedirs(os.path.join(stub_root, "rebel", "v2"))
            open(os.path.join(stub_root, "rebel", "__init__.py"), "w").close()
            with open(os.path.join(stub_root, "rebel", "v2", "__init__.py"), "w") as f:
                f.write("raise ImportError('rebel.v2 stub for the test')\n")
            result = _run_diagnose([stub_root, _PACKAGE_ROOT])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rebel.v2 stub for the test", result.stderr)
        self.assertIn("librebel_v2_rt.so: not mapped", result.stderr)
        self.assertIn("verdict: no runtime is mapped to compare against", result.stderr)

    def test_reports_the_runtime_rebel_v2_maps(self):
        spec = importlib.util.find_spec("rebel.v2")
        if spec is None or spec.origin is None:
            self.skipTest("rebel.v2 is not importable here")
        rebel_root = str(pathlib.Path(spec.origin).parents[2])
        result = _run_diagnose([_PACKAGE_ROOT, rebel_root])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stderr, r"librebel_v2_rt\.so: /\S+/librebel_v2_rt\.so")
        self.assertRegex(result.stderr, r"runtime ABI: [0-9a-f]{64}")


if __name__ == "__main__":
    run_tests()
