# Owner(s): ["module: PrivateUse1"]

"""Tests for ``python -m torch_rbln.diagnose`` and the report it prints.

diagnose is what an ``import torch_rbln`` that fails points at, so it has to run exactly when
the runtime cannot be loaded, and say which ``rbln``, which ``librbln_rt.so`` and which ABI ids
it found.
"""

import importlib.util
import os
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
_RUNTIME = "/tree/build/rbln/librbln_rt.so"


def _diagnostics(**overrides) -> dict:
    d = {
        "torch_rbln": {},
        "rbln_runtime": {
            "package": {"found": True, "origin": "/tree/rbln/python/rbln/__init__.py"},
            "module": "/tree/rbln/python/rbln/runtime/__init__.py",
            "path": _RUNTIME,
            "mapped": [_RUNTIME],
            "under_rebel_home": True,
            "error": None,
        },
        "abi": {
            "built_abi": "a" * 64,
            "built_include_dir": "/tree/rbln/include",
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
        self.assertIn("rbln package: /tree/rbln/python/rbln/__init__.py", text)
        self.assertIn(f"librbln_rt.so: {_RUNTIME}", text)
        self.assertIn(f"built ABI:   {'a' * 64}", text)
        self.assertIn("headers:   /tree/rbln/include", text)
        self.assertIn(f"runtime ABI: {'a' * 64}", text)
        self.assertIn("verdict: OK", text)

    def test_an_unimportable_rbln_says_where_to_get_it(self):
        d = _diagnostics()
        d["rbln_runtime"] = {"package": {"found": False}, "path": None, "mapped": [], "error": "No module named 'rbln'"}
        text = env_diagnostic.format_diagnostics(d)
        self.assertIn("rbln package: not importable", text)
        self.assertIn("PYTHONPATH=$REBEL_HOME/rbln/python", text)
        self.assertIn("librbln_rt.so: not mapped", text)

    def test_two_copies_and_a_runtime_outside_rebel_home_are_flagged(self):
        d = _diagnostics()
        d["rbln_runtime"].update(mapped=[_RUNTIME, "/other/librbln_rt.so"], under_rebel_home=False)
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

    def test_runs_when_rbln_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as stub_root:
            os.makedirs(os.path.join(stub_root, "rbln"))
            with open(os.path.join(stub_root, "rbln", "__init__.py"), "w") as f:
                f.write("raise ImportError('rbln stub for the test')\n")
            result = _run_diagnose([stub_root, _PACKAGE_ROOT])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rbln stub for the test", result.stderr)
        self.assertIn("librbln_rt.so: not mapped", result.stderr)
        self.assertIn("verdict: no runtime is mapped to compare against", result.stderr)

    def test_reports_the_runtime_rbln_maps(self):
        spec = importlib.util.find_spec("rbln")
        if spec is None or spec.origin is None:
            self.skipTest("rbln is not importable here")
        rbln_root = os.path.dirname(os.path.dirname(spec.origin))
        result = _run_diagnose([_PACKAGE_ROOT, rbln_root])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stderr, r"librbln_rt\.so: /\S+/librbln_rt\.so")
        self.assertRegex(result.stderr, r"runtime ABI: [0-9a-f]{64}")


if __name__ == "__main__":
    run_tests()
