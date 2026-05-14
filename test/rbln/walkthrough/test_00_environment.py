# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 0: environment verification.

Ported from walkthrough_guide/0.environment_verification.py.

Verifies that the environment is correctly configured for RBLN:
- torch and torch_rbln import and expose version strings.
- The `rbln-smi` CLI is available on PATH and exits successfully.
"""

import shutil
import subprocess
import sys

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln


@pytest.mark.test_set_ci
class TestEnvironmentVerification(TestCase):
    """Environment sanity checks (Walkthrough example 0)."""

    def test_python_version_available(self):
        """Python version string should be reportable."""
        self.assertIsInstance(sys.version, str)
        self.assertGreater(len(sys.version), 0)

    def test_torch_version_available(self):
        """torch.__version__ should be a non-empty string."""
        self.assertTrue(hasattr(torch, "__version__"))
        self.assertIsInstance(torch.__version__, str)
        self.assertGreater(len(torch.__version__), 0)

    def test_torch_rbln_version_available(self):
        """torch_rbln.__version__ should be a non-empty string."""
        self.assertTrue(hasattr(torch_rbln, "__version__"))
        self.assertIsInstance(torch_rbln.__version__, str)
        self.assertGreater(len(torch_rbln.__version__), 0)

    def test_rbln_smi_available(self):
        """`rbln-smi` binary should exist on PATH and run successfully."""
        rbln_smi = shutil.which("rbln-smi")
        if rbln_smi is None:
            self.skipTest("rbln-smi is not available on PATH")

        result = subprocess.run(
            [rbln_smi],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rbln-smi failed with exit {result.returncode}: {result.stderr}",
        )


instantiate_device_type_tests(TestEnvironmentVerification, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
