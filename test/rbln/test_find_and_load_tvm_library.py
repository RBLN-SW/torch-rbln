# Owner(s): ["module: PrivateUse1"]

"""
Test find_and_load_tvm_library error handling when librbln.so is not found.

Validates the FileNotFoundError message: searched directories, diagnose hint,
and optional env (REBEL_HOME, LD_LIBRARY_PATH, PYTHONPATH).
"""

import os


# Avoid running torch_backends_entry_point on import (would load librbln.so)
os.environ["TORCH_RBLN_DIAGNOSE"] = "1"

from unittest.mock import patch

import pytest
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln


@pytest.mark.test_set_ci
class TestFindAndLoadTvmLibraryLibrblnError(TestCase):
    """Test FileNotFoundError when librbln.so is not in any search path."""

    def test_librbln_not_found_raises_with_expected_message(self):
        # No directory contains librbln.so -> find_and_load_tvm_library raises
        fake_searched = ["/fake/tvm/lib", "/fake/rebel/build"]
        with (
            patch(
                "torch_rbln.get_dll_directories",
                return_value=[],  # so os.walk never finds librbln.so
            ),
            patch(
                "torch_rbln.get_dll_directory_candidates",
                return_value=[(p, True) for p in fake_searched],
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                torch_rbln.find_and_load_tvm_library("librbln.so")
        err = ctx.exception
        self.assertIn("Could not find librbln.so", str(err))
        self.assertIn("Searched directories (in order):", str(err))
        self.assertIn("/fake/tvm/lib", str(err))
        self.assertIn("/fake/rebel/build", str(err))
        self.assertIn("python -m torch_rbln.diagnose", str(err))
        self.assertIn("rebel-compiler", str(err))
        self.assertIn("REBEL_HOME", str(err))

    def test_librbln_not_found_includes_env_hint_when_set(self):
        fake_searched = ["/another/path"]
        with (
            patch(
                "torch_rbln.get_dll_directories",
                return_value=[],
            ),
            patch(
                "torch_rbln.get_dll_directory_candidates",
                return_value=[(p, True) for p in fake_searched],
            ),
            patch.dict(os.environ, {"REBEL_HOME": "/my/rebel"}, clear=False),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                torch_rbln.find_and_load_tvm_library("librbln.so")
        err = ctx.exception
        self.assertIn("Relevant env:", str(err))
        self.assertIn("REBEL_HOME=/my/rebel", str(err))

    def test_other_lib_not_found_raises_generic_message(self):
        with patch(
            "torch_rbln.get_dll_directories",
            return_value=[],
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                torch_rbln.find_and_load_tvm_library("libother.so")
        err = ctx.exception
        self.assertIn("libother.so not found in any known site-packages path", str(err))
        self.assertNotIn("librbln.so", str(err))


instantiate_device_type_tests(TestFindAndLoadTvmLibraryLibrblnError, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
