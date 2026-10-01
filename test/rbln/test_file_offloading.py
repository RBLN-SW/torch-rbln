# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.offload`` and ``torch.rbln.release_offload_temp_storage``.

RBLN tensors live in device memory with no host-side copy, so the runtime has nothing to
page out to files. The Python surface stays so callers get a clear answer:

* ``offload`` is still exported, and calling it raises ``NotImplementedError`` instead of
  silently doing nothing;
* ``release_offload_temp_storage`` is still exported and reports that it removed nothing,
  so a shutdown path that calls it keeps working.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln.memory as torch_rbln_memory


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in pyproject.
# It asserts an absence at the torch level: the runtime has no file offloading to exercise.
pytestmark = pytest.mark.torch_rbln_only


@pytest.mark.test_set_ci
class TestFileOffloading(TestCase):
    def test_offload_is_exported(self):
        from torch_rbln.memory import offload  # noqa: F401

        self.assertIn("offload", torch_rbln_memory.__all__)
        self.assertTrue(hasattr(torch.rbln, "offload"))

    def test_offload_raises_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "offload"):
            with torch.rbln.offload():
                pass

    def test_release_offload_temp_storage_is_exported(self):
        from torch_rbln.memory import release_offload_temp_storage  # noqa: F401

        self.assertIn("release_offload_temp_storage", torch_rbln_memory.__all__)
        self.assertTrue(hasattr(torch.rbln, "release_offload_temp_storage"))

    def test_release_offload_temp_storage_removes_nothing(self):
        self.assertEqual(torch.rbln.release_offload_temp_storage(), 0)
        self.assertEqual(torch.rbln.release_offload_temp_storage(), 0)


if __name__ == "__main__":
    run_tests()
