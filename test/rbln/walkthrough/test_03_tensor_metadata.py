# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 3: tensor metadata and view ops on RBLN.

Ported from walkthrough_guide/3.rbln_tensor_metadata_handling.py.

Verifies that RBLN tensors participate correctly in PyTorch's tensor
metadata / stride handling:
- `view()` and `reshape()` change shape/strides without copying when legal.
- `transpose()` and `permute()` reorder dimensions.
- `is_contiguous()` / `contiguous()` report and enforce layout.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestTensorMetadata(TestCase):
    """Walkthrough example 3: view/reshape/transpose/contiguous on RBLN."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_view_preserves_data_with_new_shape(self, dtype):
        """view() should produce a tensor with the new shape and shared storage."""
        x = torch.randn(4, 64, device=self.rbln_device, dtype=dtype)
        y = x.view(8, 32)
        self.assertEqual(y.shape, (8, 32))
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(x.data_ptr(), y.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    def test_transpose_permutes_dimensions(self, dtype):
        """transpose(0, 1) should swap dimensions and share storage."""
        x = torch.randn(4, 64, device=self.rbln_device, dtype=dtype)
        z = x.transpose(0, 1)
        self.assertEqual(z.shape, (64, 4))
        self.assertEqual(z.device.type, "rbln")
        self.assertEqual(x.data_ptr(), z.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    def test_reshape_contiguous(self, dtype):
        """reshape() on a contiguous input should match view() semantics."""
        x = torch.randn(4, 64, device=self.rbln_device, dtype=dtype)
        r = x.reshape(2, 128)
        self.assertEqual(r.shape, (2, 128))
        self.assertEqual(r.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_permute_reorders_dimensions(self, dtype):
        """permute() should reorder dims and share storage with the input."""
        x = torch.randn(2, 3, 4, device=self.rbln_device, dtype=dtype)
        p = x.permute(2, 0, 1)
        self.assertEqual(p.shape, (4, 2, 3))
        self.assertEqual(p.device.type, "rbln")
        self.assertEqual(x.data_ptr(), p.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    def test_transposed_is_not_contiguous(self, dtype):
        """A transposed rectangular tensor should not be contiguous."""
        x = torch.randn(4, 64, device=self.rbln_device, dtype=dtype)
        t = x.transpose(0, 1)
        self.assertFalse(t.is_contiguous())

    @dtypes(*SUPPORTED_DTYPES)
    def test_contiguous_produces_contiguous_copy(self, dtype):
        """`.contiguous()` on a non-contiguous view should return a contiguous tensor."""
        x = torch.randn(4, 64, device=self.rbln_device, dtype=dtype)
        t = x.transpose(0, 1)
        c = t.contiguous()
        self.assertTrue(c.is_contiguous())
        self.assertEqual(c.shape, t.shape)
        self.assertEqual(c.device.type, "rbln")


instantiate_device_type_tests(TestTensorMetadata, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
