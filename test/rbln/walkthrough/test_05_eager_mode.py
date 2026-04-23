# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 5: torch eager execution on RBLN.

Ported from walkthrough_guide/5.eager_mode.py.

Verifies that standard PyTorch eager ops run on RBLN without `torch.compile`:
- Pointwise ops (add, relu, mul) and in-place variants (`mul_`), with scalar broadcast.
- Reductions (sum, mean, max) both full and along a dim.
- Matrix multiply via `torch.matmul`, the `@` operator, and matmul + bias.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestEagerMode(TestCase):
    """Walkthrough example 5: eager pointwise, reduction, and matmul on RBLN."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_pointwise_add_and_relu(self, dtype):
        """`x + y` followed by `torch.relu` should stay on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        y = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)

        z = x + y
        w = torch.relu(z)

        self.assertEqual(w.shape, (64, 64))
        self.assertEqual(w.dtype, dtype)
        self.assertEqual(w.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_scalar_broadcast_mul_and_add(self, dtype):
        """Scalar broadcast via `*` and `+` should keep shape and device."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        y = x * 2.0 + 1.0
        self.assertEqual(y.shape, (64, 64))
        self.assertEqual(y.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_inplace_mul_(self, dtype):
        """`x.mul_(scalar)` should mutate in place on RBLN without moving storage."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        original_ptr = x.data_ptr()

        returned = x.mul_(2.0)

        # Same tensor, same storage.
        self.assertIs(returned, x)
        self.assertEqual(x.data_ptr(), original_ptr)
        self.assertEqual(x.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_sum_full(self, dtype):
        """Full reduction via `tensor.sum()` should produce a scalar tensor on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        s = x.sum()
        self.assertEqual(s.shape, ())
        self.assertEqual(s.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_sum_along_dim(self, dtype):
        """Reduction along a dim should produce a 1-D tensor on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        row_sum = x.sum(dim=1)
        self.assertEqual(row_sum.shape, (64,))
        self.assertEqual(row_sum.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_mean_full(self, dtype):
        """Full `mean()` should produce a scalar tensor on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        m = x.mean()
        self.assertEqual(m.shape, ())
        self.assertEqual(m.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_mean_along_dim(self, dtype):
        """`mean(dim=...)` should produce a 1-D tensor on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        row_mean = x.mean(dim=1)
        self.assertEqual(row_mean.shape, (64,))
        self.assertEqual(row_mean.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_max_full(self, dtype):
        """Full `max()` should produce a scalar tensor on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        m = x.max()
        self.assertEqual(m.shape, ())
        self.assertEqual(m.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_reduction_max_along_dim(self, dtype):
        """`max(dim=...)` should return a `(values, indices)` pair, both on RBLN."""
        x = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        row_max = x.max(dim=1)
        self.assertEqual(row_max.values.shape, (64,))
        self.assertEqual(row_max.indices.shape, (64,))
        self.assertEqual(row_max.values.device.type, "rbln")
        self.assertEqual(row_max.indices.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_matmul_and_at_operator_match(self, dtype):
        """`torch.matmul(a, b)` and `a @ b` should produce the same result."""
        a = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        b = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)

        z = torch.matmul(a, b)
        z2 = a @ b

        self.assertEqual(z.shape, (64, 64))
        self.assertEqual(z2.shape, (64, 64))
        self.assertEqual(z.device.type, "rbln")
        torch.testing.assert_close(z, z2)

    @dtypes(*SUPPORTED_DTYPES)
    def test_matmul_with_broadcast_bias(self, dtype):
        """`matmul(a, b) + bias` with a broadcastable bias should stay on RBLN."""
        a = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        b = torch.randn(64, 64, device=self.rbln_device, dtype=dtype)
        bias = torch.randn(64, device=self.rbln_device, dtype=dtype)

        out = torch.matmul(a, b) + bias

        self.assertEqual(out.shape, (64, 64))
        self.assertEqual(out.device.type, "rbln")


instantiate_device_type_tests(TestEagerMode, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
