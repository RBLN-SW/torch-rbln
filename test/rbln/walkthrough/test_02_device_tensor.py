# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 2: tensor creation and movement on RBLN.

Ported from walkthrough_guide/2.rbln_device_tensor.py.

Verifies that:
- Tensors can be created directly on RBLN with expected shape/dtype.
- Factory ops (`zeros`, `empty_like`, `randn`) succeed on RBLN.
- CPU -> RBLN -> CPU round-trip preserves values.
- `.to(device, dtype=...)` moves and casts in one call.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestRBLNDeviceTensor(TestCase):
    """Walkthrough example 2: tensor creation/movement on RBLN."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_randn_on_rbln(self, dtype):
        """`torch.randn` should create a tensor directly on RBLN with the requested shape and dtype."""
        x = torch.randn(2, 64, device=self.rbln_device, dtype=dtype)
        self.assertEqual(x.device.type, "rbln")
        self.assertEqual(x.shape, (2, 64))
        self.assertEqual(x.dtype, dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_empty_like_on_rbln(self, dtype):
        """`torch.empty_like` should preserve shape and produce an RBLN tensor."""
        x = torch.randn(2, 64, device=self.rbln_device, dtype=dtype)
        e = torch.empty_like(x, device=self.rbln_device)
        self.assertEqual(e.device.type, "rbln")
        self.assertEqual(e.shape, x.shape)

    @dtypes(*SUPPORTED_DTYPES)
    def test_zeros_on_rbln(self, dtype):
        """`torch.zeros` should produce an RBLN tensor with the requested shape."""
        z = torch.zeros(2, 64, device=self.rbln_device, dtype=dtype)
        self.assertEqual(z.device.type, "rbln")
        self.assertEqual(z.shape, (2, 64))
        self.assertEqual(z.dtype, dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_cpu_to_rbln_to_cpu_round_trip(self, dtype):
        """A CPU tensor moved to RBLN and back should equal the original."""
        cpu_tensor = torch.randn(2, 64, device="cpu", dtype=dtype)
        rbln_tensor = cpu_tensor.to("rbln")
        back_to_cpu = rbln_tensor.to("cpu")

        self.assertEqual(rbln_tensor.device.type, "rbln")
        self.assertEqual(back_to_cpu.device.type, "cpu")
        torch.testing.assert_close(back_to_cpu, cpu_tensor)

    @dtypes(*SUPPORTED_DTYPES)
    def test_to_with_device_and_dtype(self, dtype):
        """`.to(device, dtype=...)` should move and cast in a single call."""
        t = torch.randn(3, 3, device="cpu", dtype=torch.float32)
        t_rbln = t.to(self.rbln_device, dtype=dtype)
        self.assertEqual(t_rbln.device.type, "rbln")
        self.assertEqual(t_rbln.device.index, 0)
        self.assertEqual(t_rbln.dtype, dtype)


instantiate_device_type_tests(TestRBLNDeviceTensor, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
