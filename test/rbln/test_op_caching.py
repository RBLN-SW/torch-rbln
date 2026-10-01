# Owner(s): ["module: PrivateUse1"]

"""
Test suite for verifying operator caching behavior in various scenarios.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import SUPPORTED_DTYPES
from torch_rbln._internal import compile_cache, warm_cache


ATOL = 0.01
RTOL = 0.01


@pytest.mark.test_set_ci
class TestOpCaching(TestCase):
    rbln_device = torch.device("rbln:0")
    shapes = [(2, 64), (2, 128), (2, 16)]

    def _reset_caches(self):
        compile_cache.clear_rbln_compile_cache()
        warm_cache.clear()
        self.assertEqual(self._compiles(), 0)

    def _compiles(self):
        return len(compile_cache._compiled_op_cache)

    def _device_produced(self, cpu_tensor):
        # neg twice leaves the value as it was.
        return torch.neg(torch.neg(cpu_tensor.to(self.rbln_device)))

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", shapes)
    def test_same_input(self, dtype, shape):
        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_caches()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(self._compiles(), 1)

        new_rbln_out = torch.abs(rbln_tensor)  # No recompilation
        self.assertEqual(self._compiles(), 1)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_different_shape_recompilation(self, dtype):
        shape = (2, 64)

        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        different_shape = (4, 64)
        self.assertNotEqual(different_shape, shape)
        different_shape_cpu_tensor = torch.randn(different_shape, dtype=dtype, device="cpu")
        different_shape_cpu_out = torch.abs(different_shape_cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)
        different_shape_rbln_tensor = different_shape_cpu_tensor.to(self.rbln_device)

        self._reset_caches()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(self._compiles(), 1)

        self.assertNotEqual(rbln_tensor.size(), different_shape_rbln_tensor.size())
        different_shape_rbln_out = torch.abs(different_shape_rbln_tensor)  # Recompilation
        self.assertEqual(self._compiles(), 2)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(different_shape_rbln_out, different_shape_cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", shapes)
    def test_no_recompilation_across_instances(self, dtype, shape):
        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)
        new_rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_caches()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(self._compiles(), 1)

        self.assertEqual(new_rbln_tensor.dtype, rbln_tensor.dtype)
        self.assertEqual(new_rbln_tensor.size(), rbln_tensor.size())
        self.assertEqual(new_rbln_tensor.stride(), rbln_tensor.stride())
        self.assertEqual(new_rbln_tensor.storage_offset(), rbln_tensor.storage_offset())
        new_rbln_out = torch.abs(new_rbln_tensor)  # No recompilation
        self.assertEqual(self._compiles(), 1)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", shapes)
    def test_device_produced_input_reuse(self, dtype, shape):
        """A tensor a device op wrote shares the compiled function with one copied from
        the host."""
        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        copied = cpu_tensor.to(self.rbln_device)
        produced = self._device_produced(cpu_tensor)

        self._reset_caches()

        rbln_out = torch.abs(copied)  # Initial compilation
        self.assertEqual(self._compiles(), 1)

        self.assertEqual(copied.dtype, produced.dtype)
        self.assertEqual(copied.size(), produced.size())
        self.assertEqual(copied.stride(), produced.stride())
        self.assertEqual(copied.storage_offset(), produced.storage_offset())
        new_rbln_out = torch.abs(produced)  # No recompilation
        self.assertEqual(self._compiles(), 1)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_scalar_argument_is_part_of_the_profile(self, dtype):
        cpu_tensor = torch.randn(2, 64, dtype=dtype, device="cpu")
        rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_caches()

        out_two = torch.mul(rbln_tensor, 2.0)
        out_three = torch.mul(rbln_tensor, 3.0)
        self.assertEqual(self._compiles(), 2)
        torch.mul(rbln_tensor, 2.0)
        self.assertEqual(self._compiles(), 2)

        self.assertEqual(out_two, cpu_tensor * 2.0, atol=ATOL, rtol=RTOL)
        self.assertEqual(out_three, cpu_tensor * 3.0, atol=ATOL, rtol=RTOL)


instantiate_device_type_tests(TestOpCaching, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
