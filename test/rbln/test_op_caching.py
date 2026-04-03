# Owner(s): ["module: PrivateUse1"]

"""
Test suite for verifying operator caching behavior in various scenarios.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestOpCaching(TestCase):
    rbln_device = torch.device("rbln:0")
    shapes = [(2, 16), (2, 64)]

    def _reset_dynamo_counters(self):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 0)

    def _create_custom_float16_rbln_tensor(self, fp16_cpu_tensor, device):
        self.assertEqual(fp16_cpu_tensor.device.type, "cpu")
        self.assertEqual(fp16_cpu_tensor.dtype, torch.float16)

        fp16_rbln_tensor = fp16_cpu_tensor.to(device)
        # Run device op to get custom float16 tensor.
        # Use neg twice to avoid changing the value.
        cf16_rbln_tensor = torch.neg(torch.neg(fp16_rbln_tensor))
        return cf16_rbln_tensor

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", shapes)
    def test_same_input(self, dtype, shape):
        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_dynamo_counters()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        new_rbln_out = torch.abs(rbln_tensor)  # No recompilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertEqual(rbln_out, cpu_out)
        self.assertEqual(new_rbln_out, cpu_out)

    @dtypes(*SUPPORTED_DTYPES)
    def test_different_shape_recompilation(self, dtype):
        shape = (2, 16)

        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        different_shape = (4, 16)
        self.assertNotEqual(different_shape, shape)
        different_shape_cpu_tensor = torch.randn(different_shape, dtype=dtype, device="cpu")
        different_shape_cpu_out = torch.abs(different_shape_cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)
        different_shape_rbln_tensor = different_shape_cpu_tensor.to(self.rbln_device)

        self._reset_dynamo_counters()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertNotEqual(rbln_tensor.size(), different_shape_rbln_tensor.size())
        different_shape_rbln_out = torch.abs(different_shape_rbln_tensor)  # Recompilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 2)

        self.assertEqual(rbln_out, cpu_out)
        self.assertEqual(different_shape_rbln_out, different_shape_cpu_out)

    @parametrize("shape", shapes)
    def test_custom_float16_reuse(self, shape):
        fp16_cpu_tensor = torch.randn(shape, dtype=torch.float16, device="cpu")
        cpu_out = torch.abs(fp16_cpu_tensor)

        fp16_rbln_tensor = fp16_cpu_tensor.to(self.rbln_device)
        cf16_rbln_tensor = self._create_custom_float16_rbln_tensor(fp16_cpu_tensor, self.rbln_device)

        self._reset_dynamo_counters()

        rbln_out = torch.abs(fp16_rbln_tensor)  # Initial compilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertEqual(fp16_rbln_tensor.dtype, cf16_rbln_tensor.dtype)
        self.assertEqual(fp16_rbln_tensor.size(), cf16_rbln_tensor.size())
        self.assertEqual(fp16_rbln_tensor.stride(), cf16_rbln_tensor.stride())
        self.assertEqual(fp16_rbln_tensor.storage_offset(), cf16_rbln_tensor.storage_offset())
        new_rbln_out = torch.abs(cf16_rbln_tensor)  # No recompilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertEqual(rbln_out, cpu_out)
        self.assertEqual(new_rbln_out, cpu_out)


instantiate_device_type_tests(TestOpCaching, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
