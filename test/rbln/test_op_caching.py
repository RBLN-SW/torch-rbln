# Owner(s): ["module: PrivateUse1"]

"""
Test suite for verifying operator caching behavior in various scenarios.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


ATOL = 0.01
RTOL = 0.01


@pytest.mark.test_set_ci
class TestOpCaching(TestCase):
    rbln_device = torch.device("rbln:0")
    # Both shapes must be 64-elem-aligned on the last dim: the
    # ``compile_and_run_view_aware`` 64-alignment guard now routes
    # ``last_dim % 64 != 0`` calls through ``cpu_fallback_path``, which
    # never enters torch.compile and therefore never bumps the
    # ``unique_graphs`` counter these caching tests assert on.
    shapes = [(2, 64), (2, 128)]

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

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_different_shape_recompilation(self, dtype):
        # 64-elem-aligned last dim so both shapes traverse the device
        # compile path (see comment on ``shapes`` above).
        shape = (2, 64)

        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        different_shape = (4, 64)
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

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(different_shape_rbln_out, different_shape_cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", shapes)
    def test_no_recompilation_across_instances(self, dtype, shape):
        cpu_tensor = torch.randn(shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)

        rbln_tensor = cpu_tensor.to(self.rbln_device)
        new_rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_dynamo_counters()

        rbln_out = torch.abs(rbln_tensor)  # Initial compilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertEqual(new_rbln_tensor.dtype, rbln_tensor.dtype)
        self.assertEqual(new_rbln_tensor.size(), rbln_tensor.size())
        self.assertEqual(new_rbln_tensor.stride(), rbln_tensor.stride())
        self.assertEqual(new_rbln_tensor.storage_offset(), rbln_tensor.storage_offset())
        new_rbln_out = torch.abs(new_rbln_tensor)  # No recompilation
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 1)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @parametrize("shape", shapes)
    def test_custom_float16_reuse(self, shape):
        """A tensor produced by a device op carries the internal
        ``custom_float16`` representation; it must share the compiled graph
        with a freshly moved fp16 tensor (no recompilation). Sibling
        ``test_no_recompilation_across_instances`` only covers two identical
        ``.to(device)`` copies — it never exercises the fp16 ↔
        ``custom_float16`` guard equivalence this test locks in."""
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

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)
        self.assertEqual(new_rbln_out, cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_unaligned_shape_uses_cpu_fallback(self, dtype):
        """Sibling to ``test_same_input`` / ``test_different_shape_recompilation``:
        locks in the ``compile_and_run_view_aware`` 64-alignment guard.

        The caching tests above use 64-aligned shapes so the device path
        runs and exercises torch.compile's graph cache. This test inverts
        the contract: a shape whose last-dim is NOT a multiple of 64 must
        route through ``cpu_fallback_path`` and therefore must NOT increment
        ``unique_graphs`` (cpu_fallback never enters torch.compile). If the
        alignment guard regresses (e.g. accidentally short-circuits to
        always-false), this test catches it from the opposite direction
        than the device-path caching tests.

        Correctness on the value side is also asserted: the cpu_fallback
        result must match the upstream CPU result element-wise.
        """
        unaligned_shape = (2, 16)  # 16 % 64 != 0 → ``_last_dim_unaligned`` True
        cpu_tensor = torch.randn(unaligned_shape, dtype=dtype, device="cpu")
        cpu_out = torch.abs(cpu_tensor)
        rbln_tensor = cpu_tensor.to(self.rbln_device)

        self._reset_dynamo_counters()

        rbln_out = torch.abs(rbln_tensor)
        # cpu_fallback never triggers torch.compile → unique_graphs stays 0.
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 0)

        # Second call: same routing decision, still no compile.
        _ = torch.abs(rbln_tensor)
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 0)

        self.assertEqual(rbln_out, cpu_out, atol=ATOL, rtol=RTOL)


instantiate_device_type_tests(TestOpCaching, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
