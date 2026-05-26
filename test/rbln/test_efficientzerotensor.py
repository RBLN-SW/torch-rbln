# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `_efficientzerotensor` factory.

`aten::_efficientzerotensor` is a "logically zero" tensor that PyTorch may
produce as a backward-pass scratch and as the reference value for ops like
`aten::sgn` (which decomposes into `_efficientzerotensor` plus a `where`).
Before the C++ native registration this op fell through to the CPU-fallback
generic path, where the tensor surfaced into `at::sgn` with an empty / non-
materialised storage and tripped a SIGSEGV inside the rebel runtime.

These tests verify the factory's core invariants on the RBLN device:

* Returns a tensor with the requested shape/dtype on `rbln`.
* Reads as zero everywhere (semantically the same as `torch.zeros`).
* `torch.sgn` (which internally relies on `_efficientzerotensor`) runs to
  completion on RBLN without segfault and matches the CPU result.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@pytest.mark.test_set_ci
class TestEfficientZeroTensor(TestCase):
    """`aten::_efficientzerotensor` should produce a logically-zero RBLN tensor.

    The op is dtype-agnostic at the C++ level (it just allocates and
    zero-marks v-memory) but we keep an explicit fp16 + int64 cross because
    those are the two dtypes that actually reach the op on real workloads
    (fp16 KV scratch, int64 indexing scratch). ``SUPPORTED_DTYPES`` alone
    would only cover fp16, so we add int64 as an explicit ``@parametrize``.
    """

    @parametrize("dtype", [torch.float16, torch.int64])
    @parametrize("shape", [(3, 4), (2,), (2, 3, 5)])
    def test_shape_and_dtype_preserved(self, dtype, shape) -> None:
        t = torch.ops.aten._efficientzerotensor(shape, dtype=dtype, device=torch.device("rbln"))
        self.assertEqual(t.device.type, "rbln")
        self.assertEqual(t.dtype, dtype)
        self.assertEqual(tuple(t.shape), shape)

    @parametrize("dtype", [torch.float16, torch.int64])
    def test_zero_init_value(self, dtype) -> None:
        # Materialise to CPU and check elementwise equality with a freshly-
        # allocated zero tensor of the same shape and dtype.
        t = torch.ops.aten._efficientzerotensor((2, 3), dtype=dtype, device=torch.device("rbln"))
        expected = torch.zeros(2, 3, dtype=dtype)
        self.assertEqual(t.to("cpu"), expected)


@pytest.mark.test_set_ci
class TestSgnDecomposition(TestCase):
    """`aten::sgn` decomposes through `_efficientzerotensor`; verify safety + value.

    Only fp16 is exercised here — sgn on integer dtypes routes through a
    different decomposition that doesn't depend on ``_efficientzerotensor``,
    which is the regression surface this suite is meant to guard.
    """

    def test_sgn_matches_cpu_float16(self) -> None:
        # Uses both negative, zero, and positive entries to exercise all
        # branches of sgn's decomposition (where over zero/nonzero).
        x_cpu = torch.tensor([-2.5, 0.0, 1.5, -0.0, 4.0], dtype=torch.float16)
        x_rbln = x_cpu.to("rbln")

        out_rbln = torch.sgn(x_rbln)
        out_cpu = torch.sgn(x_cpu)

        self.assertEqual(out_rbln.device.type, "rbln")
        self.assertEqual(out_rbln.dtype, torch.float16)
        self.assertEqual(out_rbln.to("cpu"), out_cpu)

    def test_sgn_matches_cpu_zeros_only(self) -> None:
        x_cpu = torch.zeros(8, dtype=torch.float16)
        x_rbln = x_cpu.to("rbln")
        self.assertEqual(torch.sgn(x_rbln).to("cpu"), torch.sgn(x_cpu))


instantiate_device_type_tests(TestEfficientZeroTensor, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestSgnDecomposition, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
