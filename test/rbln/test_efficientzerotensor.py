# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `_efficientzerotensor` factory.

`aten::_efficientzerotensor` is a "logically zero" tensor that PyTorch may
produce as a backward-pass scratch and as the reference value for ops like
`aten::sgn` (which decomposes into `_efficientzerotensor` plus a `where`).
Before the V4 native registration this op fell through to the CPU-fallback
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
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestEfficientZeroTensor(TestCase):
    """`aten::_efficientzerotensor` should produce a logically-zero RBLN tensor."""

    def test_shape_and_dtype_preserved_float16(self) -> None:
        t = torch.ops.aten._efficientzerotensor((3, 4), dtype=torch.float16, device=torch.device("rbln"))
        self.assertEqual(t.device.type, "rbln")
        self.assertEqual(t.dtype, torch.float16)
        self.assertEqual(tuple(t.shape), (3, 4))

    def test_shape_and_dtype_preserved_int64(self) -> None:
        t = torch.ops.aten._efficientzerotensor((2,), dtype=torch.int64, device=torch.device("rbln"))
        self.assertEqual(t.device.type, "rbln")
        self.assertEqual(t.dtype, torch.int64)
        self.assertEqual(tuple(t.shape), (2,))

    def test_zero_init_value_float16(self) -> None:
        t = torch.ops.aten._efficientzerotensor((2, 3), dtype=torch.float16, device=torch.device("rbln"))
        # Materialise to CPU and check elementwise equality with a freshly-
        # allocated zero tensor of the same shape and dtype.
        expected = torch.zeros(2, 3, dtype=torch.float16)
        self.assertEqual(t.to("cpu"), expected)

    def test_zero_init_value_int64(self) -> None:
        t = torch.ops.aten._efficientzerotensor((4,), dtype=torch.int64, device=torch.device("rbln"))
        expected = torch.zeros(4, dtype=torch.int64)
        self.assertEqual(t.to("cpu"), expected)


@pytest.mark.test_set_ci
class TestSgnDecomposition(TestCase):
    """`aten::sgn` decomposes through `_efficientzerotensor`; verify safety + value."""

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


if __name__ == "__main__":
    run_tests()
