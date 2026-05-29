# Owner(s): ["module: PrivateUse1"]

"""Tests for the native `repeat_interleave.Tensor` registration and the
two composite forms that decompose through it (`self_int` and
`self_Tensor`).

The `.Tensor` form is the index-builder primitive: given a 1-D `repeats`
tensor, it returns the 1-D int64 index list where each `i` appears
`repeats[i]` times. The composite `self_*` forms build their gather indices
via `.Tensor` and then call `index_select`; both pieces are now native v2v
on RBLN, so the whole composite chain runs without a host round-trip of
`self`.

Coverage:
  - `.Tensor` form: basic, empty, zeros in repeats, large, int32 + int64,
    `output_size` validation
  - `.self_int` form (scalar repeats): basic, dim 0 / inner / negative,
    various dtypes, in-place pattern
  - `.self_Tensor` form (per-element repeats): basic, dim 0 / inner, mix
    of zero and positive counts
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import arange as _arange, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestRepeatInterleaveV2V(TestCase):
    """Tests for native ``aten::repeat_interleave.Tensor`` and the
    composite ``.self_int`` / ``.self_Tensor`` forms."""

    # ---- .Tensor (index-builder) primitive ----

    def test_tensor_basic(self):
        r_cpu = torch.tensor([3, 1, 2], dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_int32_input(self):
        r_cpu = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_empty(self):
        r_cpu = torch.tensor([], dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_zeros_interspersed(self):
        r_cpu = torch.tensor([2, 0, 3, 0, 1], dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_all_zeros(self):
        r_cpu = torch.zeros(5, dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_single_repeat(self):
        r_cpu = torch.tensor([7], dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_large(self):
        r_cpu = torch.arange(1, 101, dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu)
        got = torch.repeat_interleave(_to_dev(r_cpu))
        _eq(got, expected)

    def test_tensor_output_size_matches(self):
        r_cpu = torch.tensor([3, 1, 2], dtype=torch.int64)
        expected = torch.repeat_interleave(r_cpu, output_size=6)
        got = torch.repeat_interleave(_to_dev(r_cpu), output_size=6)
        _eq(got, expected)

    def test_tensor_output_size_mismatch_rejected(self):
        r_dev = _to_dev(torch.tensor([3, 1, 2], dtype=torch.int64))
        with pytest.raises(RuntimeError, match="output_size|allocated size"):
            torch.repeat_interleave(r_dev, output_size=99)

    def test_tensor_negative_repeat_rejected(self):
        r_dev = _to_dev(torch.tensor([2, -1, 3], dtype=torch.int64))
        with pytest.raises(RuntimeError, match="non-negative|can not be negative"):
            torch.repeat_interleave(r_dev)

    # ---- .self_int — scalar repeats (composite decomposes through .Tensor) ----

    @dtypes(*ENGINE_DTYPES)
    @parametrize("repeats", [1, 2, 3, 5])
    def test_self_int_dim0(self, dtype, repeats):
        self_cpu = _arange((4, 3), dtype) + 1
        expected = self_cpu.repeat_interleave(repeats, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(repeats, dim=0)
        _eq(got, expected)

    @dtypes(torch.float16, torch.int32)
    @parametrize("dim", [0, 1, -1])
    def test_self_int_arbitrary_dim(self, dtype, dim):
        self_cpu = _arange((2, 3, 4), dtype) + 1
        expected = self_cpu.repeat_interleave(2, dim=dim)
        got = _to_dev(self_cpu).repeat_interleave(2, dim=dim)
        _eq(got, expected)

    def test_self_int_dim_none_flattens(self):
        """No `dim` → output is flat (self.flatten() then repeated)."""
        self_cpu = _arange((2, 3), torch.float32) + 1
        expected = self_cpu.repeat_interleave(3)
        got = _to_dev(self_cpu).repeat_interleave(3)
        _eq(got, expected)

    def test_self_int_repeats_one_is_identity(self):
        """`repeat_interleave(1, dim=d)` returns a tensor equal to self."""
        self_cpu = _arange((4, 5), torch.float16)
        got = _to_dev(self_cpu).repeat_interleave(1, dim=0)
        _eq(got, self_cpu)

    def test_self_int_zero_repeats(self):
        self_cpu = _arange((3, 4), torch.float32)
        expected = self_cpu.repeat_interleave(0, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(0, dim=0)
        _eq(got, expected)

    # ---- .self_Tensor — per-element repeats ----

    @dtypes(torch.float16, torch.float32, torch.int64)
    def test_self_tensor_dim0(self, dtype):
        self_cpu = _arange((4, 3), dtype) + 1
        counts_cpu = torch.tensor([1, 3, 2, 1], dtype=torch.int64)
        expected = self_cpu.repeat_interleave(counts_cpu, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts_cpu), dim=0)
        _eq(got, expected)

    def test_self_tensor_inner_dim(self):
        self_cpu = _arange((2, 4), torch.float32) + 1
        counts_cpu = torch.tensor([2, 1, 0, 3], dtype=torch.int64)
        expected = self_cpu.repeat_interleave(counts_cpu, dim=1)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts_cpu), dim=1)
        _eq(got, expected)

    def test_self_tensor_with_zeros(self):
        """Zeros in `counts` mean "drop this row" — output is shorter."""
        self_cpu = _arange((5, 2), torch.int32)
        counts_cpu = torch.tensor([1, 0, 2, 0, 3], dtype=torch.int64)
        expected = self_cpu.repeat_interleave(counts_cpu, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts_cpu), dim=0)
        _eq(got, expected)

    def test_self_tensor_dim_none_flattens(self):
        self_cpu = _arange((2, 3), torch.float32)
        counts_cpu = torch.tensor([1, 2, 0, 3, 1, 2], dtype=torch.int64)
        expected = self_cpu.repeat_interleave(counts_cpu)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts_cpu))
        _eq(got, expected)

    def test_self_tensor_3d_axis_negative(self):
        self_cpu = _arange((2, 3, 4), torch.float16)
        counts_cpu = torch.tensor([1, 2, 0], dtype=torch.int64)
        expected = self_cpu.repeat_interleave(counts_cpu, dim=-2)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts_cpu), dim=-2)
        _eq(got, expected)

    # ---- stress / non-contig ----

    def test_self_int_non_contig_self(self):
        big = _arange((6, 12), torch.float32)
        self_cpu = big[:, :8]
        assert not self_cpu.is_contiguous()
        expected = self_cpu.repeat_interleave(2, dim=0)
        got = _to_dev(big)[:, :8].repeat_interleave(2, dim=0)
        _eq(got, expected)

    def test_self_tensor_large_batch_small_inner(self):
        """Batch-large pattern — counts ≪ batch_size for sparse selection."""
        self_cpu = _arange((128, 16), torch.float16) + 1
        counts = torch.zeros(128, dtype=torch.int64)
        counts[::4] = 2
        expected = self_cpu.repeat_interleave(counts, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(_to_dev(counts), dim=0)
        _eq(got, expected)

    def test_self_int_large_repeats_GQA_pattern(self):
        """GQA-style head expansion: (n_kv_heads, head_dim) repeated r=q/kv times."""
        n_kv, head_dim, r = 8, 128, 4
        self_cpu = _arange((n_kv, head_dim), torch.float16) + 1
        expected = self_cpu.repeat_interleave(r, dim=0)
        got = _to_dev(self_cpu).repeat_interleave(r, dim=0)
        _eq(got, expected)


instantiate_device_type_tests(TestRepeatInterleaveV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
