# Owner(s): ["module: PrivateUse1"]

"""Tests for the v2v-based index_copy.out kernel.

Coverage is organised by behaviour:
  - Basic correctness across dtypes / rank / axis
  - Index patterns: consecutive runs (coalesce), scattered, duplicates,
    empty, single element, in-bounds boundary values
  - Stride combinations: contig / non-contig self, source, out
  - In-place vs out-of-place (out aliasing self) — the in-place
    short-circuits the self → out initialisation
  - 0-D self, 0-numel self, source with zero index count
  - Index dtype (int32 vs int64) and index device (CPU vs RBLN)
  - Error paths: wrong shape, out-of-range index, mismatched dtype

Each test compares against PyTorch's CPU reference. Same-dtype copies are
checked bitwise; the dtype is always preserved by index_copy so no float
tolerance is needed.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import arange as _arange, DEVICE, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestIndexCopyV2V(TestCase):
    """Tests for the native v2v ``aten::index_copy.out`` kernel and the
    ``index_copy_`` / ``index_copy`` forms that dispatch through it."""

    # ---- basic correctness ----

    @dtypes(*ENGINE_DTYPES)
    @parametrize("axis", [0, 1, -1])
    def test_basic_2d(self, dtype, axis):
        self_cpu = _arange((5, 4), dtype)
        src_cpu = _arange((2, 4), dtype) + 100 if axis == 0 else _arange((5, 2), dtype) + 100
        idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(axis, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), axis, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    @dtypes(torch.float16, torch.int32)
    @parametrize("axis", [0, 1, 2])
    def test_basic_3d(self, dtype, axis):
        self_cpu = _arange((3, 4, 5), dtype)
        src_shape = list(self_cpu.shape)
        src_shape[axis] = 2
        src_cpu = _arange(tuple(src_shape), dtype) + 1000
        idx_cpu = torch.tensor([0, 2], dtype=torch.int64) if axis < 2 else torch.tensor([1, 3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(axis, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), axis, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_4d_axis_negative(self):
        """axis=-2 on a rank-4 tensor."""
        self_cpu = _arange((2, 3, 4, 5), torch.float32)
        src_cpu = _arange((2, 3, 2, 5), torch.float32) + 1000
        idx_cpu = torch.tensor([0, 3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(-2, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), -2, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    # ---- index patterns ----

    def test_consecutive_run_coalesces(self):
        """One contiguous run → single v2v slab inside the engine."""
        self_cpu = _arange((10, 4), torch.float16)
        src_cpu = _arange((4, 4), torch.float16) + 1000
        idx_cpu = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_scattered_indices(self):
        self_cpu = _arange((8, 3), torch.float32)
        src_cpu = _arange((5, 3), torch.float32) + 1000
        idx_cpu = torch.tensor([7, 0, 4, 2, 1], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_multiple_runs(self):
        """Mixed: two consecutive runs plus a scattered tail."""
        self_cpu = _arange((10, 4), torch.int32)
        src_cpu = _arange((6, 4), torch.int32) + 1000
        idx_cpu = torch.tensor([0, 1, 2, 5, 6, 9], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_duplicate_indices_last_write_wins(self):
        """PyTorch leaves duplicate-index behaviour implementation-defined; both
        upstream CPU and our v2v path apply runs in order so the LAST write wins.
        Test that our result matches the CPU reference."""
        self_cpu = _arange((6, 3), torch.float32)
        src_cpu = _arange((4, 3), torch.float32) + 1000
        idx_cpu = torch.tensor([2, 4, 5, 2], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_single_index(self):
        self_cpu = _arange((5, 4), torch.float16)
        src_cpu = _arange((1, 4), torch.float16) + 500
        idx_cpu = torch.tensor([3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_full_replacement(self):
        """index covers every row in order — out should be byte-equal to source."""
        n, c = 4, 3
        self_cpu = _arange((n, c), torch.int64)
        src_cpu = _arange((n, c), torch.int64) + 1000
        idx_cpu = torch.arange(n, dtype=torch.int64)
        expected = src_cpu.clone()
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_boundary_indices(self):
        """First and last valid index values."""
        self_cpu = _arange((8, 2), torch.float32)
        src_cpu = _arange((2, 2), torch.float32) + 999
        idx_cpu = torch.tensor([0, 7], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    # ---- in-place vs out-of-place ----

    def test_in_place_index_copy(self):
        """tensor.index_copy_(...) dispatches with out aliasing self — the kernel
        must skip the self → out copy and write directly."""
        self_cpu = _arange((6, 4), torch.float16)
        src_cpu = _arange((2, 4), torch.float16) + 100
        idx_cpu = torch.tensor([1, 4], dtype=torch.int64)

        self_dev = _to_dev(self_cpu)
        self_dev.index_copy_(0, _to_dev(idx_cpu), _to_dev(src_cpu))

        expected = self_cpu.clone()
        expected.index_copy_(0, idx_cpu, src_cpu)
        _eq(self_dev, expected)

    def test_in_place_returns_same_tensor(self):
        """The in-place form must return self (same Python object semantics)."""
        self_dev = _to_dev(_arange((4, 3), torch.float32))
        idx = torch.tensor([0], dtype=torch.int64, device=DEVICE)
        src = _to_dev(_arange((1, 3), torch.float32) + 100)
        ret = self_dev.index_copy_(0, idx, src)
        assert ret is self_dev

    def test_out_with_existing_data_is_overwritten(self):
        """`out` already has data — it must be overwritten by self + indexed updates,
        not OR'd or accumulated."""
        self_cpu = _arange((4, 3), torch.float32)
        src_cpu = _arange((2, 3), torch.float32) + 100
        idx_cpu = torch.tensor([0, 2], dtype=torch.int64)

        out_dev = torch.full((4, 3), -1.0, dtype=torch.float32, device=DEVICE)
        torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu), out=out_dev)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        _eq(out_dev, expected)

    # ---- stride combinations ----

    def test_non_contig_self(self):
        """self is a non-contig view (column slice of larger tensor)."""
        big = _arange((6, 12), torch.float16)
        self_cpu = big[:, :8]
        assert not self_cpu.is_contiguous()
        src_cpu = _arange((2, 8), torch.float16) + 1000
        idx_cpu = torch.tensor([0, 4], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(big)[:, :8], 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_non_contig_source(self):
        """source is a non-contig view."""
        big_src = _arange((4, 16), torch.float32)
        src_cpu = big_src[:, :8]
        assert not src_cpu.is_contiguous()
        self_cpu = _arange((6, 8), torch.float32) + 999
        idx_cpu = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(big_src)[:, :8])
        _eq(got, expected)

    def test_both_non_contig(self):
        big_self = _arange((6, 12), torch.int32)
        self_cpu = big_self[:, :8]
        big_src = _arange((4, 16), torch.int32) + 1000
        src_cpu = big_src[:, :8]
        idx_cpu = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(big_self)[:, :8], 0, _to_dev(idx_cpu), _to_dev(big_src)[:, :8])
        _eq(got, expected)

    def test_non_contig_out_staging(self):
        """out is a non-contig view — kernel stages through contig then copies back."""
        big_out = torch.zeros((4, 12), dtype=torch.float16, device=DEVICE)
        out_view = big_out[:, :8]
        assert not out_view.is_contiguous()

        self_cpu = _arange((4, 8), torch.float16)
        src_cpu = _arange((2, 8), torch.float16) + 100
        idx_cpu = torch.tensor([1, 3], dtype=torch.int64)

        torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu), out=out_view)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        _eq(out_view, expected)

    def test_axis_non_inner_stride(self):
        """index_copy along axis 0 of a tensor whose axis-0 elements are NOT contiguous
        in memory — runs cannot coalesce into a single v2v call."""
        big = _arange((6, 12), torch.float32)
        self_cpu = big[:, ::2]
        src_cpu = _arange((3, 6), torch.float32) + 1000
        idx_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(big)[:, ::2], 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    # ---- 0-D and empty edge cases ----

    @parametrize("dim", [0, -1])
    def test_zero_d_self(self, dim):
        self_cpu = torch.tensor(42, dtype=torch.float32)
        src_cpu = torch.tensor(100, dtype=torch.float32)
        idx_cpu = torch.tensor([0], dtype=torch.int64)
        expected = src_cpu
        got = torch.index_copy(_to_dev(self_cpu), dim, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_zero_d_self_one_d_source(self):
        """torch._refs.index_select on a 0-D input decomposes to
        ``empty_like(x).index_copy(0, idx, x.expand_as(idx))`` — 0-D self
        with a 1-D source whose every element equals x. Matches CPU's
        last-write-wins semantic."""
        self_cpu = torch.tensor(42, dtype=torch.float32)
        x = torch.tensor(7, dtype=torch.float32)
        idx_cpu = torch.tensor([0, 0, 0], dtype=torch.int64)
        src_cpu = x.expand_as(idx_cpu).contiguous()
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_zero_d_self_one_d_source_distinct_values(self):
        """Non-broadcast 1-D source with distinct values on 0-D self: last
        write wins, so the result is source[-1]. PyTorch's index_copy meta
        admits this shape (self.dim()=0 or source.dim()=0 → mismatch OK)."""
        self_cpu = torch.tensor(42, dtype=torch.float32)
        src_cpu = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        idx_cpu = torch.tensor([0, 0, 0], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    @dtypes(torch.float16, torch.int64)
    def test_empty_index(self, dtype):
        """Empty index → out is byte-equal to self."""
        self_cpu = _arange((5, 4), dtype)
        src_cpu = torch.empty(0, 4, dtype=dtype)
        idx_cpu = torch.tensor([], dtype=torch.int64)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, self_cpu)

    def test_empty_self_along_dim(self):
        """self has 0 along some non-dim axis — op valid, out is also empty."""
        self_cpu = torch.empty((5, 0), dtype=torch.float32)
        src_cpu = torch.empty((2, 0), dtype=torch.float32)
        idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, self_cpu)

    # ---- index dtype / device variations ----

    def test_index_dtype_int64(self):
        self_cpu = _arange((5, 3), torch.float32)
        src_cpu = _arange((2, 3), torch.float32) + 100
        idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_index_dtype_int32_rejected_upstream(self):
        """index_copy upstream is strict about index dtype — it accepts only int64
        (unlike index_select). We don't need to special-case this on our side;
        PyTorch's meta function raises before reaching the RBLN kernel."""
        self_dev = _to_dev(_arange((5, 3), torch.float32))
        src_dev = _to_dev(_arange((2, 3), torch.float32))
        idx_int32 = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="long tensor for index"):
            torch.index_copy(self_dev, 0, idx_int32, src_dev)

    def test_index_cross_device_rejected_upstream(self):
        """index_copy upstream enforces same-device for self/index/source — unlike
        index_select, an index on a different device is rejected before reaching
        the kernel."""
        self_dev = _to_dev(_arange((5, 3), torch.float16))
        src_dev = _to_dev(_arange((2, 3), torch.float16))
        idx_cpu = torch.tensor([0, 4], dtype=torch.int64)
        with pytest.raises(RuntimeError, match="same device"):
            torch.index_copy(self_dev, 0, idx_cpu, src_dev)

    # ---- error paths ----

    def test_index_out_of_range_rejected(self):
        self_dev = _to_dev(_arange((5, 3), torch.float32))
        src_dev = _to_dev(_arange((1, 3), torch.float32))
        idx_dev = torch.tensor([10], dtype=torch.int64, device=DEVICE)
        with pytest.raises((IndexError, RuntimeError), match="out of range|out of bounds"):
            torch.index_copy(self_dev, 0, idx_dev, src_dev)

    def test_dim_out_of_range_rejected(self):
        self_dev = _to_dev(_arange((3, 4), torch.float32))
        src_dev = _to_dev(_arange((1, 4), torch.float32))
        idx_dev = torch.tensor([0], dtype=torch.int64, device=DEVICE)
        with pytest.raises((IndexError, RuntimeError), match="dim|range|bounds"):
            torch.index_copy(self_dev, 5, idx_dev, src_dev)

    def test_source_size_mismatch_rejected(self):
        self_dev = _to_dev(_arange((4, 3), torch.float32))
        bad_src = _to_dev(_arange((1, 4), torch.float32))
        idx_dev = torch.tensor([0], dtype=torch.int64, device=DEVICE)
        with pytest.raises(RuntimeError):
            torch.index_copy(self_dev, 0, idx_dev, bad_src)

    @dtypes(*ENGINE_DTYPES)
    def test_out_source_overlap_rejected(self, dtype):
        """``out`` and ``source`` sharing storage (``source`` is a slice of ``out``) must raise ``RuntimeError``."""
        self_ref = _to_dev(_arange((5,), dtype))
        idx = torch.tensor([0, 2], dtype=torch.long)
        out = _to_dev(_arange((5,), dtype))
        source = out[1:3]
        with pytest.raises(RuntimeError):
            torch.index_copy(self_ref, 0, idx, source, out=out)

    @dtypes(*ENGINE_DTYPES)
    def test_out_internal_overlap_rejected(self, dtype):
        """``out`` with internal overlap (broadcast view) must raise ``RuntimeError``."""
        self_ref = _to_dev(_arange((2, 4), dtype))
        source = _to_dev(_arange((1, 4), dtype))
        idx = torch.tensor([0], dtype=torch.long)
        out = torch.empty(1, 4, dtype=dtype, device=DEVICE).expand(2, 4)
        with pytest.raises(RuntimeError):
            torch.index_copy(self_ref, 0, idx, source, out=out)

    # ---- large / stress ----

    def test_large_consecutive_run(self):
        """Big tensor with one large consecutive run — the coalesce should issue
        one v2v of the whole run rather than per-row."""
        n, c = 1024, 128
        run_len = 256
        self_cpu = _arange((n, c), torch.float16)
        src_cpu = _arange((run_len, c), torch.float16) + 99
        idx_cpu = torch.arange(100, 100 + run_len, dtype=torch.int64)
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)

    def test_many_scattered_indices(self):
        """Many scattered indices — exercises K v2v calls inside one V2VBatch."""
        n, c = 256, 8
        n_idx = 64
        self_cpu = _arange((n, c), torch.float32)
        src_cpu = _arange((n_idx, c), torch.float32) + 10000
        idx_cpu = torch.arange(n_idx - 1, -1, -1, dtype=torch.int64) + 10
        expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
        got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
        _eq(got, expected)


instantiate_device_type_tests(TestIndexCopyV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
