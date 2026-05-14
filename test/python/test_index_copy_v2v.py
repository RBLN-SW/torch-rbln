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

import os


os.environ.setdefault("TORCH_RBLN_EAGER_MALLOC", "1")
os.environ.setdefault("TORCH_RBLN_DEPLOY", "ON")

import pytest
import torch

import torch_rbln  # noqa: F401


DEVICE = torch.device("rbln:0")

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64]


def _to_dev(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, device=DEVICE)
    out.copy_(x)
    return out


def _eq(actual_dev: torch.Tensor, expected_cpu: torch.Tensor):
    actual_cpu = actual_dev.cpu()
    assert actual_cpu.shape == expected_cpu.shape, (
        f"shape mismatch: device={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)}"
    )
    assert actual_cpu.dtype == expected_cpu.dtype, (
        f"dtype mismatch: device={actual_cpu.dtype} expected={expected_cpu.dtype}"
    )
    assert torch.equal(actual_cpu, expected_cpu), f"bitwise mismatch:\n  device={actual_cpu}\n  expected={expected_cpu}"


def _arange(shape, dtype):
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=dtype).reshape(shape) if shape else torch.tensor(0, dtype=dtype)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_basic_2d(dtype, axis):
    self_cpu = _arange((5, 4), dtype)
    src_cpu = _arange((2, 4), dtype) + 100 if axis == 0 else _arange((5, 2), dtype) + 100
    idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(axis, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), axis, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_basic_3d(dtype, axis):
    self_cpu = _arange((3, 4, 5), dtype)
    src_shape = list(self_cpu.shape)
    src_shape[axis] = 2
    src_cpu = _arange(tuple(src_shape), dtype) + 1000
    idx_cpu = torch.tensor([0, 2], dtype=torch.int64) if axis < 2 else torch.tensor([1, 3], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(axis, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), axis, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_4d_axis_negative():
    """axis=-2 on a rank-4 tensor."""
    self_cpu = _arange((2, 3, 4, 5), torch.float32)
    src_cpu = _arange((2, 3, 2, 5), torch.float32) + 1000
    idx_cpu = torch.tensor([0, 3], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(-2, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), -2, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


# ---------------------------------------------------------------------------
# Index patterns
# ---------------------------------------------------------------------------


def test_consecutive_run_coalesces():
    """One contiguous run → single v2v slab inside the engine."""
    self_cpu = _arange((10, 4), torch.float16)
    src_cpu = _arange((4, 4), torch.float16) + 1000
    idx_cpu = torch.tensor([2, 3, 4, 5], dtype=torch.int64)  # one run
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_scattered_indices():
    self_cpu = _arange((8, 3), torch.float32)
    src_cpu = _arange((5, 3), torch.float32) + 1000
    idx_cpu = torch.tensor([7, 0, 4, 2, 1], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_multiple_runs():
    """Mixed: two consecutive runs plus a scattered tail."""
    self_cpu = _arange((10, 4), torch.int32)
    src_cpu = _arange((6, 4), torch.int32) + 1000
    idx_cpu = torch.tensor([0, 1, 2, 5, 6, 9], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_duplicate_indices_last_write_wins():
    """PyTorch leaves duplicate-index behaviour implementation-defined; both
    upstream CPU and our v2v path apply runs in order so the LAST write wins.
    Test that our result matches the CPU reference."""
    self_cpu = _arange((6, 3), torch.float32)
    src_cpu = _arange((4, 3), torch.float32) + 1000
    # index[0] = index[3] = 2 — last write to row 2 wins on both sides
    idx_cpu = torch.tensor([2, 4, 5, 2], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_single_index():
    self_cpu = _arange((5, 4), torch.float16)
    src_cpu = _arange((1, 4), torch.float16) + 500
    idx_cpu = torch.tensor([3], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_full_replacement():
    """index covers every row in order — out should be byte-equal to source."""
    n, c = 4, 3
    self_cpu = _arange((n, c), torch.int64)
    src_cpu = _arange((n, c), torch.int64) + 1000
    idx_cpu = torch.arange(n, dtype=torch.int64)
    expected = src_cpu.clone()
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_boundary_indices():
    """First and last valid index values."""
    self_cpu = _arange((8, 2), torch.float32)
    src_cpu = _arange((2, 2), torch.float32) + 999
    idx_cpu = torch.tensor([0, 7], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


# ---------------------------------------------------------------------------
# In-place vs out-of-place
# ---------------------------------------------------------------------------


def test_in_place_index_copy():
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


def test_in_place_returns_same_tensor():
    """The in-place form must return self (same Python object semantics)."""
    self_dev = _to_dev(_arange((4, 3), torch.float32))
    idx = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    src = _to_dev(_arange((1, 3), torch.float32) + 100)
    ret = self_dev.index_copy_(0, idx, src)
    assert ret is self_dev


def test_out_with_existing_data_is_overwritten():
    """`out` already has data — it must be overwritten by self + indexed updates,
    not OR'd or accumulated."""
    self_cpu = _arange((4, 3), torch.float32)
    src_cpu = _arange((2, 3), torch.float32) + 100
    idx_cpu = torch.tensor([0, 2], dtype=torch.int64)

    out_dev = torch.full((4, 3), -1.0, dtype=torch.float32, device=DEVICE)
    torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu), out=out_dev)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    _eq(out_dev, expected)


# ---------------------------------------------------------------------------
# Stride combinations
# ---------------------------------------------------------------------------


def test_non_contig_self():
    """self is a non-contig view (column slice of larger tensor)."""
    big = _arange((6, 12), torch.float16)
    self_cpu = big[:, :8]
    assert not self_cpu.is_contiguous()
    src_cpu = _arange((2, 8), torch.float16) + 1000
    idx_cpu = torch.tensor([0, 4], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(big)[:, :8], 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_non_contig_source():
    """source is a non-contig view."""
    big_src = _arange((4, 16), torch.float32)
    src_cpu = big_src[:, :8]
    assert not src_cpu.is_contiguous()
    self_cpu = _arange((6, 8), torch.float32) + 999
    idx_cpu = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(big_src)[:, :8])
    _eq(got, expected)


def test_both_non_contig():
    big_self = _arange((6, 12), torch.int32)
    self_cpu = big_self[:, :8]
    big_src = _arange((4, 16), torch.int32) + 1000
    src_cpu = big_src[:, :8]
    idx_cpu = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(big_self)[:, :8], 0, _to_dev(idx_cpu), _to_dev(big_src)[:, :8])
    _eq(got, expected)


def test_non_contig_out_staging():
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


def test_axis_non_inner_stride():
    """index_copy along axis 0 of a tensor whose axis-0 elements are NOT contiguous
    in memory — runs cannot coalesce into a single v2v call."""
    big = _arange((6, 12), torch.float32)
    self_cpu = big[:, ::2]  # (6, 6) but inner stride 2
    src_cpu = _arange((3, 6), torch.float32) + 1000
    idx_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)  # consecutive run
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(big)[:, ::2], 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


# ---------------------------------------------------------------------------
# 0-D and empty edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, -1])
def test_zero_d_self(dim):
    self_cpu = torch.tensor(42, dtype=torch.float32)
    src_cpu = torch.tensor(100, dtype=torch.float32)
    idx_cpu = torch.tensor([0], dtype=torch.int64)
    expected = src_cpu  # the single element gets replaced
    got = torch.index_copy(_to_dev(self_cpu), dim, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
def test_empty_index(dtype):
    """Empty index → out is byte-equal to self."""
    self_cpu = _arange((5, 4), dtype)
    src_cpu = torch.empty(0, 4, dtype=dtype)
    idx_cpu = torch.tensor([], dtype=torch.int64)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, self_cpu)


def test_empty_self_along_dim():
    """self has 0 along some non-dim axis — op valid, out is also empty."""
    self_cpu = torch.empty((5, 0), dtype=torch.float32)
    src_cpu = torch.empty((2, 0), dtype=torch.float32)
    idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, self_cpu)


# ---------------------------------------------------------------------------
# Index dtype / device variations
# ---------------------------------------------------------------------------


def test_index_dtype_int64():
    self_cpu = _arange((5, 3), torch.float32)
    src_cpu = _arange((2, 3), torch.float32) + 100
    idx_cpu = torch.tensor([1, 3], dtype=torch.int64)
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


def test_index_dtype_int32_rejected_upstream():
    """index_copy upstream is strict about index dtype — it accepts only int64
    (unlike index_select). We don't need to special-case this on our side;
    PyTorch's meta function raises before reaching the RBLN kernel."""
    self_dev = _to_dev(_arange((5, 3), torch.float32))
    src_dev = _to_dev(_arange((2, 3), torch.float32))
    idx_int32 = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError, match="long tensor for index"):
        torch.index_copy(self_dev, 0, idx_int32, src_dev)


def test_index_cross_device_rejected_upstream():
    """index_copy upstream enforces same-device for self/index/source — unlike
    index_select, an index on a different device is rejected before reaching
    the kernel."""
    self_dev = _to_dev(_arange((5, 3), torch.float16))
    src_dev = _to_dev(_arange((2, 3), torch.float16))
    idx_cpu = torch.tensor([0, 4], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="same device"):
        torch.index_copy(self_dev, 0, idx_cpu, src_dev)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_index_out_of_range_rejected():
    self_dev = _to_dev(_arange((5, 3), torch.float32))
    src_dev = _to_dev(_arange((1, 3), torch.float32))
    idx_dev = torch.tensor([10], dtype=torch.int64, device=DEVICE)
    with pytest.raises(Exception, match="out of range"):
        torch.index_copy(self_dev, 0, idx_dev, src_dev)


def test_dim_out_of_range_rejected():
    self_dev = _to_dev(_arange((3, 4), torch.float32))
    src_dev = _to_dev(_arange((1, 4), torch.float32))
    idx_dev = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    with pytest.raises(Exception, match="dim|range"):
        torch.index_copy(self_dev, 5, idx_dev, src_dev)


def test_source_size_mismatch_rejected():
    self_dev = _to_dev(_arange((4, 3), torch.float32))
    bad_src = _to_dev(_arange((1, 4), torch.float32))  # wrong size on non-dim axis
    idx_dev = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    with pytest.raises(Exception):
        torch.index_copy(self_dev, 0, idx_dev, bad_src)


# ---------------------------------------------------------------------------
# Large / stress
# ---------------------------------------------------------------------------


def test_large_consecutive_run():
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


def test_many_scattered_indices():
    """Many scattered indices — exercises K v2v calls inside one V2VBatch."""
    n, c = 256, 8
    n_idx = 64
    self_cpu = _arange((n, c), torch.float32)
    src_cpu = _arange((n_idx, c), torch.float32) + 10000
    # Reverse order — no coalesce possible
    idx_cpu = torch.arange(n_idx - 1, -1, -1, dtype=torch.int64) + 10
    expected = self_cpu.clone().index_copy_(0, idx_cpu, src_cpu)
    got = torch.index_copy(_to_dev(self_cpu), 0, _to_dev(idx_cpu), _to_dev(src_cpu))
    _eq(got, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
