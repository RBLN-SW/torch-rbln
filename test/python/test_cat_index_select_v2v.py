"""Tests for the C++ v2v-based cat / index_select kernels.

Each test allocates the input on rbln:0, runs the op on device, and compares
against the CPU reference. Coverage:
  - cat: contiguous & non-contiguous inputs, all axes, dtype mix, edge cases
  - index_select: contiguous & non-contig self, runs of consecutive indices,
    scattered indices, index on CPU vs index on RBLN, dim variants
  - stack: just verifies it routes through cat correctly
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_RBLN_EAGER_MALLOC", "1")
os.environ.setdefault("TORCH_RBLN_DEPLOY", "ON")

import pytest
import torch

import torch_rbln  # noqa: F401

DEVICE = torch.device("rbln:0")


def _to_dev(x: torch.Tensor) -> torch.Tensor:
    """Materialise a CPU tensor on rbln:0 with the same layout/dtype."""
    out = torch.empty_like(x, device=DEVICE)
    out.copy_(x)
    return out


def _check(actual_dev: torch.Tensor, expected_cpu: torch.Tensor, atol: float = 0.0, rtol: float = 0.0):
    actual_cpu = actual_dev.cpu()
    assert actual_cpu.shape == expected_cpu.shape, (
        f"shape mismatch: device={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)}"
    )
    assert actual_cpu.dtype == expected_cpu.dtype, (
        f"dtype mismatch: device={actual_cpu.dtype} expected={expected_cpu.dtype}"
    )
    if atol == 0.0 and rtol == 0.0:
        assert torch.equal(actual_cpu, expected_cpu), (
            f"bitwise mismatch\n  device={actual_cpu}\n  expected={expected_cpu}"
        )
    else:
        torch.testing.assert_close(actual_cpu, expected_cpu, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# cat tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64])
def test_cat_2input_axis0_contig(dtype):
    a_cpu = torch.arange(24, dtype=dtype).reshape(3, 8)
    b_cpu = torch.arange(24, 48, dtype=dtype).reshape(3, 8)
    expected = torch.cat([a_cpu, b_cpu], dim=0)
    got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=0)
    _check(got, expected)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_cat_3d_all_axes(axis):
    a_cpu = torch.arange(2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
    b_cpu = torch.arange(100, 100 + 2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
    expected = torch.cat([a_cpu, b_cpu], dim=axis)
    got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=axis)
    _check(got, expected)


def test_cat_many_inputs_axis0():
    """28 inputs (like test.py's stack(k_layers) decomposition)."""
    n = 28
    inputs_cpu = [torch.full((1, 4, 8, 16), float(i), dtype=torch.bfloat16) for i in range(n)]
    expected = torch.cat(inputs_cpu, dim=0)
    inputs_dev = [_to_dev(x) for x in inputs_cpu]
    got = torch.cat(inputs_dev, dim=0)
    _check(got, expected)


def test_cat_single_input():
    a_cpu = torch.randn(3, 5, dtype=torch.float32)
    expected = torch.cat([a_cpu], dim=0)
    got = torch.cat([_to_dev(a_cpu)], dim=0)
    _check(got, expected)


def test_cat_with_some_empty_inputs():
    """PyTorch ignores empty inputs that have numel == 0."""
    a_cpu = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    empty_cpu = torch.zeros(0, 4, dtype=torch.float16)
    b_cpu = torch.arange(20, 32, dtype=torch.float16).reshape(3, 4)
    expected = torch.cat([a_cpu, empty_cpu, b_cpu], dim=0)
    got = torch.cat([_to_dev(a_cpu), _to_dev(empty_cpu), _to_dev(b_cpu)], dim=0)
    _check(got, expected)


def test_cat_all_empty_inputs():
    """When every input is empty, output is empty (default behaviour for the
    pre-allocated `out` tensor: our kernel returns it unchanged)."""
    empties_cpu = [torch.zeros(0, 4, dtype=torch.float16) for _ in range(3)]
    expected = torch.cat(empties_cpu, dim=0)
    got = torch.cat([_to_dev(x) for x in empties_cpu], dim=0)
    _check(got, expected)


def test_cat_non_contig_input_unsqueezed_slice():
    """The exact pattern from test.py: kv[0, blk, :, 0, :, :] slices that are
    actually contiguous on inspection, then unsqueeze(0) for stack-decomposition."""
    kv_cpu = torch.arange(2 * 4 * 8 * 1 * 16 * 8, dtype=torch.bfloat16).reshape(2, 4, 8, 1, 16, 8)
    kv = _to_dev(kv_cpu)
    layers_cpu = [kv_cpu[0, blk, :, 0, :, :].unsqueeze(0) for blk in range(4)]
    layers_dev = [kv[0, blk, :, 0, :, :].unsqueeze(0) for blk in range(4)]
    expected = torch.cat(layers_cpu, dim=0)
    got = torch.cat(layers_dev, dim=0)
    _check(got, expected)


def test_cat_genuinely_non_contig_input():
    """A non-contig view that cannot be coalesced fully: kv[0, :, :, 0, ::2, :]
    has stride hole at the second-to-last dim."""
    kv_cpu = torch.arange(2 * 4 * 8 * 1 * 16 * 8, dtype=torch.float32).reshape(2, 4, 8, 1, 16, 8)
    kv = _to_dev(kv_cpu)
    view_cpu = kv_cpu[0, :, :, 0, ::2, :]  # (4, 8, 8, 8) with non-trivial stride
    view_dev = kv[0, :, :, 0, ::2, :]
    assert not view_dev.is_contiguous(), "expected non-contig view"
    expected = torch.cat([view_cpu, view_cpu], dim=0)
    got = torch.cat([view_dev, view_dev], dim=0)
    _check(got, expected)


def test_cat_axis_last_dim():
    a_cpu = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
    b_cpu = torch.arange(30, dtype=torch.float16).reshape(2, 3, 5)
    expected = torch.cat([a_cpu, b_cpu], dim=-1)
    got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=-1)
    _check(got, expected)


def test_cat_mixed_sizes_at_axis():
    """Different shape[axis] per input (the common case)."""
    a_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b_cpu = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    c_cpu = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected = torch.cat([a_cpu, b_cpu, c_cpu], dim=1)
    got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu), _to_dev(c_cpu)], dim=1)
    _check(got, expected)


# ---------------------------------------------------------------------------
# stack tests (CompositeImplicitAutograd → cat)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_stack_decomposes_to_cat(dim):
    inputs_cpu = [torch.randn(3, 4, dtype=torch.float32) for _ in range(5)]
    expected = torch.stack(inputs_cpu, dim=dim)
    got = torch.stack([_to_dev(x) for x in inputs_cpu], dim=dim)
    _check(got, expected)


def test_stack_28_layers_bf16():
    """The test.py shape pattern."""
    layers_cpu = [torch.randn(4, 8, 16, dtype=torch.bfloat16) for _ in range(28)]
    expected = torch.stack(layers_cpu, dim=0)
    got = torch.stack([_to_dev(x) for x in layers_cpu], dim=0)
    _check(got, expected)


# ---------------------------------------------------------------------------
# index_select tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32])
def test_index_select_basic_dim0(dtype):
    src_cpu = torch.arange(30, dtype=dtype).reshape(5, 6)
    idx_cpu = torch.tensor([3, 0, 4, 1], dtype=torch.long)
    expected = torch.index_select(src_cpu, 0, idx_cpu)
    got = torch.index_select(_to_dev(src_cpu), 0, idx_cpu)
    _check(got, expected)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_index_select_all_dims(axis):
    src_cpu = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    idx_cpu = torch.tensor([1, 0, 1], dtype=torch.long)
    expected = torch.index_select(src_cpu, axis, idx_cpu)
    got = torch.index_select(_to_dev(src_cpu), axis, idx_cpu)
    _check(got, expected)


def test_index_select_consecutive_arange_index():
    """When the index is a contiguous arange, the coalesce_runs path should
    collapse into a single v2v per outer index."""
    src_cpu = torch.arange(28 * 8 * 1024 * 4, dtype=torch.bfloat16).reshape(28, 8, 1024, 4)
    idx_cpu = torch.arange(64, dtype=torch.long)  # consecutive 0..63
    expected = torch.index_select(src_cpu, 2, idx_cpu)
    got = torch.index_select(_to_dev(src_cpu), 2, idx_cpu)
    _check(got, expected)


def test_index_select_scattered_index():
    src_cpu = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    idx_cpu = torch.tensor([7, 2, 9, 0, 5, 5, 7], dtype=torch.long)
    expected = torch.index_select(src_cpu, 0, idx_cpu)
    got = torch.index_select(_to_dev(src_cpu), 0, idx_cpu)
    _check(got, expected)


def test_index_select_index_on_device():
    src_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    idx_cpu = torch.tensor([4, 0, 2], dtype=torch.long)
    idx_dev = idx_cpu.to(DEVICE)
    expected = torch.index_select(src_cpu, 0, idx_cpu)
    got = torch.index_select(_to_dev(src_cpu), 0, idx_dev)
    _check(got, expected)


def test_index_select_non_contig_self():
    """self is a non-contiguous slice — kernel must use stride-aware copy."""
    base_cpu = torch.arange(2 * 4 * 8 * 6, dtype=torch.float32).reshape(2, 4, 8, 6)
    base_dev = _to_dev(base_cpu)
    self_cpu = base_cpu[0, :, ::2, :]  # (4, 4, 6) non-contig at dim 1 (stride hole)
    self_dev = base_dev[0, :, ::2, :]
    assert not self_dev.is_contiguous(), "expected non-contig"
    idx = torch.tensor([3, 0, 2, 1], dtype=torch.long)
    expected = torch.index_select(self_cpu, 1, idx)
    got = torch.index_select(self_dev, 1, idx)
    _check(got, expected)


def test_index_select_empty_index():
    src_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    idx = torch.tensor([], dtype=torch.long)
    expected = torch.index_select(src_cpu, 0, idx)
    got = torch.index_select(_to_dev(src_cpu), 0, idx)
    _check(got, expected)


def test_index_select_repeat_index():
    """A single index repeated many times — exercises the run path with length 1."""
    src_cpu = torch.arange(20, dtype=torch.float16).reshape(4, 5)
    idx = torch.tensor([2, 2, 2, 2, 2], dtype=torch.long)
    expected = torch.index_select(src_cpu, 0, idx)
    got = torch.index_select(_to_dev(src_cpu), 0, idx)
    _check(got, expected)


def test_index_select_int32_index():
    src_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    idx = torch.tensor([3, 0, 4], dtype=torch.int32)
    expected = torch.index_select(src_cpu, 0, idx)
    got = torch.index_select(_to_dev(src_cpu), 0, idx)
    _check(got, expected)


# ---------------------------------------------------------------------------
# integration: end-to-end mimic of test.py's hot path
# ---------------------------------------------------------------------------


def test_test_py_pattern_small():
    """Smaller version of test.py's gather pattern to verify cat + stack +
    index_select compose correctly."""
    N_LAYERS, N_BLOCKS, N_KV_H, BLK_SZ, HEAD_D = 4, 8, 2, 32, 8
    kv_cpu_list = [torch.randn(2, N_BLOCKS, N_KV_H, 1, BLK_SZ, HEAD_D, dtype=torch.bfloat16) for _ in range(N_LAYERS)]
    kv_dev_list = [_to_dev(kv) for kv in kv_cpu_list]
    slot_idx = torch.arange(BLK_SZ // 2, dtype=torch.long)
    blk = 3

    def gather(kv_caches):
        k_layers = [kv[0, blk, :, 0, :, :] for kv in kv_caches]
        v_layers = [kv[1, blk, :, 0, :, :] for kv in kv_caches]
        k = torch.stack(k_layers).index_select(2, slot_idx)
        v = torch.stack(v_layers).index_select(2, slot_idx)
        return torch.stack([k, v])

    expected = gather(kv_cpu_list)
    got = gather(kv_dev_list)
    _check(got.cpu(), expected) if got.device.type != "cpu" else _check(got, expected)


# ---------------------------------------------------------------------------
# v2v call-count tests — guard against perf regressions in the
# stride-coalescing / axis-absorption algorithm. Correctness alone would not
# catch e.g. `outer_end = max(contig_start, axis+1)` being re-introduced,
# which silently inflates v2v counts by `shape[axis]`×.
# ---------------------------------------------------------------------------


from torch_rbln._C import _transfer_stats_snapshot, _transfer_stats_reset


def _count_v2v(fn):
    """Run fn() with a clean counter snapshot; return (v2v_calls, v2v_bytes, fallback_dispatches)."""
    # Warm to ensure any lazy init has happened — we want the steady-state count.
    fn()
    _transfer_stats_reset()
    before = _transfer_stats_snapshot()
    fn()
    after = _transfer_stats_snapshot()
    return (
        after.v2v_calls - before.v2v_calls,
        after.v2v_bytes - before.v2v_bytes,
        after.fallback_dispatches - before.fallback_dispatches,
    )


def test_v2v_count_cat_axis0_contig_absorbs_axis():
    """28 contig inputs cat at axis=0 must collapse into exactly N v2v calls
    (one per input). The axis-absorption optimization is what makes this
    `N`-not-`N*shape[0]`. Total bytes must equal sum(input nbytes)."""
    n = 28
    inputs = [_to_dev(torch.randn(8, 1024, 128, dtype=torch.bfloat16)) for _ in range(n)]
    calls, by, fbk = _count_v2v(lambda: torch.cat([t.unsqueeze(0) for t in inputs], dim=0))
    assert calls == n, f"expected {n} v2v calls (1 per input), got {calls}"
    assert by == n * 8 * 1024 * 128 * 2, f"unexpected total bytes {by}"
    assert fbk == 0


def test_v2v_count_stack_rank3_inputs():
    """Reproducer for the stack-decomposition path: PyTorch passes raw rank-3
    inputs to cat at dim=0 (instead of unsqueezed rank-4). Without the
    axis-absorption fix this would emit shape[0]× more v2v calls."""
    n = 28
    inputs = [_to_dev(torch.randn(8, 1024, 128, dtype=torch.bfloat16)) for _ in range(n)]
    calls, by, fbk = _count_v2v(lambda: torch.stack(inputs))
    assert calls == n, (
        f"stack(28× (8,1024,128)) regressed: expected {n} v2v, got {calls}. "
        f"Likely outer_end cap is back to max(contig_start, axis+1) instead of "
        f"max(contig_start, axis)."
    )
    assert by == n * 8 * 1024 * 128 * 2
    assert fbk == 0


def test_v2v_count_cat_non_contig_inner_run():
    """Non-contig input (stride hole at dim 1) — the kernel must fall back to
    iterating the outer dims that include the hole, but should still cover
    every byte once. Call count = product of outer dims for each input."""
    # Parent (2, 16, 32) contig; slice at dim 1 with stride 2 → shape (16, 32) with
    # stride (64, 1). contig_suffix_start = 1 (only last dim contig).
    parent_cpu = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 32)
    parent = _to_dev(parent_cpu)
    slc = parent[0, ::2, :]  # (8, 32) with strides (64, 1) — non-contig at dim 0
    assert not slc.is_contiguous()
    # cat 3 copies at dim=0 → outer iter must cover dim 0 (8 elements) per input
    calls, by, fbk = _count_v2v(lambda: torch.cat([slc, slc, slc], dim=0))
    # contig_start of slc = 1 (dim 1 contig, dim 0 has stride hole).
    # outer_end = max(1, 0) = 1. Outer iter = shape[:1] = (8,) → 8 v2v per input.
    assert calls == 3 * 8, f"expected {3*8} v2v calls, got {calls}"
    assert by == 3 * 8 * 32 * 4
    assert fbk == 0


def test_v2v_count_cat_axis_middle_contig_input():
    """Cat at a middle axis with fully-contig inputs: outer_end = max(0, axis) =
    axis. Per-outer iteration covers dims [0, axis); per call writes one full
    block of `shape[axis:]` elements."""
    # input shape (3, 5, 7) at axis=1, 2 inputs.
    a = _to_dev(torch.randn(3, 5, 7, dtype=torch.float32))
    b = _to_dev(torch.randn(3, 5, 7, dtype=torch.float32))
    calls, by, fbk = _count_v2v(lambda: torch.cat([a, b], dim=1))
    # outer_end = max(0, 1) = 1, outer iter = shape[:1] = (3,) → 3 v2v per input.
    assert calls == 2 * 3, f"expected 6 v2v calls, got {calls}"
    assert by == 2 * 3 * 5 * 7 * 4
    assert fbk == 0


def test_v2v_count_cat_axis_last_dim():
    """Cat at last dim: every input contributes outer_count=prod(shape[:-1])
    v2v calls (the inner-block is just the last dim)."""
    a = _to_dev(torch.randn(2, 3, 4, dtype=torch.float32))
    b = _to_dev(torch.randn(2, 3, 5, dtype=torch.float32))
    calls, by, fbk = _count_v2v(lambda: torch.cat([a, b], dim=-1))
    # outer_end = max(0, 2) = 2, outer = shape[:2] = (2,3) → 6 v2v per input.
    assert calls == 2 * 2 * 3, f"expected 12 v2v calls, got {calls}"
    assert by == 2 * 3 * (4 + 5) * 4
    assert fbk == 0


def test_v2v_count_index_select_run_coalescing():
    """index_select with a consecutive arange index must emit one v2v per
    (pre × between) outer index — the run-coalescing collapses the inner
    per-element loop into a single contig slab."""
    self_t = _to_dev(torch.randn(28, 8, 1024, 128, dtype=torch.bfloat16))
    idx = torch.arange(64, dtype=torch.long)
    calls, by, fbk = _count_v2v(lambda: torch.index_select(self_t, 2, idx))
    # contig_start = 0, outer_end = max(0, 3) = 3.
    # pre_count = 28*8 = 224, btw_count = 1, runs = 1 → 224 v2v.
    assert calls == 224, f"expected 224 v2v calls, got {calls}"
    assert by == 224 * 64 * 128 * 2  # 224 calls × (run_len * inner_block_bytes)
    assert fbk == 0


def test_v2v_count_index_select_scattered():
    """A scattered (non-consecutive) index should produce one v2v per
    (pre × between × run-of-1) — i.e. `pre_count * len(idx)` calls."""
    self_t = _to_dev(torch.randn(4, 3, 10, 5, dtype=torch.float32))
    idx = torch.tensor([7, 2, 9, 0, 5], dtype=torch.long)  # 5 scattered indices → 5 runs of 1
    calls, by, fbk = _count_v2v(lambda: torch.index_select(self_t, 2, idx))
    # pre = 4*3 = 12, btw = 1, runs = 5 → 60 v2v
    assert calls == 12 * 5, f"expected 60 v2v calls, got {calls}"
    assert by == 12 * 5 * 5 * 4  # 60 × (1 * inner_block_bytes)
    assert fbk == 0


def test_v2v_count_no_cpu_fallback_for_full_workload():
    """End-to-end smoke test: the test.py-style gather must produce ZERO
    CPU fallback dispatches (cat + index_select + stack are all native v2v)."""
    layers = [_to_dev(torch.randn(2, 4, 2, 1, 32, 8, dtype=torch.bfloat16)) for _ in range(4)]
    slot_idx = torch.arange(16, dtype=torch.long)

    def gather():
        ks = [kv[0, 0, :, 0, :, :] for kv in layers]
        vs = [kv[1, 0, :, 0, :, :] for kv in layers]
        k = torch.stack(ks).index_select(2, slot_idx)
        v = torch.stack(vs).index_select(2, slot_idx)
        return torch.stack([k, v])

    _, _, fbk = _count_v2v(gather)
    assert fbk == 0, f"expected zero CPU fallback dispatches, got {fbk}"


# ---------------------------------------------------------------------------
# View-variation tests — these exercise the stride-coalescing / offset math
# under unusual input layouts: permute, expand (stride==0), flip (negative
# stride), narrow, multi-axis stride holes, non-zero storage_offset,
# unsqueeze-in-middle, size-1 dims at various positions.
#
# For each: assert (a) output matches CPU reference bit-exactly and (b) no
# CPU fallback was needed (sanity check that we genuinely ran on device).
# ---------------------------------------------------------------------------


def _run_check_no_fallback(cpu_fn, dev_fn, atol=0.0, rtol=0.0):
    """Run dev_fn(), check it equals cpu_fn(), and verify zero CPU fallback."""
    expected = cpu_fn()
    _transfer_stats_reset()
    before = _transfer_stats_snapshot()
    got = dev_fn()
    after = _transfer_stats_snapshot()
    fbk = after.fallback_dispatches - before.fallback_dispatches
    assert fbk == 0, f"CPU fallback was triggered ({fbk} dispatches) — kernel is not handling this input"
    _check(got, expected, atol=atol, rtol=rtol)


def test_cat_permuted_input():
    """Permute swaps dim order → non-canonical strides. Cat must still work."""
    a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    b_cpu = torch.arange(100, 124, dtype=torch.float32).reshape(2, 3, 4)
    a_perm_cpu = a_cpu.permute(2, 0, 1)  # shape (4, 2, 3), strides (1, 12, 4)
    b_perm_cpu = b_cpu.permute(2, 0, 1)
    _run_check_no_fallback(
        lambda: torch.cat([a_perm_cpu, b_perm_cpu], dim=0),
        lambda: torch.cat([_to_dev(a_cpu).permute(2, 0, 1), _to_dev(b_cpu).permute(2, 0, 1)], dim=0),
    )


def test_cat_transposed_2d():
    """Classic transpose case."""
    a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    a_t_cpu = a_cpu.t()  # shape (5, 4), strides (1, 5)
    _run_check_no_fallback(
        lambda: torch.cat([a_t_cpu, a_t_cpu], dim=0),
        lambda: torch.cat([_to_dev(a_cpu).t(), _to_dev(a_cpu).t()], dim=0),
    )


def test_cat_expanded_input():
    """Expand creates stride-0 dims (broadcasting). Cat must materialise the
    repeated values correctly."""
    base_cpu = torch.arange(6, dtype=torch.float32).reshape(1, 6)  # shape (1, 6)
    exp_cpu = base_cpu.expand(4, 6)  # shape (4, 6), strides (0, 1)
    _run_check_no_fallback(
        lambda: torch.cat([exp_cpu, exp_cpu], dim=0),
        lambda: torch.cat([_to_dev(base_cpu).expand(4, 6), _to_dev(base_cpu).expand(4, 6)], dim=0),
    )


@pytest.mark.skip(
    reason="aten::flip itself is a CPU-fallback op on RBLN, so the input "
    "reaching our cat is already materialised — this test would only "
    "exercise the upstream flip fallback, not our kernel."
)
def test_cat_flipped_input():
    pass


def test_cat_narrow_input():
    """`narrow` slices a contig region (no stride hole). Should hit the fast
    path: contig_suffix_start matches the narrowed shape."""
    big_cpu = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    a_cpu = big_cpu.narrow(0, 1, 3)  # rows 1..3, contig
    b_cpu = big_cpu.narrow(0, 2, 2)  # rows 2..3, contig
    _run_check_no_fallback(
        lambda: torch.cat([a_cpu, b_cpu], dim=0),
        lambda: torch.cat([_to_dev(big_cpu).narrow(0, 1, 3), _to_dev(big_cpu).narrow(0, 2, 2)], dim=0),
    )


def test_cat_multi_axis_stride_holes():
    """View with stride holes on multiple dims simultaneously."""
    base_cpu = torch.arange(2 * 6 * 8 * 4, dtype=torch.float32).reshape(2, 6, 8, 4)
    v_cpu = base_cpu[:, ::2, ::2, :]  # shape (2, 3, 4, 4) with holes at dims 1 & 2
    _run_check_no_fallback(
        lambda: torch.cat([v_cpu, v_cpu], dim=0),
        lambda: torch.cat([_to_dev(base_cpu)[:, ::2, ::2, :]] * 2, dim=0),
    )


def test_cat_non_zero_storage_offset():
    """A slice that starts mid-storage — storage_offset > 0 but otherwise contig."""
    big_cpu = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    a_cpu = big_cpu[2:5]  # shape (3, 8), contig, storage_offset = 16
    _run_check_no_fallback(
        lambda: torch.cat([a_cpu, a_cpu], dim=1),
        lambda: torch.cat([_to_dev(big_cpu)[2:5], _to_dev(big_cpu)[2:5]], dim=1),
    )


def test_cat_unsqueeze_middle():
    """unsqueeze inserts a size-1 dim with potentially non-canonical stride."""
    a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    a_u_cpu = a_cpu.unsqueeze(2)  # shape (2, 3, 1, 4)
    _run_check_no_fallback(
        lambda: torch.cat([a_u_cpu, a_u_cpu], dim=0),
        lambda: torch.cat([_to_dev(a_cpu).unsqueeze(2)] * 2, dim=0),
    )


def test_cat_size_one_dim_at_various_positions():
    """Inputs whose shapes already include size-1 dims at varying positions."""
    cases = [
        (1, 4, 8),
        (4, 1, 8),
        (4, 8, 1),
        (1, 1, 8),
        (1, 4, 1),
    ]
    for shape in cases:
        a_cpu = torch.randn(*shape, dtype=torch.float32)
        b_cpu = torch.randn(*shape, dtype=torch.float32)
        _run_check_no_fallback(
            lambda: torch.cat([a_cpu, b_cpu], dim=0),
            lambda: torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=0),
        )


def test_cat_negative_dim_resolves_correctly():
    """Negative dim must produce the same result as the positive equivalent."""
    a_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
    b_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
    for neg, pos in [(-1, 2), (-2, 1), (-3, 0)]:
        expected = torch.cat([a_cpu, b_cpu], dim=pos)
        got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=neg)
        _check(got, expected)


def test_cat_mixed_contig_and_non_contig_inputs():
    """One contig + one non-contig in the same cat — algorithm must handle
    each input independently."""
    base_cpu = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    contig_cpu = base_cpu[:3]                  # contig
    non_contig_cpu = base_cpu[:, ::2]          # shape (5, 4) non-contig
    # Make shapes compatible at non-cat dims:
    contig_cpu = base_cpu[:3, :4]              # (3, 4)
    non_contig_cpu = base_cpu[:3, ::2]         # (3, 4) non-contig at dim 1
    _run_check_no_fallback(
        lambda: torch.cat([contig_cpu, non_contig_cpu], dim=0),
        lambda: torch.cat([_to_dev(base_cpu)[:3, :4], _to_dev(base_cpu)[:3, ::2]], dim=0),
    )


# Same view-variation coverage for index_select.


def test_index_select_permuted_self():
    a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    self_perm_cpu = a_cpu.permute(1, 2, 0)  # (3, 4, 2)
    idx = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(self_perm_cpu, 0, idx),
        lambda: torch.index_select(_to_dev(a_cpu).permute(1, 2, 0), 0, idx),
    )


def test_index_select_transposed_self():
    a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    idx = torch.tensor([2, 0, 3], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(a_cpu.t(), 0, idx),
        lambda: torch.index_select(_to_dev(a_cpu).t(), 0, idx),
    )


def test_index_select_expanded_self():
    base_cpu = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    exp_cpu = base_cpu.expand(4, 6)
    idx = torch.tensor([2, 0], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(exp_cpu, 0, idx),
        lambda: torch.index_select(_to_dev(base_cpu).expand(4, 6), 0, idx),
    )


def test_index_select_narrowed_self():
    big_cpu = torch.arange(50, dtype=torch.float32).reshape(10, 5)
    self_cpu = big_cpu.narrow(0, 2, 5)  # rows 2..6
    idx = torch.tensor([4, 0, 2], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(self_cpu, 0, idx),
        lambda: torch.index_select(_to_dev(big_cpu).narrow(0, 2, 5), 0, idx),
    )


def test_index_select_unsqueezed_self():
    a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    self_cpu = a_cpu.unsqueeze(1)  # (2, 1, 3, 4)
    idx = torch.tensor([2, 0, 1], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(self_cpu, 2, idx),
        lambda: torch.index_select(_to_dev(a_cpu).unsqueeze(1), 2, idx),
    )


def test_index_select_size_one_dim_at_position():
    """index_select on a self that has size-1 dims."""
    a_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    idx = torch.tensor([2, 0, 1], dtype=torch.long)
    _run_check_no_fallback(
        lambda: torch.index_select(a_cpu, 0, idx),
        lambda: torch.index_select(_to_dev(a_cpu), 0, idx),
    )


def test_index_select_negative_dim():
    # Pick a shape that's big enough on every dim so a 3-element index is in range.
    a_cpu = torch.arange(2 * 5 * 7 * 4, dtype=torch.float32).reshape(2, 5, 7, 4)
    # idx values must be in range for ALL three target dims (min size = 2 here).
    # Use one shared idx per (neg, pos) pair that fits each axis.
    for neg, pos in [(-1, 3), (-2, 2), (-3, 1), (-4, 0)]:
        axis_size = a_cpu.size(pos)
        idx = torch.tensor([axis_size - 1, 0, axis_size // 2], dtype=torch.long)
        expected = torch.index_select(a_cpu, pos, idx)
        got = torch.index_select(_to_dev(a_cpu), neg, idx)
        _check(got, expected)


def test_index_select_2d_index_rejected():
    """2-D index must be rejected by our validation (not silently flattened)."""
    a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    idx_2d = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    with pytest.raises(RuntimeError):
        torch.index_select(_to_dev(a_cpu), 0, idx_2d)


def test_index_select_oob_index_rejected():
    """Out-of-range index value must error, not silently corrupt output."""
    a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    bad_idx = torch.tensor([0, 1, 4], dtype=torch.long)  # 4 is OOB (axis 0 has size 4)
    with pytest.raises(RuntimeError):
        torch.index_select(_to_dev(a_cpu), 0, bad_idx)


def test_cat_rank_mismatch_rejected():
    a_cpu = torch.randn(3, 4, dtype=torch.float32)
    b_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=0)


def test_cat_non_cat_dim_size_mismatch_rejected():
    a_cpu = torch.randn(3, 4, dtype=torch.float32)
    b_cpu = torch.randn(3, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=0)


def test_cat_dim_out_of_range_rejected():
    a_cpu = torch.randn(3, 4, dtype=torch.float32)
    with pytest.raises((RuntimeError, IndexError)):
        torch.cat([_to_dev(a_cpu), _to_dev(a_cpu)], dim=5)
