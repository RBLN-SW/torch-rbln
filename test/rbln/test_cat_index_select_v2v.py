# Owner(s): ["module: PrivateUse1"]

"""Tests for the C++ v2v-based cat / index_select kernels.

Each test allocates the input on rbln:0, runs the op on device, and compares
against the CPU reference. Coverage:
  - cat: contiguous & non-contiguous inputs, all axes, dtype mix, edge cases
  - index_select: contiguous & non-contig self, runs of consecutive indices,
    scattered indices, index on CPU vs index on RBLN, dim variants
  - stack: just verifies it routes through cat correctly
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import close, DEVICE, ENGINE_DTYPES, eq, to_dev


def _check(actual_dev: torch.Tensor, expected_cpu: torch.Tensor, atol: float = 0.0, rtol: float = 0.0):
    """Thin wrapper that picks `eq` (bitwise) or `close` (tolerance) based on
    whether the caller passed non-zero tolerance."""
    if atol == 0.0 and rtol == 0.0:
        eq(actual_dev, expected_cpu)
    else:
        close(actual_dev, expected_cpu, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# cat / stack
# ---------------------------------------------------------------------------


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestCatV2V(TestCase):
    """Tests for the native v2v `aten::cat` kernel and the `aten::stack`
    composite that decomposes through it."""

    @dtypes(*ENGINE_DTYPES)
    def test_cat_2input_axis0_contig(self, dtype):
        a_cpu = torch.arange(24, dtype=dtype).reshape(3, 8)
        b_cpu = torch.arange(24, 48, dtype=dtype).reshape(3, 8)
        expected = torch.cat([a_cpu, b_cpu], dim=0)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=0)
        _check(got, expected)

    @parametrize("axis", [0, 1, 2, -1, -2, -3])
    def test_cat_3d_all_axes(self, axis):
        a_cpu = torch.arange(2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
        b_cpu = torch.arange(100, 100 + 2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
        expected = torch.cat([a_cpu, b_cpu], dim=axis)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=axis)
        _check(got, expected)

    def test_cat_many_inputs_axis0(self):
        """28 inputs (like test.py's stack(k_layers) decomposition)."""
        n = 28
        inputs_cpu = [torch.full((1, 4, 8, 16), float(i), dtype=torch.bfloat16) for i in range(n)]
        expected = torch.cat(inputs_cpu, dim=0)
        inputs_dev = [to_dev(x) for x in inputs_cpu]
        got = torch.cat(inputs_dev, dim=0)
        _check(got, expected)

    def test_cat_single_input(self):
        a_cpu = torch.randn(3, 5, dtype=torch.float32)
        expected = torch.cat([a_cpu], dim=0)
        got = torch.cat([to_dev(a_cpu)], dim=0)
        _check(got, expected)

    def test_cat_with_some_empty_inputs(self):
        """PyTorch ignores empty inputs that have numel == 0."""
        a_cpu = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        empty_cpu = torch.zeros(0, 4, dtype=torch.float16)
        b_cpu = torch.arange(20, 32, dtype=torch.float16).reshape(3, 4)
        expected = torch.cat([a_cpu, empty_cpu, b_cpu], dim=0)
        got = torch.cat([to_dev(a_cpu), to_dev(empty_cpu), to_dev(b_cpu)], dim=0)
        _check(got, expected)

    def test_cat_all_empty_inputs(self):
        """When every input is empty, output is empty (default behaviour for the
        pre-allocated `out` tensor: our kernel returns it unchanged)."""
        empties_cpu = [torch.zeros(0, 4, dtype=torch.float16) for _ in range(3)]
        expected = torch.cat(empties_cpu, dim=0)
        got = torch.cat([to_dev(x) for x in empties_cpu], dim=0)
        _check(got, expected)

    def test_cat_non_contig_input_unsqueezed_slice(self):
        """The exact pattern from test.py: kv[0, blk, :, 0, :, :] slices that are
        actually contiguous on inspection, then unsqueeze(0) for stack-decomposition."""
        kv_cpu = torch.arange(2 * 4 * 8 * 1 * 16 * 8, dtype=torch.bfloat16).reshape(2, 4, 8, 1, 16, 8)
        kv = to_dev(kv_cpu)
        layers_cpu = [kv_cpu[0, blk, :, 0, :, :].unsqueeze(0) for blk in range(4)]
        layers_dev = [kv[0, blk, :, 0, :, :].unsqueeze(0) for blk in range(4)]
        expected = torch.cat(layers_cpu, dim=0)
        got = torch.cat(layers_dev, dim=0)
        _check(got, expected)

    def test_cat_genuinely_non_contig_input(self):
        """A non-contig view that cannot be coalesced fully: kv[0, :, :, 0, ::2, :]
        has stride hole at the second-to-last dim."""
        kv_cpu = torch.arange(2 * 4 * 8 * 1 * 16 * 8, dtype=torch.float32).reshape(2, 4, 8, 1, 16, 8)
        kv = to_dev(kv_cpu)
        view_cpu = kv_cpu[0, :, :, 0, ::2, :]
        view_dev = kv[0, :, :, 0, ::2, :]
        assert not view_dev.is_contiguous(), "expected non-contig view"
        expected = torch.cat([view_cpu, view_cpu], dim=0)
        got = torch.cat([view_dev, view_dev], dim=0)
        _check(got, expected)

    def test_cat_axis_last_dim(self):
        a_cpu = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
        b_cpu = torch.arange(30, dtype=torch.float16).reshape(2, 3, 5)
        expected = torch.cat([a_cpu, b_cpu], dim=-1)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=-1)
        _check(got, expected)

    def test_cat_mixed_sizes_at_axis(self):
        """Different shape[axis] per input (the common case)."""
        a_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b_cpu = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        c_cpu = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        expected = torch.cat([a_cpu, b_cpu, c_cpu], dim=1)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu), to_dev(c_cpu)], dim=1)
        _check(got, expected)

    # ---- stack (CompositeImplicitAutograd → cat) ----

    @parametrize("dim", [0, 1, 2, -1])
    def test_stack_decomposes_to_cat(self, dim):
        inputs_cpu = [torch.randn(3, 4, dtype=torch.float32) for _ in range(5)]
        expected = torch.stack(inputs_cpu, dim=dim)
        got = torch.stack([to_dev(x) for x in inputs_cpu], dim=dim)
        _check(got, expected)

    def test_stack_28_layers_bf16(self):
        """The test.py shape pattern."""
        layers_cpu = [torch.randn(4, 8, 16, dtype=torch.bfloat16) for _ in range(28)]
        expected = torch.stack(layers_cpu, dim=0)
        got = torch.stack([to_dev(x) for x in layers_cpu], dim=0)
        _check(got, expected)

    # ---- view-variation tests for cat ----

    def test_cat_permuted_input(self):
        """Permute swaps dim order → non-canonical strides."""
        a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        b_cpu = torch.arange(100, 124, dtype=torch.float32).reshape(2, 3, 4)
        expected = torch.cat([a_cpu.permute(2, 0, 1), b_cpu.permute(2, 0, 1)], dim=0)
        got = torch.cat([to_dev(a_cpu).permute(2, 0, 1), to_dev(b_cpu).permute(2, 0, 1)], dim=0)
        _check(got, expected)

    def test_cat_transposed_2d(self):
        a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        expected = torch.cat([a_cpu.t(), a_cpu.t()], dim=0)
        got = torch.cat([to_dev(a_cpu).t(), to_dev(a_cpu).t()], dim=0)
        _check(got, expected)

    def test_cat_expanded_input(self):
        """Expand creates stride-0 dims (broadcasting). Cat must materialise the
        repeated values correctly."""
        base_cpu = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        exp_cpu = base_cpu.expand(4, 6)
        expected = torch.cat([exp_cpu, exp_cpu], dim=0)
        got = torch.cat([to_dev(base_cpu).expand(4, 6), to_dev(base_cpu).expand(4, 6)], dim=0)
        _check(got, expected)

    @pytest.mark.skip(
        reason="aten::flip itself is a CPU-fallback op on RBLN, so the input "
        "reaching our cat is already materialised — this test would only "
        "exercise the upstream flip fallback, not our kernel."
    )
    def test_cat_flipped_input(self):
        pass

    def test_cat_narrow_input(self):
        """`narrow` slices a contig region (no stride hole). Should hit the fast
        path: contig_suffix_start matches the narrowed shape."""
        big_cpu = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        a_cpu = big_cpu.narrow(0, 1, 3)
        b_cpu = big_cpu.narrow(0, 2, 2)
        expected = torch.cat([a_cpu, b_cpu], dim=0)
        got = torch.cat([to_dev(big_cpu).narrow(0, 1, 3), to_dev(big_cpu).narrow(0, 2, 2)], dim=0)
        _check(got, expected)

    def test_cat_multi_axis_stride_holes(self):
        """View with stride holes on multiple dims simultaneously."""
        base_cpu = torch.arange(2 * 6 * 8 * 4, dtype=torch.float32).reshape(2, 6, 8, 4)
        v_cpu = base_cpu[:, ::2, ::2, :]
        expected = torch.cat([v_cpu, v_cpu], dim=0)
        got = torch.cat([to_dev(base_cpu)[:, ::2, ::2, :]] * 2, dim=0)
        _check(got, expected)

    def test_cat_non_zero_storage_offset(self):
        """A slice that starts mid-storage — storage_offset > 0 but otherwise contig."""
        big_cpu = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        a_cpu = big_cpu[2:5]
        expected = torch.cat([a_cpu, a_cpu], dim=1)
        got = torch.cat([to_dev(big_cpu)[2:5], to_dev(big_cpu)[2:5]], dim=1)
        _check(got, expected)

    def test_cat_unsqueeze_middle(self):
        """unsqueeze inserts a size-1 dim with potentially non-canonical stride."""
        a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        a_u_cpu = a_cpu.unsqueeze(2)
        expected = torch.cat([a_u_cpu, a_u_cpu], dim=0)
        got = torch.cat([to_dev(a_cpu).unsqueeze(2)] * 2, dim=0)
        _check(got, expected)

    @parametrize("shape", [(1, 4, 8), (4, 1, 8), (4, 8, 1), (1, 1, 8), (1, 4, 1)])
    def test_cat_size_one_dim_at_various_positions(self, shape):
        """Inputs whose shapes already include size-1 dims at varying positions."""
        a_cpu = torch.randn(*shape, dtype=torch.float32)
        b_cpu = torch.randn(*shape, dtype=torch.float32)
        expected = torch.cat([a_cpu, b_cpu], dim=0)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=0)
        _check(got, expected)

    @parametrize("neg,pos", [(-1, 2), (-2, 1), (-3, 0)])
    def test_cat_negative_dim_resolves_correctly(self, neg, pos):
        a_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
        b_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
        expected = torch.cat([a_cpu, b_cpu], dim=pos)
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=neg)
        _check(got, expected)

    def test_cat_mixed_contig_and_non_contig_inputs(self):
        """One contig + one non-contig in the same cat — algorithm must handle
        each input independently."""
        base_cpu = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        contig_cpu = base_cpu[:3, :4]
        non_contig_cpu = base_cpu[:3, ::2]
        expected = torch.cat([contig_cpu, non_contig_cpu], dim=0)
        got = torch.cat([to_dev(base_cpu)[:3, :4], to_dev(base_cpu)[:3, ::2]], dim=0)
        _check(got, expected)

    # ---- error paths ----

    def test_cat_rank_mismatch_rejected(self):
        a_cpu = torch.randn(3, 4, dtype=torch.float32)
        b_cpu = torch.randn(3, 4, 5, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=0)

    def test_cat_non_cat_dim_size_mismatch_rejected(self):
        a_cpu = torch.randn(3, 4, dtype=torch.float32)
        b_cpu = torch.randn(3, 5, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=0)

    def test_cat_dim_out_of_range_rejected(self):
        a_cpu = torch.randn(3, 4, dtype=torch.float32)
        with pytest.raises((RuntimeError, IndexError)):
            torch.cat([to_dev(a_cpu), to_dev(a_cpu)], dim=5)

    @dtypes(*ENGINE_DTYPES)
    def test_cat_out_internal_overlap_rejected(self, dtype):
        """``out`` with internal overlap (broadcast view) must raise ``RuntimeError``."""
        a = torch.arange(12, dtype=dtype, device=DEVICE).reshape(3, 4)
        b = torch.arange(100, 112, dtype=dtype, device=DEVICE).reshape(3, 4)
        out = torch.empty(1, 8, dtype=dtype, device=DEVICE).expand(3, 8)
        with pytest.raises(RuntimeError):
            torch.cat([a, b], dim=1, out=out)

    # ---- opinfo conformance regressions ----

    def test_cat_all_zero_axis_resizes_out_to_zero(self):
        """cat of all-empty inputs along an axis where each size is 0 must resize
        `out` to the empty shape (regression for the all-empty fast-path that
        used to return `out` unchanged)."""
        out_dev = torch.empty(1, dtype=torch.float32, device=DEVICE)
        with pytest.warns(UserWarning, match="An output with one or more elements"):
            torch.cat([torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE)], dim=0, out=out_dev)
        assert tuple(out_dev.shape) == (0,)

    def test_stack_empty_inputs_dim_negative_one(self):
        """stack of two (0,1,0) tensors at dim=-1 must produce (0,1,0,2). The
        pre-fix code dropped these as empties and returned the wrong-shape `out`
        unchanged."""
        empties_cpu = [torch.empty(0, 1, 0, dtype=torch.float32) for _ in range(2)]
        expected = torch.stack(empties_cpu, dim=-1)
        got = torch.stack([to_dev(x) for x in empties_cpu], dim=-1)
        assert tuple(got.shape) == tuple(expected.shape) == (0, 1, 0, 2)

    def test_cat_legacy_empty_1d_placeholder_is_skipped(self):
        """A (0,) 1-D placeholder among rank-2 inputs is silently ignored, matching
        PyTorch's `at::native::cat` legacy behaviour."""
        a_cpu = torch.randn(5, 5, dtype=torch.float32)
        placeholder_cpu = torch.empty(0, dtype=torch.float32)
        expected = torch.cat([placeholder_cpu, a_cpu], dim=1)
        got = torch.cat([to_dev(placeholder_cpu), to_dev(a_cpu)], dim=1)
        _check(got, expected)

    def test_cat_out_on_wrong_device_raises_type_error(self):
        """cat / stack family must raise TypeError (not RuntimeError) for `out` on
        the wrong device. Required by test/ops/test_ops.py test_out Case 3."""
        a_dev = torch.randn(2, 3, dtype=torch.float32, device=DEVICE)
        out_cpu = torch.empty(2, 6, dtype=torch.float32, device="cpu")
        with pytest.raises(TypeError):
            torch.cat([a_dev, a_dev], dim=1, out=out_cpu)

    def test_cat_mixed_dtype_promotes_to_common(self):
        """Mixed-dtype inputs are promoted to the common dtype (PyTorch semantics).
        Previously we hard-errored with `promotion is not supported`."""
        a_cpu = torch.randn(3, 4, dtype=torch.float16)
        b_cpu = torch.randn(3, 2, dtype=torch.float64)
        expected = torch.cat([a_cpu, b_cpu], dim=1)
        assert expected.dtype == torch.float64
        got = torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=1)
        assert got.dtype == torch.float64
        _check(got, expected, atol=1e-3, rtol=1e-3)

    @dtypes(*ENGINE_DTYPES)
    def test_cat_out_non_contig_routes_through_staging(self, dtype):
        """Non-contig ``out`` exercises the staging-buffer path."""
        a_cpu = torch.arange(12, dtype=dtype).reshape(3, 4)
        b_cpu = torch.arange(100, 112, dtype=dtype).reshape(3, 4)
        expected = torch.cat([a_cpu, b_cpu], dim=0)
        out = torch.empty(4, 6, dtype=dtype, device=DEVICE).t()
        self.assertFalse(out.is_contiguous())
        torch.cat([to_dev(a_cpu), to_dev(b_cpu)], dim=0, out=out)
        _check(out, expected)

    @pytest.mark.usefixtures("enable_eager_malloc")
    def test_large_strided_cat(self):
        """``aten::cat`` over a large batched strided copy."""
        # Baseline allocations exercise the bulk strided copy path.
        baseline_allocs = [torch.empty(102400, 2560, dtype=torch.float16, device=DEVICE) for _ in range(2)]
        try:
            a = torch.empty(4, 40, 1024, dtype=torch.float16, device=DEVICE).transpose(1, 2)
            b = torch.empty(4, 40, 1024, dtype=torch.float16, device=DEVICE).transpose(1, 2)
            out = torch.cat([a, b], dim=-1)
            self.assertEqual(tuple(out.shape), (4, 1024, 80))
        finally:
            del baseline_allocs


# ---------------------------------------------------------------------------
# index_select
# ---------------------------------------------------------------------------


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestIndexSelectV2V(TestCase):
    """Tests for the native v2v `aten::index_select` kernel."""

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.int32)
    def test_index_select_basic_dim0(self, dtype):
        src_cpu = torch.arange(30, dtype=dtype).reshape(5, 6)
        idx_cpu = torch.tensor([3, 0, 4, 1], dtype=torch.long)
        expected = torch.index_select(src_cpu, 0, idx_cpu)
        got = torch.index_select(to_dev(src_cpu), 0, idx_cpu)
        _check(got, expected)

    @parametrize("axis", [0, 1, 2, -1])
    def test_index_select_all_dims(self, axis):
        src_cpu = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        idx_cpu = torch.tensor([1, 0, 1], dtype=torch.long)
        expected = torch.index_select(src_cpu, axis, idx_cpu)
        got = torch.index_select(to_dev(src_cpu), axis, idx_cpu)
        _check(got, expected)

    def test_index_select_consecutive_arange_index(self):
        """When the index is a contiguous arange, the coalesce_runs path should
        collapse into a single v2v per outer index."""
        src_cpu = torch.arange(28 * 8 * 1024 * 4, dtype=torch.bfloat16).reshape(28, 8, 1024, 4)
        idx_cpu = torch.arange(64, dtype=torch.long)
        expected = torch.index_select(src_cpu, 2, idx_cpu)
        got = torch.index_select(to_dev(src_cpu), 2, idx_cpu)
        _check(got, expected)

    def test_index_select_scattered_index(self):
        src_cpu = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        idx_cpu = torch.tensor([7, 2, 9, 0, 5, 5, 7], dtype=torch.long)
        expected = torch.index_select(src_cpu, 0, idx_cpu)
        got = torch.index_select(to_dev(src_cpu), 0, idx_cpu)
        _check(got, expected)

    def test_index_select_index_on_device(self):
        src_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        idx_cpu = torch.tensor([4, 0, 2], dtype=torch.long)
        idx_dev = idx_cpu.to(DEVICE)
        expected = torch.index_select(src_cpu, 0, idx_cpu)
        got = torch.index_select(to_dev(src_cpu), 0, idx_dev)
        _check(got, expected)

    def test_index_select_non_contig_self(self):
        """self is a non-contiguous slice — kernel must use stride-aware copy."""
        base_cpu = torch.arange(2 * 4 * 8 * 6, dtype=torch.float32).reshape(2, 4, 8, 6)
        base_dev = to_dev(base_cpu)
        self_cpu = base_cpu[0, :, ::2, :]
        self_dev = base_dev[0, :, ::2, :]
        assert not self_dev.is_contiguous(), "expected non-contig"
        idx = torch.tensor([3, 0, 2, 1], dtype=torch.long)
        expected = torch.index_select(self_cpu, 1, idx)
        got = torch.index_select(self_dev, 1, idx)
        _check(got, expected)

    def test_index_select_empty_index(self):
        src_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        idx = torch.tensor([], dtype=torch.long)
        expected = torch.index_select(src_cpu, 0, idx)
        got = torch.index_select(to_dev(src_cpu), 0, idx)
        _check(got, expected)

    def test_index_select_repeat_index(self):
        """A single index repeated many times — exercises the run path with length 1."""
        src_cpu = torch.arange(20, dtype=torch.float16).reshape(4, 5)
        idx = torch.tensor([2, 2, 2, 2, 2], dtype=torch.long)
        expected = torch.index_select(src_cpu, 0, idx)
        got = torch.index_select(to_dev(src_cpu), 0, idx)
        _check(got, expected)

    def test_index_select_int32_index(self):
        src_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        idx = torch.tensor([3, 0, 4], dtype=torch.int32)
        expected = torch.index_select(src_cpu, 0, idx)
        got = torch.index_select(to_dev(src_cpu), 0, idx)
        _check(got, expected)

    @parametrize("dim", [0, -1])
    @parametrize("idx_value", [torch.tensor([0], dtype=torch.long), torch.tensor(0, dtype=torch.long)])
    def test_index_select_zero_dim_self(self, dim, idx_value):
        """0-D self with index_select: PyTorch returns a 0-D tensor mirroring self's
        rank. dim must be 0 or -1; index must contain exactly one value (= 0)."""
        s_cpu = torch.tensor(3.5, dtype=torch.float16)
        expected = torch.index_select(s_cpu, dim, idx_value)
        got = torch.index_select(to_dev(s_cpu), dim, idx_value)
        _check(got, expected)

    def test_index_select_zero_dim_self_invalid_dim_rejected(self):
        """For 0-D self, only dim in {0, -1} is valid."""
        s_dev = to_dev(torch.tensor(1.0, dtype=torch.float32))
        idx = torch.tensor([0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(s_dev, 1, idx)

    def test_index_select_zero_dim_self_multi_index_rejected(self):
        """For 0-D self, index must contain exactly one value (PyTorch upstream)."""
        s_dev = to_dev(torch.tensor(1.0, dtype=torch.float32))
        bad_idx = torch.tensor([0, 0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(s_dev, 0, bad_idx)

    def test_index_select_zero_dim_self_oob_index_rejected(self):
        """For 0-D self, the single index value must be 0."""
        s_dev = to_dev(torch.tensor(1.0, dtype=torch.float32))
        bad_idx = torch.tensor([1], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(s_dev, 0, bad_idx)

    def test_index_select_consecutive_run_on_noncontig_axis(self):
        """When self is non-contiguous AT the index axis (stride > inner block),
        run coalescing must NOT collapse consecutive indices into a single v2v —
        consecutive index values map to memory separated by padding/NaN bytes.

        Reproduces the bug found via PyTorch opinfo's `noncontiguous_samples`
        test, which interleaves NaN padding between data elements."""
        S = 5
        data = torch.arange(S * S, dtype=torch.float16).reshape(S, S)
        padded = torch.empty(S, S, 2, dtype=torch.float16)
        padded[..., 0] = math.nan
        padded[..., 1] = data
        self_cpu = padded[..., 1]
        assert not self_cpu.is_contiguous() and self_cpu.stride() == (2 * S, 2)

        self_dev = padded.to(DEVICE)[..., 1]
        idx = torch.tensor([0, 3, 4, 1], dtype=torch.long)
        expected = torch.index_select(self_cpu, -1, idx)
        got = torch.index_select(self_dev, -1, idx)
        _check(got, expected)

    # ---- integration: end-to-end mimic of test.py's hot path ----

    def test_test_py_pattern_small(self):
        """Smaller version of test.py's gather pattern to verify cat + stack +
        index_select compose correctly."""
        N_LAYERS, N_BLOCKS, N_KV_H, BLK_SZ, HEAD_D = 4, 8, 2, 32, 8
        kv_cpu_list = [
            torch.randn(2, N_BLOCKS, N_KV_H, 1, BLK_SZ, HEAD_D, dtype=torch.bfloat16) for _ in range(N_LAYERS)
        ]
        kv_dev_list = [to_dev(kv) for kv in kv_cpu_list]
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
        _check(got, expected)

    # ---- view-variation tests for index_select ----

    def test_index_select_permuted_self(self):
        a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        self_perm_cpu = a_cpu.permute(1, 2, 0)
        idx = torch.tensor([1, 0, 2, 1], dtype=torch.long)
        expected = torch.index_select(self_perm_cpu, 0, idx)
        got = torch.index_select(to_dev(a_cpu).permute(1, 2, 0), 0, idx)
        _check(got, expected)

    def test_index_select_transposed_self(self):
        a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        idx = torch.tensor([2, 0, 3], dtype=torch.long)
        expected = torch.index_select(a_cpu.t(), 0, idx)
        got = torch.index_select(to_dev(a_cpu).t(), 0, idx)
        _check(got, expected)

    def test_index_select_expanded_self(self):
        base_cpu = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        exp_cpu = base_cpu.expand(4, 6)
        idx = torch.tensor([2, 0], dtype=torch.long)
        expected = torch.index_select(exp_cpu, 0, idx)
        got = torch.index_select(to_dev(base_cpu).expand(4, 6), 0, idx)
        _check(got, expected)

    def test_index_select_narrowed_self(self):
        big_cpu = torch.arange(50, dtype=torch.float32).reshape(10, 5)
        self_cpu = big_cpu.narrow(0, 2, 5)
        idx = torch.tensor([4, 0, 2], dtype=torch.long)
        expected = torch.index_select(self_cpu, 0, idx)
        got = torch.index_select(to_dev(big_cpu).narrow(0, 2, 5), 0, idx)
        _check(got, expected)

    def test_index_select_unsqueezed_self(self):
        a_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        self_cpu = a_cpu.unsqueeze(1)
        idx = torch.tensor([2, 0, 1], dtype=torch.long)
        expected = torch.index_select(self_cpu, 2, idx)
        got = torch.index_select(to_dev(a_cpu).unsqueeze(1), 2, idx)
        _check(got, expected)

    def test_index_select_size_one_dim_at_position(self):
        """index_select on a self that has size-1 dims."""
        a_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
        idx = torch.tensor([2, 0, 1], dtype=torch.long)
        expected = torch.index_select(a_cpu, 0, idx)
        got = torch.index_select(to_dev(a_cpu), 0, idx)
        _check(got, expected)

    @parametrize("neg,pos", [(-1, 3), (-2, 2), (-3, 1), (-4, 0)])
    def test_index_select_negative_dim(self, neg, pos):
        a_cpu = torch.arange(2 * 5 * 7 * 4, dtype=torch.float32).reshape(2, 5, 7, 4)
        axis_size = a_cpu.size(pos)
        idx = torch.tensor([axis_size - 1, 0, axis_size // 2], dtype=torch.long)
        expected = torch.index_select(a_cpu, pos, idx)
        got = torch.index_select(to_dev(a_cpu), neg, idx)
        _check(got, expected)

    @dtypes(*ENGINE_DTYPES)
    def test_index_select_out_non_contig_routes_through_staging(self, dtype):
        """Non-contig ``out`` exercises the staging-buffer path."""
        src_cpu = torch.arange(20, dtype=dtype).reshape(4, 5)
        idx = torch.tensor([3, 0, 2], dtype=torch.long)
        expected = torch.index_select(src_cpu, 0, idx)
        out = torch.empty(5, 3, dtype=dtype, device=DEVICE).t()
        self.assertFalse(out.is_contiguous())
        torch.index_select(to_dev(src_cpu), 0, idx, out=out)
        _check(out, expected)

    # ---- error paths ----

    def test_index_select_2d_index_rejected(self):
        """2-D index must be rejected by our validation (not silently flattened)."""
        a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        idx_2d = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(to_dev(a_cpu), 0, idx_2d)

    @dtypes(*ENGINE_DTYPES)
    def test_index_select_out_aliased_with_self_rejected(self, dtype):
        """``out`` aliased with ``self`` must raise ``RuntimeError``."""
        a = torch.arange(5, dtype=dtype, device=DEVICE)
        idx = torch.tensor([1, 0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(a, 0, idx, out=a)

    @dtypes(*ENGINE_DTYPES)
    def test_index_select_out_internal_overlap_rejected(self, dtype):
        """``out`` with internal overlap (broadcast view) must raise ``RuntimeError``."""
        src = torch.arange(20, dtype=dtype, device=DEVICE).reshape(4, 5)
        idx = torch.tensor([0, 2], dtype=torch.long)
        out = torch.empty(1, 5, dtype=dtype, device=DEVICE).expand(2, 5)
        with pytest.raises(RuntimeError):
            torch.index_select(src, 0, idx, out=out)

    def test_index_select_oob_index_rejected(self):
        """Out-of-range index value must error, not silently corrupt output."""
        a_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        bad_idx = torch.tensor([0, 1, 4], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(to_dev(a_cpu), 0, bad_idx)

    def test_index_select_empty_self_oob_index_rejected(self):
        """self empty on the indexed axis (extent 0): any index is out of range and must
        raise like CPU, not resize out to (1,3) and return it uninitialized."""
        empty_self = torch.empty((0, 3), dtype=torch.float32)
        idx = torch.tensor([0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(empty_self, 0, idx)  # CPU parity
        with pytest.raises(RuntimeError):
            torch.index_select(to_dev(empty_self), 0, idx)

    def test_index_select_empty_output_valid_index(self):
        """self (3,0): extent-3 axis makes index [0] valid with empty output
        (1,0) — must NOT raise (bounds fix must not reject valid indices)."""
        empty_self = torch.empty((3, 0), dtype=torch.float32)
        idx = torch.tensor([0], dtype=torch.long)
        expected = torch.index_select(empty_self, 0, idx)
        got = torch.index_select(to_dev(empty_self), 0, idx)
        _check(got, expected)

    def test_index_select_empty_output_oob_index_rejected(self):
        """self (3,0): index 5 exceeds extent 3 and must raise like CPU even though the
        output would be empty (numel()==0 must not swallow it)."""
        empty_self = torch.empty((3, 0), dtype=torch.float32)
        idx = torch.tensor([5], dtype=torch.long)
        with pytest.raises(RuntimeError):
            torch.index_select(empty_self, 0, idx)  # CPU parity
        with pytest.raises(RuntimeError):
            torch.index_select(to_dev(empty_self), 0, idx)


instantiate_device_type_tests(TestCatV2V, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestIndexSelectV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
