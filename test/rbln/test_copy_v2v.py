# Owner(s): ["module: PrivateUse1"]

"""Tests for the v2v-based copy refactor.

These tests exercise the strided v2v engine and the always-v2v copy path
through `tensor.copy_(other)` and `tensor.to(device)`. Coverage is organised by
the property under test rather than by op:

  - Engine: stride patterns that exercise contig_suffix_start logic on each
    side (contig / sliced / transposed / permuted / expanded / unfolded /
    storage-offset>0), in all device pair directions (h2d / d2h / d2d).
  - Broadcast: expand() / stride==0 paths through the engine.
  - Aliasing: identity copy, overlapping views.
  - Numel edges: 0-numel along various dims, 0-D scalars, single element.
  - Dtype: same-dtype (engine) and cross-dtype (CPU cast staging).
  - cat / index_select retrofit: patterns that route through the new engine
    via `out.narrow(...)` views — verifies the retrofit is wired correctly.

Each test compares against a CPU reference. Same-dtype copies are checked
bitwise; dtype-converted copies use `assert_close` with float tolerance.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import (
    arange as _arange_like,
    close as _close,
    CPU,
    DEVICE,
    ENGINE_DTYPES,
    eq as _eq,
    to_dev as _to_dev,
)


_STRIDED_PATTERNS = [
    ((4, 6), (slice(None), slice(None, 4)), "row-contig col-slice"),
    ((4, 6), (slice(None, 3), slice(None)), "outer-slice"),
    ((4, 6), (slice(1, 4), slice(2, 6)), "offset both dims"),
    ((4, 6), (slice(None), slice(None, None, 2)), "strided inner"),
    ((3, 5, 7), (slice(None), slice(1, 4), slice(None)), "middle-dim slice"),
    ((3, 5, 7), (slice(None), slice(None), slice(0, 5)), "innermost slice"),
    ((2, 3, 4, 5), (slice(None), slice(None), slice(1, 3), slice(None)), "4-d mid"),
]


def _noncontig_2d(shape, dtype, device):
    """A non-contig view of size `shape` on `device`. Uses [:N, :M] of a bigger
    base so storage_offset=0 but stride[0] is larger than expected."""
    big = torch.empty((shape[0], shape[1] + 2), dtype=dtype, device=device)
    return big[:, : shape[1]]


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestCopyV2V(TestCase):
    """Tests for the v2v-based ``aten::copy_`` path and the strided
    ``strided_v2v_copy`` engine that drives it."""

    # ---- engine fast paths — both sides contig, or 0-D / single element ----

    @dtypes(*ENGINE_DTYPES)
    @parametrize("shape", [(), (1,), (8,), (3, 5), (2, 3, 4), (2, 3, 4, 5)])
    def test_engine_both_contig_same_dtype(self, dtype, shape):
        src_cpu = _arange_like(shape, dtype) if shape else torch.tensor(7, dtype=dtype)
        src_dev = _to_dev(src_cpu)
        dst_dev = torch.empty(shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_dev)
        _eq(dst_dev, src_cpu)

    @dtypes(*ENGINE_DTYPES)
    def test_engine_zero_d_scalar(self, dtype):
        """0-D scalar should route through the rank==0 fast path."""
        src_cpu = torch.tensor(42, dtype=dtype)
        src_dev = _to_dev(src_cpu)
        dst_dev = torch.empty((), dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_dev)
        _eq(dst_dev, src_cpu)

    @dtypes(*ENGINE_DTYPES)
    def test_engine_single_element(self, dtype):
        src_cpu = torch.tensor([99], dtype=dtype)
        src_dev = _to_dev(src_cpu)
        dst_dev = torch.empty(1, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_dev)
        _eq(dst_dev, src_cpu)

    # ---- engine strided paths — non-contig on one side, both sides, all axes ----

    @dtypes(torch.float16, torch.int32)
    def test_engine_dst_sliced_inner(self, dtype):
        """dst is a contig outer slice of a larger tensor; src is contig.
        Inner-most dim stays contig in both → 1 strided range with outer K=axis-0 size."""
        big_cpu = _arange_like((4, 12), dtype) * 0
        big_dev = _to_dev(big_cpu)
        dst_view = big_dev[:, :8]
        src_cpu = _arange_like((4, 8), dtype) + 1
        src_dev = _to_dev(src_cpu)

        dst_view.copy_(src_dev)

        expected = big_cpu.clone()
        expected[:, :8] = src_cpu
        _eq(big_dev, expected)

    @dtypes(torch.float16, torch.int64)
    def test_engine_src_sliced_inner(self, dtype):
        src_big_cpu = _arange_like((4, 12), dtype) + 1
        src_big_dev = _to_dev(src_big_cpu)
        src_view = src_big_dev[:, :8]
        dst_dev = torch.empty((4, 8), dtype=dtype, device=DEVICE)

        dst_dev.copy_(src_view)

        _eq(dst_dev, src_big_cpu[:, :8])

    @dtypes(torch.float32, torch.int32)
    def test_engine_non_contig_both_sides(self, dtype):
        """Both dst and src are non-contig views of independent big tensors."""
        src_big_cpu = _arange_like((6, 10), dtype) + 100
        dst_big_cpu = torch.zeros((8, 12), dtype=dtype)

        src_big_dev = _to_dev(src_big_cpu)
        dst_big_dev = _to_dev(dst_big_cpu)

        src_view = src_big_dev[2:6, 1:6]
        dst_view = dst_big_dev[1:5, 2:7]
        assert src_view.shape == dst_view.shape == (4, 5)

        dst_view.copy_(src_view)

        expected = dst_big_cpu.clone()
        expected[1:5, 2:7] = src_big_cpu[2:6, 1:6]
        _eq(dst_big_dev, expected)

    @dtypes(torch.bfloat16, torch.int32)
    @parametrize("perm", [(1, 0), (0, 1)])
    def test_engine_transpose_2d(self, dtype, perm):
        src_cpu = _arange_like((4, 6), dtype)
        src_dev = _to_dev(src_cpu)
        src_t = src_dev.permute(*perm)
        dst_dev = torch.empty(src_t.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_t)
        _eq(dst_dev, src_cpu.permute(*perm).contiguous())

    @dtypes(*ENGINE_DTYPES)
    @parametrize("perm", [(2, 0, 1), (1, 2, 0), (0, 2, 1)])
    def test_engine_permute_3d(self, dtype, perm):
        src_cpu = _arange_like((2, 3, 4), dtype)
        src_dev = _to_dev(src_cpu)
        src_p = src_dev.permute(*perm)
        dst_dev = torch.empty(src_p.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_p)
        _eq(dst_dev, src_cpu.permute(*perm).contiguous())

    @dtypes(torch.float32)
    def test_engine_storage_offset_nonzero(self, dtype):
        """Storage offset > 0 must be respected — data_ptr() handles it."""
        big_cpu = _arange_like((6, 8), dtype)
        big_dev = _to_dev(big_cpu)
        src_view = big_dev[2:5, 3:7]
        assert src_view.storage_offset() > 0

        dst_dev = torch.empty(src_view.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_view)
        _eq(dst_dev, big_cpu[2:5, 3:7])

    @dtypes(torch.int32)
    def test_engine_strided_step_inner(self, dtype):
        """Strided inner dim (step>1) breaks inner contig in src; engine must
        handle one slab per surviving inner position."""
        src_cpu = _arange_like((4, 16), dtype)
        src_dev = _to_dev(src_cpu)
        src_view = src_dev[:, ::2]
        assert not src_view.is_contiguous()

        dst_dev = torch.empty(src_view.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_view)
        _eq(dst_dev, src_cpu[:, ::2])

    # ---- broadcast: stride==0 dims must each emit their own write ----

    @dtypes(torch.float16, torch.int32)
    def test_engine_broadcast_outer(self, dtype):
        src_cpu = _arange_like((1, 8), dtype) + 1
        src_dev = _to_dev(src_cpu)
        expanded = src_dev.expand(4, 8)
        dst_dev = torch.empty((4, 8), dtype=dtype, device=DEVICE)
        dst_dev.copy_(expanded)
        _eq(dst_dev, src_cpu.expand(4, 8).contiguous())

    @dtypes(*ENGINE_DTYPES)
    def test_engine_broadcast_inner(self, dtype):
        """stride==0 on the innermost non-size-1 dim — engine must NOT absorb
        it into the inner contig block."""
        src_cpu = torch.tensor([7.0], dtype=dtype)
        src_dev = _to_dev(src_cpu)
        expanded = src_dev.expand(8)
        dst_dev = torch.empty((8,), dtype=dtype, device=DEVICE)
        dst_dev.copy_(expanded)
        _eq(dst_dev, torch.full((8,), 7.0, dtype=dtype))

    @dtypes(torch.bfloat16)
    def test_engine_broadcast_multi_dim(self, dtype):
        src_cpu = _arange_like((1, 1, 4), dtype) + 10
        src_dev = _to_dev(src_cpu)
        expanded = src_dev.expand(3, 5, 4)
        dst_dev = torch.empty((3, 5, 4), dtype=dtype, device=DEVICE)
        dst_dev.copy_(expanded)
        _eq(dst_dev, src_cpu.expand(3, 5, 4).contiguous())

    @dtypes(torch.float32)
    def test_engine_broadcast_into_noncontig_dst(self, dtype):
        """Combination test: expand src + non-contig sliced dst."""
        src_cpu = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
        src_dev = _to_dev(src_cpu)
        expanded = src_dev.expand(3, 4)

        big_cpu = torch.zeros((6, 10), dtype=dtype)
        big_dev = _to_dev(big_cpu)
        dst_view = big_dev[1:4, 2:6]

        dst_view.copy_(expanded)

        expected = big_cpu.clone()
        expected[1:4, 2:6] = src_cpu.expand(3, 4)
        _eq(big_dev, expected)

    # ---- aliasing ----

    def test_self_copy_is_noop(self):
        """t.copy_(t) on identical views must not corrupt data."""
        t_cpu = _arange_like((3, 5), torch.float32)
        t_dev = _to_dev(t_cpu)
        t_dev.copy_(t_dev)
        _eq(t_dev, t_cpu)

    def test_self_view_copy_full_overlap(self):
        """t.copy_(t.view(t.shape)) — same storage / offset / strides → no-op path."""
        t_cpu = _arange_like((4, 4), torch.int32)
        t_dev = _to_dev(t_cpu)
        same = t_dev.view(t_dev.shape)
        t_dev.copy_(same)
        _eq(t_dev, t_cpu)

    # ---- numel / shape edge cases ----

    @dtypes(torch.float16, torch.int64)
    @parametrize("shape", [(0,), (0, 5), (3, 0), (2, 0, 4)])
    def test_zero_numel_copy(self, dtype, shape):
        src_cpu = torch.empty(shape, dtype=dtype)
        src_dev = _to_dev(src_cpu) if src_cpu.numel() > 0 else torch.empty(shape, dtype=dtype, device=DEVICE)
        dst_dev = torch.empty(shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_dev)
        assert tuple(dst_dev.shape) == shape

    @dtypes(torch.float32)
    def test_high_rank_5d(self, dtype):
        shape = (2, 3, 2, 3, 4)
        src_cpu = _arange_like(shape, dtype)
        src_dev = _to_dev(src_cpu)
        src_view = src_dev[:, 1:, :, :2, :]
        dst_dev = torch.empty(src_view.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_view)
        _eq(dst_dev, src_cpu[:, 1:, :, :2, :])

    # ---- device pair coverage (h2d / d2h / d2d) × contig matrix ----

    @dtypes(torch.float16, torch.int32)
    @parametrize("src_dev_str,dst_dev_str", [("cpu", "rbln"), ("rbln", "cpu"), ("rbln", "rbln")])
    @parametrize("src_contig", [True, False])
    @parametrize("dst_contig", [True, False])
    def test_device_pair_layout_matrix(self, dtype, src_dev_str, dst_dev_str, src_contig, dst_contig):
        """Full 3 × 2 × 2 × N device/layout/dtype matrix. Each combination must
        produce a correct copy regardless of which staging path the kernel picks."""
        src_dev = torch.device(src_dev_str + (":0" if src_dev_str == "rbln" else ""))
        dst_dev = torch.device(dst_dev_str + (":0" if dst_dev_str == "rbln" else ""))
        shape = (4, 6)

        base = _arange_like(shape, dtype) + 1
        if src_contig:
            src = base.to(src_dev)
        else:
            src = _noncontig_2d(shape, dtype, src_dev)
            src.copy_(base.to(src_dev))

        if dst_contig:
            dst = torch.empty(shape, dtype=dtype, device=dst_dev)
        else:
            dst = _noncontig_2d(shape, dtype, dst_dev)

        dst.zero_()
        dst.copy_(src)

        expected = base
        actual = dst.cpu() if dst.device != CPU else dst
        assert torch.equal(actual, expected), (
            f"Mismatch ({src_dev_str}{'_c' if src_contig else '_nc'} → "
            f"{dst_dev_str}{'_c' if dst_contig else '_nc'} dtype={dtype}):\n"
            f"  actual={actual}\n  expected={expected}"
        )

    # ---- dtype: same dtype is bitwise; cross-dtype uses CPU cast staging ----

    @parametrize("src_dtype", ENGINE_DTYPES)
    @parametrize("dst_dtype", ENGINE_DTYPES)
    def test_cross_dtype_copy_d2d(self, src_dtype, dst_dtype):
        """Cross-dtype d2d still routes correctly (CPU staging today)."""
        if src_dtype == dst_dtype:
            pytest.skip("covered by same-dtype tests")
        src_cpu = (_arange_like((3, 5), torch.int32) % 7).to(src_dtype)
        src_dev = _to_dev(src_cpu)
        dst_dev = torch.empty((3, 5), dtype=dst_dtype, device=DEVICE)
        dst_dev.copy_(src_dev)
        expected = src_cpu.to(dst_dtype)
        _close(dst_dev, expected)

    @parametrize("src_dtype,dst_dtype", [(torch.float32, torch.float16), (torch.int64, torch.int32)])
    def test_cross_dtype_with_noncontig(self, src_dtype, dst_dtype):
        """Cross-dtype + non-contig src — both staging concerns at once."""
        src_cpu = (_arange_like((4, 8), torch.int32) % 11).to(src_dtype)
        src_dev = _to_dev(src_cpu)
        src_view = src_dev[:, :6]
        dst_dev = torch.empty((4, 6), dtype=dst_dtype, device=DEVICE)
        dst_dev.copy_(src_view)
        _close(dst_dev, src_cpu[:, :6].to(dst_dtype))

    # ---- cat retrofit — verifies cat still works correctly via the shared engine ----

    @dtypes(torch.float16, torch.int32)
    @parametrize("axis", [0, 1, -1])
    def test_cat_basic_via_engine(self, dtype, axis):
        a_cpu = _arange_like((3, 4), dtype) + 1
        b_cpu = _arange_like((3, 4), dtype) + 100
        expected = torch.cat([a_cpu, b_cpu], dim=axis)
        got = torch.cat([_to_dev(a_cpu), _to_dev(b_cpu)], dim=axis)
        _eq(got, expected)

    @dtypes(torch.float32)
    def test_cat_with_noncontig_input(self, dtype):
        """One input is a non-contig view — engine handles it inside the per-input
        narrow-and-copy loop."""
        a_big_cpu = _arange_like((4, 12), dtype) + 1
        a_big_dev = _to_dev(a_big_cpu)
        a_view = a_big_dev[:, :6]
        b_cpu = _arange_like((4, 6), dtype) + 200
        b_dev = _to_dev(b_cpu)

        expected = torch.cat([a_big_cpu[:, :6], b_cpu], dim=0)
        got = torch.cat([a_view, b_dev], dim=0)
        _eq(got, expected)

    def test_cat_28_inputs_batched(self):
        """Many inputs share a single V2VBatch — verifies batch path doesn't
        silently lose entries."""
        n = 28
        inputs_cpu = [torch.full((1, 4, 8, 16), float(i), dtype=torch.bfloat16) for i in range(n)]
        expected = torch.cat(inputs_cpu, dim=0)
        inputs_dev = [_to_dev(x) for x in inputs_cpu]
        got = torch.cat(inputs_dev, dim=0)
        _eq(got, expected)

    # ---- index_select retrofit — same engine, different driver ----

    @dtypes(torch.float16, torch.int64)
    def test_index_select_consecutive_run(self, dtype):
        """Indices form one consecutive run → engine emits a single strided range."""
        src_cpu = _arange_like((6, 4), dtype)
        src_dev = _to_dev(src_cpu)
        idx = torch.tensor([2, 3, 4], dtype=torch.int64, device=DEVICE)
        expected = torch.index_select(src_cpu, 0, idx.cpu())
        got = torch.index_select(src_dev, 0, idx)
        _eq(got, expected)

    @dtypes(torch.float32)
    def test_index_select_scattered(self, dtype):
        src_cpu = _arange_like((8, 3), dtype)
        src_dev = _to_dev(src_cpu)
        idx_vals = [5, 0, 7, 2, 0]
        idx_cpu = torch.tensor(idx_vals, dtype=torch.int64)
        idx_dev = idx_cpu.to(DEVICE)
        expected = torch.index_select(src_cpu, 0, idx_cpu)
        got = torch.index_select(src_dev, 0, idx_dev)
        _eq(got, expected)

    @parametrize("axis", [0, 1, 2])
    def test_index_select_non_contig_self(self, axis):
        """index_select from a non-contig source — narrow() must preserve the
        stride pattern and the engine must walk axis correctly."""
        big_cpu = _arange_like((4, 5, 6, 7), torch.float16)
        big_dev = _to_dev(big_cpu)
        self_cpu = big_cpu[:, :, :4, :]
        self_dev = big_dev[:, :, :4, :]
        idx = torch.tensor([0, 1, 1, 0], dtype=torch.int64, device=DEVICE)
        expected = torch.index_select(self_cpu, axis, idx.cpu())
        got = torch.index_select(self_dev, axis, idx)
        _eq(got, expected)

    # ---- mass parametrized sanity: random-ish strided patterns across dtype × shape ----

    @dtypes(torch.float16, torch.int32)
    @parametrize(
        "full_shape,sliced,descr",
        _STRIDED_PATTERNS,
    )
    def test_strided_pattern_d2d(self, dtype, full_shape, sliced, descr):
        base_cpu = _arange_like(full_shape, dtype) + 1
        base_dev = _to_dev(base_cpu)
        src_view = base_dev[sliced]
        dst_dev = torch.empty(src_view.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(src_view)
        _eq(dst_dev, base_cpu[sliced])

    @parametrize(
        "full_shape,sliced,descr",
        _STRIDED_PATTERNS,
    )
    def test_strided_pattern_d2h(self, full_shape, sliced, descr):
        base_cpu = _arange_like(full_shape, torch.float32)
        base_dev = _to_dev(base_cpu)
        src_view = base_dev[sliced]
        actual_cpu = src_view.cpu()
        assert torch.equal(actual_cpu, base_cpu[sliced])

    @parametrize(
        "full_shape,sliced,descr",
        _STRIDED_PATTERNS,
    )
    def test_strided_pattern_h2d_nc_dst(self, full_shape, sliced, descr):
        """Non-contig dst on device, contig src on host — exercises the
        CPU→RBLN staging-then-engine path."""
        base_dev = torch.zeros(full_shape, dtype=torch.float32, device=DEVICE)
        base_cpu = torch.zeros(full_shape, dtype=torch.float32)
        src_cpu = (
            torch.arange(int(torch.tensor(full_shape).prod().item()), dtype=torch.float32).reshape(full_shape)[sliced]
            + 1
        )

        dst_view_dev = base_dev[sliced]
        dst_view_dev.copy_(src_cpu)

        expected = base_cpu.clone()
        expected[sliced] = src_cpu
        _eq(base_dev, expected)

    # ---- stress: large outer count with small inner block ----

    def test_many_small_blocks_d2d(self):
        """1024 × 1 strided slabs in one copy. Confirms we don't lose entries
        when the V2VBatch grows past trivial sizes."""
        src_cpu = _arange_like((1024, 4), torch.int32)
        src_dev = _to_dev(src_cpu)
        src_view = src_dev[:, ::2]
        dst_dev = torch.empty(src_view.shape, dtype=torch.int32, device=DEVICE)
        dst_dev.copy_(src_view)
        _eq(dst_dev, src_cpu[:, ::2])

    # ---- smoke: full coverage parametrize for hot path combinations ----

    @dtypes(*ENGINE_DTYPES)
    @parametrize(
        "shape,viewer",
        [
            ((6, 8), lambda t: t[:, :6]),
            ((6, 8), lambda t: t[2:5, :]),
            ((6, 8), lambda t: t.transpose(0, 1)),
            ((3, 4, 5), lambda t: t.permute(2, 0, 1)),
            ((3, 4, 5), lambda t: t[:, :2, :]),
            ((4, 4), lambda t: t[::2, :]),
        ],
    )
    def test_view_to_contig_d2d_grid(self, dtype, shape, viewer):
        base_cpu = _arange_like(shape, dtype) + 1
        base_dev = _to_dev(base_cpu)
        view = viewer(base_dev)
        dst_dev = torch.empty(view.shape, dtype=dtype, device=DEVICE)
        dst_dev.copy_(view)
        _eq(dst_dev, viewer(base_cpu).contiguous())


instantiate_device_type_tests(TestCopyV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
