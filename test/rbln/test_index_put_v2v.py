# Owner(s): ["module: PrivateUse1"]

"""Tests for the v2v ``aten::_index_put_impl_`` kernel (``index_put_``, ``x[...] = v``).

Coverage is organized by behavior:
  - Point writes: the vLLM min_tokens pattern, per-point values, slabs under a
    prefix index, broadcast indices
  - Index patterns: negative, duplicate (last write wins), empty
  - Placement and layout: host indices and host scalar, non-contiguous views
  - Upstream's index forms: ``None`` slots, non-adjacent indices, masks
  - Error paths, raised by the kernel rather than the CPU fallback
  - ``accumulate=True``, which stays on the CPU

The CPU fallback gives the same values, so every device-path test also asserts
that ``_index_put_impl_`` did not fall back. Results are compared bitwise with
the CPU reference: the op only moves elements.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import arange as _arange, DEVICE, ENGINE_DTYPES, eq as _eq, to_dev as _to_dev
from torch_rbln import _C


_OP = "_index_put_impl_"

_VIEWS = {
    "transposed": lambda t: t.t(),
    "row_slice": lambda t: t[2:10],
    "col_slice": lambda t: t[:, :64],
    "step_slice": lambda t: t[:, ::2],
}


def _fallbacks(fn, ops=(_OP,)) -> int:
    """Run ``fn``; return how often any of ``ops`` fell back to the CPU."""
    torch.rbln.synchronize()
    _C._dispatch_fallback_by_op_reset()
    fn()
    torch.rbln.synchronize()
    return sum(count for name, count in dict(_C._dispatch_fallback_by_op()).items() if any(op in name for op in ops))


@pytest.mark.npu(count=1)
@pytest.mark.usefixtures("enable_deploy_mode")
class TestIndexPutV2V(TestCase):
    def _check(self, self_cpu, indices_cpu, values_cpu, *, index_device=DEVICE, value_device=DEVICE):
        expected = self_cpu.clone().index_put_(indices_cpu, values_cpu)
        got = _to_dev(self_cpu)
        indices = tuple(i.to(index_device) for i in indices_cpu)
        values = values_cpu.to(value_device)
        self.assertEqual(_fallbacks(lambda: got.index_put_(indices, values)), 0, msg="fell back to the CPU")
        _eq(got, expected)

    def _check_setitem(self, self_cpu, key, values_cpu):
        """``x[key] = values``; ``key(device)`` builds the key on that device."""
        expected = self_cpu.clone()
        expected[key("cpu")] = values_cpu
        got = _to_dev(self_cpu)
        key_d, values = key(DEVICE), values_cpu.to(DEVICE)

        def setitem():
            got[key_d] = values

        self.assertEqual(_fallbacks(setitem), 0, msg="fell back to the CPU")
        _eq(got, expected)

    def _check_raises_in_kernel(self, fn, regex):
        def run():
            with self.assertRaisesRegex(IndexError, regex):
                fn()

        self.assertEqual(_fallbacks(run), 0, msg="raised from the CPU fallback, not the kernel")

    # ---- point writes ----

    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_point_write_scalar_value(self, index_dtype):
        """vLLM's min_tokens: (row, stop token) points set to a 0-D -inf."""
        logits = torch.randn(8, 256).to(torch.bfloat16)
        rows = torch.tensor([0, 0, 3, 5, 5, 5, 7], dtype=index_dtype)
        toks = torch.tensor([2, 255, 0, 17, 64, 128, 31], dtype=index_dtype)
        self._check(logits, (rows, toks), torch.tensor(-float("inf"), dtype=torch.bfloat16))

    @dtypes(*ENGINE_DTYPES)
    def test_point_write_per_point_values(self, dtype):
        self_cpu = _arange((6, 64), dtype)
        rows = torch.tensor([5, 0, 2])
        cols = torch.tensor([63, 1, 32])
        self._check(self_cpu, (rows, cols), torch.tensor([-1, -2, -3], dtype=dtype))

    def test_prefix_index_writes_slabs(self):
        """Fewer indices than dims: each point names a trailing slab."""
        self_cpu = _arange((4, 3, 64), torch.float32)
        rows = torch.tensor([3, 1])
        mids = torch.tensor([0, 2])
        self._check(self_cpu, (rows, mids), _arange((2, 64), torch.float32) + 1000)

    def test_prefix_index_broadcast_value(self):
        self_cpu = _arange((5, 64), torch.bfloat16)
        self._check(self_cpu, (torch.tensor([4, 0, 2]),), torch.tensor(7.0, dtype=torch.bfloat16))

    def test_broadcast_indices(self):
        """(2, 1) and (1, 3) indices broadcast to six points."""
        self_cpu = _arange((4, 64), torch.float16)
        rows = torch.tensor([[0], [3]])
        cols = torch.tensor([[1, 10, 63]])
        self._check(self_cpu, (rows, cols), _arange((2, 3), torch.float16) + 500)

    # ---- index patterns ----

    def test_negative_indices(self):
        self_cpu = _arange((4, 64), torch.float32)
        self._check(self_cpu, (torch.tensor([-1, 0]), torch.tensor([-64, -1])), torch.tensor([9.0, 8.0]))

    def test_duplicate_points_last_write_wins(self):
        self_cpu = _arange((3, 64), torch.float32)
        rows = torch.tensor([1, 2, 1, 1])
        cols = torch.tensor([5, 0, 5, 5])
        self._check(self_cpu, (rows, cols), torch.tensor([10.0, 20.0, 30.0, 40.0]))

    def test_empty_index_is_noop(self):
        self_cpu = _arange((3, 64), torch.float16)
        empty = torch.tensor([], dtype=torch.int32)
        self._check(self_cpu, (empty, empty), torch.tensor(-1.0, dtype=torch.float16))

    # ---- placement and layout ----

    def test_cpu_indices_and_cpu_scalar_value(self):
        self_cpu = _arange((4, 64), torch.float16)
        self._check(
            self_cpu,
            (torch.tensor([1, 3]), torch.tensor([2, 60])),
            torch.tensor(-3.0, dtype=torch.float16),
            index_device="cpu",
            value_device="cpu",
        )

    @parametrize("view", list(_VIEWS))
    def test_view_writes_through(self, view):
        """Writes land in the base; a column slice has gaps but no self-aliasing."""
        base_cpu = _arange((16, 128), torch.float16)
        base = _to_dev(base_cpu)
        rows = torch.tensor([0, 5, 7])
        cols = torch.tensor([0, 3, 15])
        values_cpu = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float16)
        expected = base_cpu.clone()
        _VIEWS[view](expected).index_put_((rows, cols), values_cpu)
        idx = (rows.to(DEVICE), cols.to(DEVICE))
        values = values_cpu.to(DEVICE)
        self.assertEqual(_fallbacks(lambda: _VIEWS[view](base).index_put_(idx, values)), 0)
        _eq(base, expected)

    # ---- None slots, non-adjacent indices, masks ----

    def test_none_slot_per_element_values(self):
        """``x[:, idx]``: values are (3, 2, 64), the indexed dim in place."""
        self_cpu = _arange((3, 4, 64), torch.float16)
        values = _arange((3, 2, 64), torch.float16) + 1000
        self._check_setitem(self_cpu, lambda d: (slice(None), torch.tensor([3, 1], device=d)), values)

    def test_non_adjacent_indices(self):
        """``x[i, :, j]``: the broadcast index dims go first, so values are (2, 3)."""
        self_cpu = _arange((4, 3, 64), torch.float32)
        values = _arange((2, 3), torch.float32) + 1000
        self._check_setitem(
            self_cpu, lambda d: (torch.tensor([3, 0], device=d), slice(None), torch.tensor([63, 7], device=d)), values
        )

    def test_mask_with_host_scalar_goes_to_masked_fill(self):
        """Upstream's canDispatchToMaskedFill route; neither op falls back."""
        self_cpu = torch.randn(4, 64, dtype=torch.float16)
        mask_cpu = self_cpu > 0
        expected = self_cpu.clone().index_put_((mask_cpu,), torch.tensor(-1.0, dtype=torch.float16))
        got = _to_dev(self_cpu)
        mask = mask_cpu.to(DEVICE)
        value = torch.tensor(-1.0, dtype=torch.float16)
        self.assertEqual(_fallbacks(lambda: got.index_put_((mask,), value), ops=(_OP, "masked_fill")), 0)
        _eq(got, expected)

    def test_mask_with_device_values(self):
        """A device ``values`` skips masked_fill; the mask expands to points."""
        self_cpu = _arange((4, 64), torch.float32)
        mask = (_arange((4, 64), torch.int64) % 7) == 0
        values = torch.arange(int(mask.sum()), dtype=torch.float32) + 1000
        self._check(self_cpu, (mask,), values)

    def test_mask_and_index_mixed(self):
        self_cpu = _arange((4, 3, 64), torch.float16)
        mask = torch.tensor([True, False, True])
        cols = torch.tensor([1, 62])
        self._check(self_cpu, (torch.tensor([0, 3]), mask, cols), torch.tensor(-2.0, dtype=torch.float16))

    # ---- error paths ----

    def test_out_of_bounds_raises_index_error(self):
        got = _to_dev(_arange((3, 64), torch.float32))
        idx = (torch.tensor([3], device=DEVICE), torch.tensor([0], device=DEVICE))
        self._check_raises_in_kernel(lambda: got.index_put_(idx, torch.tensor(1.0, device=DEVICE)), "out of bounds")

    def test_mask_shape_mismatch_raises_index_error(self):
        got = _to_dev(_arange((4, 64), torch.float32))
        mask = torch.ones(3, dtype=torch.bool, device=DEVICE)
        self._check_raises_in_kernel(
            lambda: got.index_put_((mask,), torch.tensor(1.0, device=DEVICE)), "shape of the mask"
        )

    def test_unbroadcastable_indices_raise_index_error(self):
        got = _to_dev(_arange((4, 64), torch.float32))
        idx = (torch.tensor([0, 1], device=DEVICE), torch.tensor([0, 1, 2], device=DEVICE))
        self._check_raises_in_kernel(lambda: got.index_put_(idx, torch.tensor(1.0, device=DEVICE)), "shape mismatch")

    # ---- accumulate stays on the CPU ----

    def test_accumulate_falls_back(self):
        self_cpu = _arange((3, 64), torch.float32)
        idx = (torch.tensor([0, 0]), torch.tensor([1, 1]))
        expected = self_cpu.clone().index_put_(idx, torch.tensor([1.0, 2.0]), accumulate=True)
        got = _to_dev(self_cpu)
        idx_d = tuple(i.to(DEVICE) for i in idx)
        values = torch.tensor([1.0, 2.0], device=DEVICE)
        self.assertGreater(_fallbacks(lambda: got.index_put_(idx_d, values, accumulate=True)), 0)
        _eq(got, expected)


instantiate_device_type_tests(TestIndexPutV2V, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
