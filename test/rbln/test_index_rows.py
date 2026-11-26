# Owner(s): ["module: PrivateUse1"]

"""Tests for the row kernel behind ``x[(rows, cols)]`` on a 2-D tensor.

``index.Tensor_out`` and ``_index_put_impl_`` hand two 1-D integer indices into a
contiguous 2-D half/bfloat16 tensor whose last dim is a multiple of 64 to
``torch_rbln/_internal/kernels/index_rows.py``: vLLM's logit bias
(``logits[(req, tok)] += bias``) and min-tokens (``logits[(req, tok)] = -inf``).

Values are integers in [-512, 512), which the device holds exactly (it rounds
fp16 to a shorter mantissa in compiled programs), so results compare bitwise
with the CPU. Each test also checks that the row kernel ran and nothing fell
back to the CPU.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils_v2v import DEVICE, eq as _eq, to_dev as _to_dev
from torch_rbln import _C
from torch_rbln._internal.kernels import index_rows


_OPS = ("index", "_index_put_impl_")
_B, _V = 8, 4096


def _values(shape, dtype) -> torch.Tensor:
    return torch.randint(-512, 512, shape).to(dtype)


def _points(k: int, *, b: int = _B, v: int = _V, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, b, (k,), generator=g), torch.randint(0, v, (k,), generator=g)


@pytest.mark.npu(count=1)
@pytest.mark.usefixtures("enable_deploy_mode")
class TestIndexRows(TestCase):
    def _run(self, fn, *, row_kernel: bool = True):
        """Run ``fn``; check that the row kernel ran (or not) and nothing fell back."""
        if row_kernel and not index_rows._row_ops_available():
            self.skipTest("rebel has no rbln::batched_*_rows ops")
        torch.rbln.synchronize()
        _C._dispatch_fallback_by_op_reset()
        with mock.patch.object(index_rows, "_run", wraps=index_rows._run) as run:
            result = fn()
        torch.rbln.synchronize()
        fallbacks = sum(n for name, n in dict(_C._dispatch_fallback_by_op()).items() if any(op in name for op in _OPS))
        self.assertEqual(fallbacks, 0, msg="fell back to the CPU")
        self.assertEqual(run.called, row_kernel, msg="row kernel ran" if not row_kernel else "row kernel did not run")
        return result

    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(torch.float16, torch.bfloat16)
    def test_gather(self, device, dtype, index_dtype):  # noqa: ARG002
        x = _values((_B, _V), dtype)
        rows, cols = (i.to(index_dtype) for i in _points(100))
        got = self._run(lambda: _to_dev(x)[(rows.to(DEVICE), cols.to(DEVICE))])
        _eq(got, x[(rows, cols)])

    @dtypes(torch.float16, torch.bfloat16)
    def test_put(self, device, dtype):  # noqa: ARG002
        x, v = _values((_B, _V), dtype), _values((100,), dtype)
        rows, cols = _points(100)
        got = _to_dev(x)
        self._run(lambda: got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE)))
        _eq(got, x.clone().index_put_((rows, cols), v))

    def test_put_scalar_value(self, device):  # noqa: ARG002
        """vLLM's min_tokens: (request, stop token) points set to a 0-D -inf."""
        x = _values((_B, _V), torch.bfloat16)
        rows, cols = _points(20)
        neg_inf = torch.tensor(-float("inf"), dtype=torch.bfloat16)
        got = _to_dev(x)

        def setitem():
            got[(rows.to(DEVICE), cols.to(DEVICE))] = neg_inf.to(DEVICE)

        self._run(setitem)
        expected = x.clone()
        expected[(rows, cols)] = neg_inf
        _eq(got, expected)

    def test_put_accumulate(self, device):  # noqa: ARG002
        x, v = _values((_B, _V), torch.float16), _values((100,), torch.float16)
        rows, cols = _points(100)
        got = _to_dev(x)
        self._run(lambda: got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE), accumulate=True))
        _eq(got, x.clone().index_put_((rows, cols), v, accumulate=True))

    def test_logit_bias(self, device):  # noqa: ARG002
        """vLLM's logit bias, ``logits[(req, tok)] += bias``: gather, add, put."""
        x, bias = _values((_B, _V), torch.float16), _values((64,), torch.float16)
        rows, cols = _points(64)
        got = _to_dev(x)
        key = (rows.to(DEVICE), cols.to(DEVICE))
        bias_d = bias.to(DEVICE)

        def logit_bias():
            got[key] += bias_d

        self._run(logit_bias)
        expected = x.clone()
        expected[(rows, cols)] += bias
        _eq(got, expected)

    def test_points_sharing_rows(self, device):  # noqa: ARG002
        """Points in one 64-wide row share a slot; a repeated point keeps its last."""
        x, v = _values((_B, _V), torch.float16), _values((12,), torch.float16)
        rows = torch.tensor([0, 0, 0, 0, 3, 3, 3, 7, 7, 7, 7, 7])
        cols = torch.tensor([5, 6, 63, 5, 64, 100, 127, 0, 1, 2, 3, 0])
        got = _to_dev(x)
        self._run(lambda: got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE)))
        _eq(got, x.clone().index_put_((rows, cols), v))

        got = _to_dev(x)
        self._run(lambda: got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE), accumulate=True))
        _eq(got, x.clone().index_put_((rows, cols), v, accumulate=True))

        got = self._run(lambda: _to_dev(x)[(rows.to(DEVICE), cols.to(DEVICE))])
        _eq(got, x[(rows, cols)])

    def test_negative_and_host_indices(self, device):  # noqa: ARG002
        x, v = _values((_B, _V), torch.float16), _values((30,), torch.float16)
        rows, cols = _points(30)
        neg_rows, neg_cols = rows - _B, cols - _V
        got = _to_dev(x)
        self._run(lambda: got.index_put_((neg_rows, neg_cols), v.to(DEVICE)))
        _eq(got, x.clone().index_put_((rows, cols), v))

    @parametrize("inference", [False, True])
    def test_index_changed_in_place(self, device, inference):  # noqa: ARG002
        """The plan is cached per index tensor; writing the index rebuilds it, also
        for inference tensors (vLLM's), which have no version counter.
        """
        x = _values((_B, _V), torch.float16)
        rows, cols = _points(40)
        new_cols = (cols + 1000) % _V
        with torch.inference_mode(inference):
            rows_d, cols_d = rows.to(DEVICE), cols.to(DEVICE)
            dx = _to_dev(x)
            _eq(self._run(lambda: dx[(rows_d, cols_d)]), x[(rows, cols)])
            _eq(self._run(lambda: dx[(rows_d, cols_d)]), x[(rows, cols)])
            cols_d.copy_(new_cols.to(DEVICE))
            _eq(self._run(lambda: dx[(rows_d, cols_d)]), x[(rows, new_cols)])

    def test_out_of_bounds_raises(self, device):  # noqa: ARG002
        x = _to_dev(_values((_B, _V), torch.float16))
        rows = torch.tensor([0, 1]).to(DEVICE)
        cols = torch.tensor([0, _V]).to(DEVICE)
        with self.assertRaisesRegex(IndexError, f"index {_V} is out of bounds for dimension 1"):
            x[(rows, cols)]

    @parametrize("shape", [(_B, _V - 1), (_B, 256)])
    def test_other_shapes_take_other_paths(self, device, shape):  # noqa: ARG002
        """A last dim not a multiple of 64, or too few 64-wide rows for the program."""
        x, v = _values(shape, torch.float16), _values((10,), torch.float16)
        rows, cols = _points(10, v=shape[1])
        got = _to_dev(x)
        torch.rbln.synchronize()
        with mock.patch.object(index_rows, "_run", wraps=index_rows._run) as run:
            got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE))
        self.assertFalse(run.called)
        _eq(got, x.clone().index_put_((rows, cols), v))

    @parametrize("why", ["env_off", "no_rebel_ops"])
    def test_turned_off_takes_point_path(self, device, why):  # noqa: ARG002
        """TORCH_RBLN_INDEX_ROW_KERNEL=OFF, or a rebel without the batched row ops."""
        x, v = _values((_B, _V), torch.float16), _values((10,), torch.float16)
        rows, cols = _points(10)
        got = _to_dev(x)
        if why == "env_off":
            off = mock.patch.dict("os.environ", {"TORCH_RBLN_INDEX_ROW_KERNEL": "OFF"})
        else:
            off = mock.patch.object(index_rows, "_row_ops_ok", new=False)
        with off:
            self._run(lambda: got.index_put_((rows.to(DEVICE), cols.to(DEVICE)), v.to(DEVICE)), row_kernel=False)
        _eq(got, x.clone().index_put_((rows, cols), v))


instantiate_device_type_tests(TestIndexRows, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
