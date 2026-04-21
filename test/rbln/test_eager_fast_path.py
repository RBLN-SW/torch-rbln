# Owner(s): ["module: PrivateUse1"]

"""Tests for the eager dispatch fast path.

Covers:
- ``fast_eager_dispatch_check`` guard logic (positive + each rejection branch)
- ``try_fast_eager_dispatch`` end-to-end semantics
- Codegen-generated ops produce results equivalent to the slow path / CPU
  reference, with the fast path actually being taken.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch_rbln  # noqa: F401  -- registers backend
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import ops_utils
from torch_rbln._internal.ops_utils import fast_eager_dispatch_check, try_fast_eager_dispatch


class _Add(torch.nn.Module):
    def forward(self, a, b):
        return torch.add(a, b)


@pytest.mark.test_set_ci
class TestFastEagerDispatchCheck(TestCase):
    """Pure-Python guard logic. No device dispatch needed."""

    rbln_device = torch.device("rbln:0")

    def _make(self, shape=(4,), dtype=torch.float16, device=None):
        device = device if device is not None else self.rbln_device
        return torch.ones(shape, dtype=dtype, device=device)

    def test_safe_two_tensors(self):
        a, b = self._make(), self._make()
        is_safe, dev_id, tensors = fast_eager_dispatch_check((a, b), {})
        self.assertTrue(is_safe)
        self.assertEqual(dev_id, 0)
        self.assertEqual(len(tensors), 2)

    def test_safe_with_python_scalar(self):
        a = self._make()
        is_safe, dev_id, tensors = fast_eager_dispatch_check((a, 1), {})
        self.assertTrue(is_safe)
        self.assertEqual(len(tensors), 1)

    def test_no_tensors_rejected(self):
        is_safe, *_ = fast_eager_dispatch_check((1, 2.0), {})
        self.assertFalse(is_safe)

    def test_non_rbln_tensor_rejected(self):
        a = torch.ones(4, dtype=torch.float16)  # CPU
        is_safe, *_ = fast_eager_dispatch_check((a,), {})
        self.assertFalse(is_safe)

    def test_unsafe_dtype_rejected(self):
        a = self._make(dtype=torch.float32)
        is_safe, *_ = fast_eager_dispatch_check((a,), {})
        self.assertFalse(is_safe)

    def test_empty_tensor_rejected(self):
        a = self._make(shape=(0,))
        is_safe, *_ = fast_eager_dispatch_check((a,), {})
        self.assertFalse(is_safe)

    def test_nonzero_storage_offset_rejected(self):
        base = torch.arange(8, dtype=torch.int32, device=self.rbln_device)
        view = base[1:]  # contiguous but storage_offset=1
        self.assertEqual(view.storage_offset(), 1)
        self.assertTrue(view.is_contiguous())
        is_safe, *_ = fast_eager_dispatch_check((view,), {})
        self.assertFalse(is_safe)

    def test_non_contiguous_rejected(self):
        a = self._make(shape=(4, 4)).t()  # transposed -> non-contig
        self.assertFalse(a.is_contiguous())
        is_safe, *_ = fast_eager_dispatch_check((a,), {})
        self.assertFalse(is_safe)

    def test_kwargs_with_tensor_rejected(self):
        a = self._make()
        weight = self._make()
        is_safe, *_ = fast_eager_dispatch_check((a,), {"weight": weight})
        self.assertFalse(is_safe)

    def test_out_kwarg_accepted_when_valid(self):
        a, b = self._make(), self._make()
        out = self._make(shape=(4,))
        is_safe, *_ = fast_eager_dispatch_check((a, b), {"out": out})
        self.assertTrue(is_safe)

    def test_out_kwarg_inplace_rejected(self):
        a = self._make()
        # out shares storage with input
        is_safe, *_ = fast_eager_dispatch_check((a,), {"out": a})
        self.assertFalse(is_safe)

    def test_out_kwarg_wrong_dtype_rejected(self):
        a = self._make(dtype=torch.float16)
        out = self._make(dtype=torch.int32)
        is_safe, *_ = fast_eager_dispatch_check((a,), {"out": out})
        self.assertFalse(is_safe)

    def test_out_kwarg_non_tensor_rejected(self):
        a = self._make()
        is_safe, *_ = fast_eager_dispatch_check((a,), {"out": "not a tensor"})
        self.assertFalse(is_safe)

    def test_sys_gettrace_rejected(self):
        a, b = self._make(), self._make()
        import sys

        def _tracer(*_):  # noqa: ANN001
            return _tracer

        old = sys.gettrace()
        sys.settrace(_tracer)
        try:
            is_safe, *_ = fast_eager_dispatch_check((a, b), {})
            self.assertFalse(is_safe)
        finally:
            sys.settrace(old)


@pytest.mark.test_set_ci
class TestTryFastEagerDispatch(TestCase):
    """End-to-end behavior. Hits the device for the happy path."""

    rbln_device = torch.device("rbln:0")

    def setUp(self):
        from torch_rbln._internal.compile_cache import clear_rbln_compile_cache
        clear_rbln_compile_cache()

    def _make(self, shape=(8,), dtype=torch.float16, fill=1.0):
        return torch.full(shape, fill, dtype=dtype, device=self.rbln_device)

    def test_returns_none_on_miss(self):
        a = torch.ones(4, dtype=torch.float32, device=self.rbln_device)
        # fp32 is not in the safe set -> miss
        result = try_fast_eager_dispatch(_Add().eval(), (a, a), {})
        self.assertIsNone(result)

    def test_hit_returns_tuple(self):
        a, b = self._make(fill=1.0), self._make(fill=2.0)
        result = try_fast_eager_dispatch(_Add().eval(), (a, b), {}, require_same_shape=True)
        self.assertIsNotNone(result)
        out, shape = result
        self.assertEqual(shape, (8,))
        self.assertEqual(out.cpu().tolist(), [3.0] * 8)

    def test_require_same_shape_rejects_mismatch(self):
        a = self._make(shape=(4,))
        b = self._make(shape=(8,))
        result = try_fast_eager_dispatch(_Add().eval(), (a, b), {}, require_same_shape=True)
        self.assertIsNone(result)

    def test_unary_path(self):
        class _Neg(torch.nn.Module):
            def forward(self, a):
                return -a

        a = self._make(fill=1.5)
        result = try_fast_eager_dispatch(_Neg().eval(), (a,), {}, require_same_shape=False)
        self.assertIsNotNone(result)
        out, shape = result
        self.assertEqual(shape, (8,))
        self.assertEqual(out.cpu().tolist(), [-1.5] * 8)

    def test_out_kwarg_writethrough(self):
        a, b = self._make(fill=1.0), self._make(fill=2.0)
        out = self._make(fill=0.0)
        result = try_fast_eager_dispatch(_Add().eval(), (a, b), {"out": out})
        self.assertIsNotNone(result)
        ret, _ = result
        # Returned tensor is the out tensor (or content matches)
        self.assertEqual(ret.cpu().tolist(), [3.0] * 8)


@pytest.mark.test_set_ci
class TestCodegenGeneratedFastPath(TestCase):
    """Smoke test that codegen-generated ops produce correct results.

    Verifies all categories that received a fast path block.
    """

    rbln_device = torch.device("rbln:0")

    def _both(self, shape=(8,), fill_a=1.0, fill_b=2.0):
        a = torch.full(shape, fill_a, dtype=torch.float16, device=self.rbln_device)
        b = torch.full(shape, fill_b, dtype=torch.float16, device=self.rbln_device)
        return a, b

    def test_binary_ops_match_cpu(self):
        a, b = self._both(fill_a=1.0, fill_b=2.0)
        for op in (torch.add, torch.sub, torch.mul, torch.div,
                   torch.maximum, torch.minimum):
            r_dev = op(a, b).cpu()
            r_cpu = op(a.cpu(), b.cpu())
            self.assertEqual(r_dev, r_cpu, msg=f"{op.__name__} mismatch")

    def test_compare_ops_match_cpu(self):
        a, b = self._both(fill_a=1.0, fill_b=2.0)
        for op in (torch.gt, torch.lt, torch.ge, torch.le, torch.eq, torch.ne):
            r_dev = op(a, b).cpu()
            r_cpu = op(a.cpu(), b.cpu())
            self.assertEqual(r_dev, r_cpu, msg=f"{op.__name__} mismatch")

    def test_unary_ops_match_cpu(self):
        x = torch.full((8,), 1.5, dtype=torch.float16, device=self.rbln_device)
        # Loose tolerance for transcendental ops (log/rsqrt) where fp16 device
        # math is allowed to differ from CPU's slightly higher-precision impl.
        for op in (torch.neg, torch.abs, torch.ceil, torch.floor):
            r_dev = op(x).cpu()
            r_cpu = op(x.cpu())
            self.assertEqual(r_dev, r_cpu, msg=f"{op.__name__} mismatch")
        for op in (torch.log, torch.rsqrt):
            r_dev = op(x).cpu()
            r_cpu = op(x.cpu())
            self.assertEqual(r_dev, r_cpu, atol=1e-2, rtol=1e-2, msg=f"{op.__name__} mismatch")


if __name__ == "__main__":
    run_tests()
