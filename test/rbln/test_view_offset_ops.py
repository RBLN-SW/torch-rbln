# Owner(s): ["module: PrivateUse1"]

"""
View-offset correctness for the in-place RBLN ops ``zero_``, ``fill_.Scalar``,
and ``.item()``.

A view's ``data_ptr()`` is an interior vaddr (storage base +
``storage_offset() * itemsize``). ``zero_`` used the lazy ``mark_zeros`` path,
which takes no offset/size and zeroes the whole enclosing allocation — so
``base[2:4].zero_()`` wrongly zeroed the entire tensor. These tests pin the
invariant: mutating a view touches only the view's elements. Each case mutates
the same view on an RBLN tensor and a CPU reference, then compares the entire
backing tensor.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@pytest.mark.test_set_ci
class TestZeroViewOffset(TestCase):
    """``zero_`` must zero only the view, not the enclosing allocation."""

    def _check(self, dtype, base_shape, view_fn):
        base_rbln = torch.ones(base_shape, dtype=dtype, device="rbln")
        base_cpu = torch.ones(base_shape, dtype=dtype)
        view_fn(base_rbln).zero_()
        view_fn(base_cpu).zero_()
        # Compare the whole backing tensor — this is what catches
        # whole-allocation over-zeroing.
        self.assertEqual(base_rbln.to("cpu"), base_cpu)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
    def test_partial_slice_1d(self, dtype):
        # base[2:4] — the canonical regression; used to zero all 8 elements.
        self._check(dtype, (8,), lambda x: x[2:4])

    @parametrize("dtype", [torch.float32, torch.int64])
    def test_prefix_view_of_larger_storage(self, dtype):
        # base[0:4] — storage_offset == 0 but spans only half the allocation;
        # the nbytes guard (not just storage_offset) keeps it off the lazy path.
        self._check(dtype, (8,), lambda x: x[0:4])

    def test_row_view_2d(self):
        # Contiguous sub-view at a non-zero offset.
        self._check(torch.float32, (3, 4), lambda x: x[1])

    def test_column_view_2d(self):
        # Non-contiguous — routes through the CPU fallback.
        self._check(torch.float32, (3, 4), lambda x: x[:, 2])

    def test_strided_view_1d(self):
        self._check(torch.float32, (8,), lambda x: x[::2])

    def test_transposed_view(self):
        # Fully spans the storage but is non-contiguous.
        self._check(torch.float32, (3, 4), lambda x: x.transpose(0, 1)[1:])

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
    def test_full_tensor_control(self, dtype):
        # Full-allocation zero_ keeps the lazy mark_zeros path (no regression).
        base_rbln = torch.ones((2, 3), dtype=dtype, device="rbln")
        base_rbln.zero_()
        self.assertEqual(base_rbln.to("cpu"), torch.zeros((2, 3), dtype=dtype))

    def test_empty_view(self):
        # Zero-element view — early return, no borrow, no mark_zeros.
        base_rbln = torch.ones(8, device="rbln")
        base_cpu = torch.ones(8)
        base_rbln[4:4].zero_()
        base_cpu[4:4].zero_()
        self.assertEqual(base_rbln.to("cpu"), base_cpu)


@pytest.mark.test_set_ci
class TestFillViewOffset(TestCase):
    """``fill_.Scalar`` must write only the view's byte range."""

    def _check(self, dtype, base_shape, view_fn, value):
        base_rbln = torch.ones(base_shape, dtype=dtype, device="rbln")
        base_cpu = torch.ones(base_shape, dtype=dtype)
        view_fn(base_rbln).fill_(value)
        view_fn(base_cpu).fill_(value)
        self.assertEqual(base_rbln.to("cpu"), base_cpu)

    @parametrize("dtype,value", [(torch.float32, 5.0), (torch.int64, 7), (torch.bool, False)])
    def test_partial_slice_1d(self, dtype, value):
        self._check(dtype, (8,), lambda x: x[2:4], value)

    def test_row_view_2d(self):
        self._check(torch.float32, (3, 4), lambda x: x[1], 9.0)

    def test_column_view_2d(self):
        # Non-contiguous — CPU fallback in fill_scalar_rbln_.
        self._check(torch.float32, (3, 4), lambda x: x[:, 2], 9.0)

    def test_strided_view_1d(self):
        self._check(torch.float32, (8,), lambda x: x[::2], 3.0)


@pytest.mark.test_set_ci
class TestItemViewOffset(TestCase):
    """``.item()`` must read the element at the view's interior vaddr."""

    @parametrize("index", [0, 1, 3, 7])
    def test_item_at_offset(self, index):
        base = torch.arange(8, dtype=torch.float32, device="rbln")
        self.assertEqual(base[index].item(), float(index))

    def test_item_2d_offset(self):
        base = torch.arange(12, dtype=torch.int64, device="rbln").view(3, 4)
        self.assertEqual(base[2, 1].item(), 9)


@pytest.mark.test_set_ci
class TestBroadcastOverlapFill(TestCase):
    """``zero_``/``fill_`` on an internally-overlapping (expand/broadcast, stride-0) view must
    match CPU's overlap-tolerant fill — write each distinct storage element once — instead of
    raising a copy_ "internal overlap" error. Regression for ``expand(...).zero_()``."""

    def _check(self, dtype, base_shape, expand_fn, op):
        base_rbln = torch.ones(base_shape, dtype=dtype, device="rbln")
        base_cpu = torch.ones(base_shape, dtype=dtype)
        op(expand_fn(base_rbln))
        op(expand_fn(base_cpu))
        # Both the backing storage (base) and the broadcast view must match the CPU result.
        self.assertEqual(base_rbln.to("cpu"), base_cpu)
        self.assertEqual(expand_fn(base_rbln).to("cpu"), expand_fn(base_cpu))

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
    def test_zero_1d_expand(self, dtype):
        # The canonical repro: ones(1).expand(3).zero_() used to raise on RBLN.
        self._check(dtype, (1,), lambda x: x.expand(3), lambda v: v.zero_())

    def test_zero_2d_leading_broadcast(self):
        self._check(torch.float32, (1, 4), lambda x: x.expand(3, 4), lambda v: v.zero_())

    def test_zero_2d_trailing_broadcast(self):
        self._check(torch.float32, (2, 1), lambda x: x.expand(2, 3), lambda v: v.zero_())

    def test_zero_both_dims_broadcast(self):
        self._check(torch.int64, (1, 1), lambda x: x.expand(3, 4), lambda v: v.zero_())

    @parametrize("dtype,value", [(torch.float32, 5.0), (torch.int64, 7), (torch.bool, True)])
    def test_fill_1d_expand(self, dtype, value):
        self._check(dtype, (1,), lambda x: x.expand(4), lambda v: v.fill_(value))

    def test_fill_mixed_real_and_broadcast_dims(self):
        # dim 0 is real (stride != 0), dim 1 is broadcast (stride 0): only dim 1 collapses.
        self._check(torch.float32, (2, 1), lambda x: x.expand(2, 5), lambda v: v.fill_(3.0))

    def test_expand_zero_does_not_raise(self):
        # Exact reviewer repro, independent of the CPU-parity helper above.
        base = torch.ones(1, device="rbln")
        base.expand(3).zero_()  # must not raise
        self.assertEqual(base.to("cpu"), torch.zeros(1))
        self.assertEqual(base.expand(3).to("cpu"), torch.zeros(3))


instantiate_device_type_tests(TestZeroViewOffset, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestFillViewOffset, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestItemViewOffset, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestBroadcastOverlapFill, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
