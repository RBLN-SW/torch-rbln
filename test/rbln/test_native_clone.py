# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `aten::clone` implementation.

`clone_rbln` (RBLNCopy.cpp) handles two paths:
* **Direct d2d (fast path):** `self` is contiguous, storage_offset() == 0,
  and the view spans the entire storage. The fresh output is allocated
  and filled with a single `c10::rbln::memcpy_v2v`, bypassing
  aten::copy_'s dispatch + TensorIterator host overhead.
* **Composite fallback (non-contig / partial-storage view):** falls
  through to the standard empty + copy_ decomposition. copy_'s
  TensorIterator handles arbitrary strided gather.

These tests verify:
* The fast path produces a tensor with the same values as the source.
* The fallback path produces correct results for arbitrary strided views
  (slice, transpose, broadcast).
* Detached storage: cloning produces a new tensor whose mutations do not
  affect the source.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@pytest.mark.test_set_ci
class TestCloneRBLNDirectD2D(TestCase):
    """The contiguous, full-storage fast path."""

    def _check(self, src_cpu: torch.Tensor) -> None:
        src_rbln = src_cpu.to("rbln")
        # Sanity-check the fast-path precondition.
        self.assertTrue(src_rbln.is_contiguous())
        self.assertEqual(src_rbln.storage_offset(), 0)

        out = src_rbln.clone()
        self.assertEqual(out.device.type, "rbln")
        self.assertEqual(out.dtype, src_cpu.dtype)
        self.assertEqual(tuple(out.shape), tuple(src_cpu.shape))
        self.assertEqual(out.to("cpu"), src_cpu)
        # Detached storage: cloning must produce an independent tensor.
        self.assertNotEqual(out.data_ptr(), src_rbln.data_ptr())

    @parametrize(
        "dtype,shape",
        [
            (torch.int64, (10,)),
            (torch.int32, (3, 4)),
            (torch.float16, (2, 3, 4)),
        ],
    )
    def test_arange_view(self, dtype, shape):
        numel = 1
        for s in shape:
            numel *= s
        self._check(torch.arange(numel, dtype=dtype).view(*shape))

    def test_bool(self):
        self._check(torch.tensor([True, False, True, True], dtype=torch.bool))

    def test_empty_tensor(self):
        # numel() == 0 — fast path is skipped (nbytes guard) but an empty
        # output of the right shape/dtype is still returned.
        src = torch.empty(0, dtype=torch.int64, device="rbln")
        out = src.clone()
        self.assertEqual(out.numel(), 0)
        self.assertEqual(out.dtype, torch.int64)


@pytest.mark.test_set_ci
class TestCloneRBLNNonContigFallback(TestCase):
    """The composite fallback path (non-contig / partial-storage views)."""

    def _check_view(self, src_cpu, view_fn):
        src_rbln = src_cpu.to("rbln")
        view_cpu = view_fn(src_cpu)
        view_rbln = view_fn(src_rbln)

        out_rbln = view_rbln.clone()
        out_cpu = view_cpu.clone()

        self.assertEqual(out_rbln.device.type, "rbln")
        self.assertEqual(out_rbln.dtype, src_cpu.dtype)
        self.assertEqual(tuple(out_rbln.shape), tuple(out_cpu.shape))
        self.assertEqual(out_rbln.to("cpu"), out_cpu)

    def test_slice_view(self):
        # 1D slice — view has storage_offset != 0.
        src = torch.arange(16, dtype=torch.int64)
        self._check_view(src, lambda x: x[4:12])

    def test_strided_view_2d(self):
        # Every second column — view is non-contiguous.
        src = torch.arange(20, dtype=torch.int32).view(4, 5)
        self._check_view(src, lambda x: x[:, ::2])

    def test_transposed_view(self):
        # transpose breaks contiguity even when the storage is fully spanned.
        src = torch.arange(12, dtype=torch.float16).view(3, 4)
        self._check_view(src, lambda x: x.transpose(0, 1))


@pytest.mark.test_set_ci
class TestCloneRBLNIsolation(TestCase):
    """Clones must not alias the source's storage."""

    def test_mutation_does_not_propagate(self):
        src = torch.arange(8, dtype=torch.int64, device="rbln")
        out = src.clone()
        out.fill_(-1)
        # `src` must keep its original values.
        self.assertEqual(src.to("cpu"), torch.arange(8, dtype=torch.int64))
        self.assertEqual(out.to("cpu"), torch.full((8,), -1, dtype=torch.int64))


instantiate_device_type_tests(TestCloneRBLNDirectD2D, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCloneRBLNNonContigFallback, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCloneRBLNIsolation, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
