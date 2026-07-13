# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `aten::fill_.Scalar` implementation.

`fill_scalar_rbln_` (RBLNTensorFactories.cpp) replaces the default
`fallback_rbln` registration for `aten::fill_.Scalar`. It borrows a host
pointer into the RBLN vmemory backing `self`, writes the scalar value with
a typed std::fill_n / std::memset, and commits via
return_borrowed(updated=true). Bypasses cpu_fallback_rbln's
redispatchBoxed(CPU) + TensorIterator path.

These tests verify:
* All supported dtypes produce the same values as the CPU reference.
* In-place semantics are preserved (same tensor identity, same storage).
* Empty tensors are a no-op.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@pytest.mark.test_set_ci
class TestFillScalarRBLN(TestCase):
    """`aten::fill_.Scalar` on RBLN must match CPU bit-for-bit."""

    def _check(self, dtype, value, sizes=(8,)):
        ref = torch.empty(*sizes, dtype=dtype).fill_(value)
        x = torch.empty(*sizes, dtype=dtype, device="rbln")
        ret = x.fill_(value)
        # fill_ returns self for in-place semantics.
        self.assertTrue(ret is x)
        self.assertEqual(x.device.type, "rbln")
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(x.to("cpu"), ref)

    @parametrize(
        "dtype,value",
        [
            (torch.int64, -1),
            (torch.int64, 0),
            (torch.int64, 12345),
            (torch.int32, -1),
            (torch.int32, 7),
            (torch.int16, -2),
            (torch.int16, 17),
            (torch.int8, -3),
            (torch.int8, 9),
            (torch.uint8, 0),
            (torch.uint8, 200),
            (torch.bool, True),
            (torch.bool, False),
            (torch.float32, 3.14),
            (torch.float32, -2.5),
            (torch.float64, 1.0 / 3.0),
            (torch.float16, 1.5),
            (torch.float16, -0.25),
            (torch.bfloat16, 0.125),
        ],
    )
    def test_matches_cpu(self, dtype, value):
        self._check(dtype, value)

    def test_multi_dim_shape(self):
        # Verify shape is preserved.
        x = torch.empty(2, 3, 4, dtype=torch.int64, device="rbln")
        x.fill_(42)
        self.assertEqual(tuple(x.shape), (2, 3, 4))
        self.assertEqual(x.to("cpu"), torch.full((2, 3, 4), 42, dtype=torch.int64))

    def test_empty_tensor_noop(self):
        # numel() == 0 must short-circuit without touching the vmemory.
        x = torch.empty(0, dtype=torch.int64, device="rbln")
        x.fill_(-1)
        self.assertEqual(x.numel(), 0)

    def test_overflow_conversion_raises_without_leaking_borrow(self):
        """A checked-cast overflow (128 into int8) throws between the borrow and
        its return; the RAII guard must still release it, else the next free leaks
        the vmem. Assert it raises and the tensor stays usable (re-fill + churn)."""
        x = torch.empty(8, dtype=torch.int8, device="rbln")
        with pytest.raises(RuntimeError, match="overflow"):
            x.fill_(128)  # 128 > int8 max (127)
        # Borrow was released on the exception path: the tensor is still usable.
        x.fill_(5)
        self.assertEqual(x.to("cpu"), torch.full((8,), 5, dtype=torch.int8))
        del x  # must free cleanly (no lingering borrow on the vaddr)
        # Reuse the freed region a few times to surface a lingering borrow.
        for _ in range(3):
            t = torch.empty(8, dtype=torch.int8, device="rbln")
            t.fill_(1)
            del t


instantiate_device_type_tests(TestFillScalarRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
