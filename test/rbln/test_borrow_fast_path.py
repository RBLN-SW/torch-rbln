# Owner(s): ["module: PrivateUse1"]

"""
End-to-end test for the V3 v-mem borrow fast path in
``cpu_fallback_rbln`` (see ``aten/src/ATen/native/rbln/RBLNCPUFallback.cpp``).

The fast path replaces the legacy ``at::_to_cpu`` /
``at::_copy_from_and_resize`` round-trip with a borrow on the rebel virtual
memory: when an op falls through to the CPU kernel, contiguous rbln-device
inputs are wrapped as CPU tensors that alias the host backing of the same
v-mem allocation. After the CPU op runs, the borrow is released and any
write-aliasing input is propagated back to the device.

The borrow path is always on; this test verifies a representative fallback
op produces correct results (a copy-elision regression would manifest as
wrong values or a crash).
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestBorrowFastPath(TestCase):
    """The borrow path must yield correct results for ops routed to the
    CPU fallback. ``aten::sigmoid`` on int32 is intentionally a non-fp16
    input (forces the dispatch shim's pre-check shortcut into the C++
    fallback), and is a unary op with no write aliasing — clean for
    borrow correctness."""

    def test_fallback_unary_int32_input(self) -> None:
        x = torch.arange(48, dtype=torch.int32, device="rbln").reshape(6, 8)
        out = torch.sigmoid(x.float()).to("cpu")
        # Reference computation on CPU.
        ref = torch.sigmoid(torch.arange(48, dtype=torch.int32).reshape(6, 8).float())
        self.assertEqual(out, ref)

    def test_borrow_path_default_on(self) -> None:
        """Sanity: a rbln tensor flowing through cpu_fallback_rbln does not
        crash and returns a tensor on the rbln device with the expected
        shape. A regression that disabled the default would most likely
        surface as a build error or a runtime exception."""
        x = torch.arange(16, dtype=torch.int32, device="rbln")
        # int32 input forces the cpu_fallback_rbln path (fp16-only kernels).
        y = torch.sigmoid(x.float())
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(tuple(y.shape), (16,))

    def test_borrow_rejected_falls_back_to_copy(self) -> None:
        """Regression: a 0-K matmul yields an all-zero (5, 10) output. Writing
        it into an ``out=`` tensor that already carries a host user view (from
        ``full``) leaves the rebel vmem entry in a sub-state whose in-place host
        borrow is rejected. Reading that tensor through cpu_fallback (and the
        dispatch shim's NaN/Inf pre-check) must fall back to a D2H copy instead
        of raising ``rbln_v_borrow_host_ptr failed`` (the PrepareUserViewBuffer
        assertion). Comparing two rbln tensors routes ``eq`` through the borrow
        path, so this exercises the fallback and checks correctness at once."""
        a = torch.zeros(5, 0, dtype=torch.float16, device="rbln")
        b = torch.zeros(0, 10, dtype=torch.float16, device="rbln")
        out = torch.full((5, 10), float("nan"), dtype=torch.float16, device="rbln")
        torch.mm(a, b, out=out)
        # rbln-vs-rbln compare -> isclose/eq -> cpu_fallback borrow on `out`.
        self.assertEqual(out, torch.zeros(5, 10, dtype=torch.float16, device="rbln"))


instantiate_device_type_tests(TestBorrowFastPath, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
