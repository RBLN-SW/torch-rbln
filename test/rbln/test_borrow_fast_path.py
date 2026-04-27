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
from torch.testing._internal.common_utils import TestCase, run_tests


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


if __name__ == "__main__":
    run_tests()
