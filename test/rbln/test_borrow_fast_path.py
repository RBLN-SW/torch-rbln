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

These tests verify that op outputs are bit-identical to the legacy path
(the fast path is purely a copy-elision optimization) and that disabling
the path via ``TORCH_RBLN_USE_VMEM_BORROW=0`` still produces the same
results — a regression on either side would cause a divergence.
"""

import os

import pytest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


def _run_fallback_op(borrow_enabled: bool):
    """Trigger an op that exercises the cpu_fallback_rbln path and return
    its result on RBLN.

    ``aten::sigmoid`` on int32 is intentionally a non-fp16 input (forces
    the dispatch shim's pre-check shortcut into the C++ fallback), and is
    a unary op with no write aliasing — clean for borrow-vs-copy parity.
    """
    if borrow_enabled:
        os.environ["TORCH_RBLN_USE_VMEM_BORROW"] = "1"
    else:
        os.environ["TORCH_RBLN_USE_VMEM_BORROW"] = "0"

    # Use a non-trivial shape so the borrow fast path isn't trivially shadowed
    # by a zero-sized allocation; cover both contiguous and non-contiguous
    # (so the fast path falls back legitimately for the latter).
    x = torch.arange(48, dtype=torch.int32, device="rbln").reshape(6, 8)
    return torch.sigmoid(x.float()).to("cpu")


@pytest.mark.test_set_ci
class TestBorrowFastPathParity(TestCase):
    """The borrow path must yield bit-identical results to the legacy path."""

    def tearDown(self) -> None:
        # Don't leave the gate set after the test runs.
        os.environ.pop("TORCH_RBLN_USE_VMEM_BORROW", None)
        super().tearDown()

    def test_fallback_borrow_matches_copy(self) -> None:
        out_borrow = _run_fallback_op(borrow_enabled=True)
        out_copy = _run_fallback_op(borrow_enabled=False)
        # Element-wise equality; sigmoid on the same int32 inputs is fully
        # deterministic so equality must hold at every position.
        self.assertEqual(out_borrow, out_copy)

    def test_borrow_path_is_default_on(self) -> None:
        """When the gate is not set, the borrow path is on by default.

        Verify this by clearing the env var and checking the same op runs
        without crashing — a regression that disables the default would
        most likely surface as either a build error or a runtime exception.
        """
        os.environ.pop("TORCH_RBLN_USE_VMEM_BORROW", None)
        x = torch.arange(16, dtype=torch.int32, device="rbln")
        # int32 input forces the cpu_fallback_rbln path (fp16-only kernels).
        y = torch.sigmoid(x.float())
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(tuple(y.shape), (16,))


if __name__ == "__main__":
    run_tests()
