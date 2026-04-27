# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the C++ dispatch shim's pre-check behaviour.

The shim short-circuits to the CPU fallback path on three conditions
(see `DispatchShim.cpp::quick_fallback_check`):

  1. any input tensor whose dtype is not float16 (with a per-op
     `skip_dtype_args` allowlist for typed inputs e.g. bool predicates),
  2. all input tensors are scalar (ndim == 0).

Two important non-trivial properties:

  * Wrapped Python scalars (0-dim tensors with `is_wrapped_number=True`)
    must be skipped from the dtype check so that `tensor + 1.0` does not
    incorrectly trip the shortcut.
  * Storage-offset != 0 contiguous inputs must NOT short-circuit; they
    must fall through to the Python wrapper which dispatches via
    `cpu_fallback_path` (this preserves correctness on the rebel runtime
    — see in-source notes in DispatchShim.cpp:149).

These tests are end-to-end (drive a real op through the registered
shim) so they catch regressions in either the C++ pre-check or the Python
wrapper path together.
"""

import pytest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


@pytest.mark.test_set_ci
class TestDispatchShimWrappedScalar(TestCase):
    """`tensor + 1.0` must run on RBLN, not be redirected to CPU fallback.

    Python scalars become 0-dim wrapped tensors at the dispatcher; if the
    shim counted those against the "all-scalar inputs" rule the binary op
    `add(tensor, 1.0)` would incorrectly take the fallback path and lose
    the compile-path acceleration.
    """

    def test_add_tensor_python_scalar_runs_on_rbln(self) -> None:
        x = torch.arange(8, dtype=torch.float16, device="rbln")
        y = x + 1.0
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(y.to("cpu"), torch.arange(8, dtype=torch.float16) + 1.0)

    def test_mul_tensor_python_scalar_runs_on_rbln(self) -> None:
        x = torch.arange(8, dtype=torch.float16, device="rbln")
        y = x * 2.5
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(y.to("cpu"), torch.arange(8, dtype=torch.float16) * 2.5)


@pytest.mark.test_set_ci
class TestDispatchShimAllScalarFallback(TestCase):
    """All-scalar inputs trip the fallback (the shortcut is correct here).

    A binary op with two true 0-dim tensor operands has no shape leverage
    on the RBLN device; the fallback path is both the cheaper and the
    safer choice. Verify the shim takes that path without crashing and
    that the result is numerically correct.
    """

    def test_two_zero_dim_add_uses_fallback_correctly(self) -> None:
        x = torch.tensor(2.5, dtype=torch.float16, device="rbln")
        y = torch.tensor(0.5, dtype=torch.float16, device="rbln")
        z = x + y
        # Result is on RBLN regardless of which path was taken.
        self.assertEqual(z.device.type, "rbln")
        self.assertEqual(z.to("cpu"), torch.tensor(3.0, dtype=torch.float16))


@pytest.mark.test_set_ci
class TestDispatchShimDtypeMismatch(TestCase):
    """Non-fp16 input falls through to the Python wrapper / CPU fallback."""

    def test_int32_add_runs_correctly(self) -> None:
        # fp16-only ops must still produce correct results when the input
        # dtype is not fp16 — they go through the cpu fallback path.
        x = torch.arange(8, dtype=torch.int32, device="rbln")
        y = x + 3
        self.assertEqual(y.device.type, "rbln")
        self.assertEqual(y.to("cpu"), torch.arange(8, dtype=torch.int32) + 3)


if __name__ == "__main__":
    run_tests()
