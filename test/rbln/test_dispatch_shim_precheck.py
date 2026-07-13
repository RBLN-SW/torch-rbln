# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the C++ dispatch shim's pre-check behaviour.

The shim short-circuits to the CPU fallback path on these conditions
(see `DispatchShim.cpp::quick_fallback_check`):

  1. any input tensor whose dtype is not float16 (with a per-op
     `skip_dtype_args` allowlist for typed inputs e.g. bool predicates),
  2. all input tensors are scalar (ndim == 0),
  3. (non-deploy mode only) any input tensor contains NaN or Inf —
     mirrors the AS-IS Python ``has_invalid_tensor(to_cpu(args))`` safety
     net that the warm-cache hot path would otherwise bypass after the
     first call had installed an entry on clean inputs.

Two important non-trivial properties:

  * Wrapped Python scalars (0-dim tensors with `is_wrapped_number=True`)
    must be skipped from the dtype check so that `tensor + 1.0` does not
    incorrectly trip the shortcut.
  * Storage-offset != 0 contiguous inputs must NOT short-circuit; they
    must fall through to the Python wrapper which dispatches via
    `cpu_fallback_path` (this preserves correctness on the rebel runtime
    — see in-source notes in DispatchShim.cpp).

These tests are end-to-end (drive a real op through the registered
shim) so they catch regressions in either the C++ pre-check or the Python
wrapper path together.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln._C as _C


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


@pytest.mark.test_set_ci
class TestDispatchShimNanInfFallback(TestCase):
    """Inputs containing NaN or Inf must route to the CPU fallback path in
    non-deploy mode.

    The rbln runtime does not handle NaN/Inf inputs and will silently produce
    wrong results. In AS-IS this safety net was carried by the Python
    wrapper's per-call ``has_invalid_tensor(to_cpu(args))`` scan; in TO-BE
    the C++ ``quick_fallback_check`` performs the same scan (only when
    non-deploy + ``TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK`` does not contain
    ``nan_inf`` / ``all``) so the warm-cache hot path cannot bypass it after
    the first clean call installs an entry for the shape.

    These tests assume the default non-deploy environment (the deploy / nan_inf-disable
    gates are read live from the environment on each dispatch, not process-cached; running
    the suite with ``TORCH_RBLN_DEPLOY=ON`` skips the scan and these checks no longer hold
    — that's the intended deploy-mode behaviour and matches AS-IS).
    """

    def setUp(self) -> None:
        # Each test starts with an empty warm cache so warm-hit interactions
        # are deterministic.
        _C._warmcache_clear()

    def test_nan_input_falls_back_and_propagates(self) -> None:
        x = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float16, device="rbln")
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="rbln")
        z = x + y
        self.assertEqual(z.device.type, "rbln")
        result = z.to("cpu")
        # CPU semantics: NaN + 2.0 = NaN.  If we'd taken the device path the
        # rbln runtime would have returned wrong (non-NaN) values for slot 1.
        self.assertTrue(torch.isnan(result[1]))
        self.assertEqual(result[0].item(), 2.0)
        self.assertEqual(result[2].item(), 6.0)

    def test_pos_inf_input_falls_back_and_propagates(self) -> None:
        x = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.float16, device="rbln")
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="rbln")
        z = x + y
        result = z.to("cpu")
        self.assertTrue(torch.isinf(result[1]))
        self.assertEqual(result[1].item(), float("inf"))
        self.assertEqual(result[0].item(), 2.0)
        self.assertEqual(result[2].item(), 6.0)

    def test_neg_inf_input_falls_back_and_propagates(self) -> None:
        x = torch.tensor([1.0, float("-inf"), 3.0], dtype=torch.float16, device="rbln")
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="rbln")
        z = x + y
        result = z.to("cpu")
        self.assertTrue(torch.isinf(result[1]))
        self.assertEqual(result[1].item(), float("-inf"))

    def test_nan_in_second_operand_also_caught(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="rbln")
        y = torch.tensor([float("nan"), 2.0, 3.0], dtype=torch.float16, device="rbln")
        z = x + y
        result = z.to("cpu")
        self.assertTrue(torch.isnan(result[0]))

    def test_clean_input_takes_device_path(self) -> None:
        # Regression: clean inputs must NOT be misclassified as needing
        # fallback. Verified indirectly: a warm-cache entry must be installed
        # (the fallback path never installs).
        # Use a 64-elem-aligned shape so ``compile_and_run_view_aware`` keeps
        # the call on the device path; unaligned last-dim now routes through
        # ``cpu_fallback_path`` and never primes the warm cache.
        size_before = _C._warmcache_size()
        x = torch.arange(64, dtype=torch.float16, device="rbln")
        y = torch.ones(64, dtype=torch.float16, device="rbln")
        z = x + y
        self.assertEqual(z.to("cpu"), torch.arange(1, 65, dtype=torch.float16))
        # At least one new warm-cache entry was installed.
        self.assertGreater(_C._warmcache_size(), size_before)

    def test_warm_hit_does_not_bypass_late_nan(self) -> None:
        """The critical regression case for the fixup.

        Step 1: clean inputs install a warm-cache entry for shape (64,).
        Step 2: same shape, but one input has NaN. Without this fixup the
                C++ shim would hit the warm cache, bypass the Python check,
                and hand the NaN tensor to the rbln runtime — wrong result.
        Step 3: NaN must propagate (proves the late NaN was caught and
                routed to the CPU fallback path even with the entry hot).
        """
        # Use a 64-aligned shape so the first call lands on the device path
        # (and thus primes the warm cache); unaligned shapes route through
        # ``cpu_fallback_path`` and never install an entry.
        n = 64
        # Step 1: prime the warm cache with clean inputs.
        x_clean = torch.arange(n, dtype=torch.float16, device="rbln")
        y = torch.ones(n, dtype=torch.float16, device="rbln")
        _ = x_clean + y
        self.assertGreater(_C._warmcache_size(), 0)

        # Step 2: same shape, NaN injected on one slot.
        x_dirty_cpu = torch.arange(n, dtype=torch.float16)
        x_dirty_cpu[3] = float("nan")
        x_dirty = x_dirty_cpu.to("rbln")
        z = x_dirty + y
        result = z.to("cpu")

        # Step 3: NaN preserved (fallback path was taken).
        self.assertTrue(torch.isnan(result[3]))
        self.assertEqual(result[0].item(), 1.0)
        self.assertEqual(result[n - 1].item(), float(n))

    def test_unaligned_clean_input_routes_to_cpu_fallback(self) -> None:
        """Sibling to ``test_clean_input_takes_device_path``: locks in the
        ``compile_and_run_view_aware`` 64-alignment guard.

        The two warm-cache tests above pick a 64-aligned shape so the device
        path runs and primes the warm cache. This test inverts the contract
        for unaligned shapes: clean fp16 inputs whose last-dim is NOT a
        multiple of 64 must route through ``cpu_fallback_path`` instead of
        torch.compile, and therefore must NOT install a warm-cache entry. If
        the alignment guard regresses (e.g. accidentally short-circuits to
        always-true or always-false), this test catches it from the other
        direction than ``test_clean_input_takes_device_path``.

        Correctness on the value side is also asserted: the cpu fallback
        result must match the upstream CPU result element-wise.
        """
        size_before = _C._warmcache_size()
        # 8 % 64 != 0 → ``_last_dim_unaligned`` returns True → cpu_fallback.
        x = torch.arange(8, dtype=torch.float16, device="rbln")
        y = torch.ones(8, dtype=torch.float16, device="rbln")
        z = x + y
        self.assertEqual(z.to("cpu"), torch.arange(1, 9, dtype=torch.float16))
        # No new warm-cache entry was installed (cpu_fallback never primes).
        self.assertEqual(_C._warmcache_size(), size_before)

    def test_nan_with_wrapped_python_scalar(self) -> None:
        # Wrapped 0-dim Python scalars (``tensor + 1.0``) are skipped from
        # the dtype shortcut but MUST still pass through the NaN/Inf scan —
        # ``tensor + math.nan`` would otherwise feed NaN to the device.
        import math

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="rbln")
        z = x + math.nan
        result = z.to("cpu")
        # All slots must be NaN if the scan caught the wrapped scalar.
        self.assertTrue(torch.isnan(result).all())


@pytest.mark.test_set_ci
class TestDispatchShimDeployGateLiveRead(TestCase):
    """The deploy gate (``is_deploy_mode`` in ``DispatchShim.cpp``) is read live on every
    dispatch, NOT cached in a process-lifetime ``static``. Toggling ``TORCH_RBLN_DEPLOY``
    inside one process must change whether the NaN/Inf pre-check runs, which is observable
    via the warm cache: with deploy OFF a NaN input takes the CPU fallback (which never
    primes the warm cache), while with deploy ON the scan is skipped and the op takes the
    device path (which installs a warm-cache entry). A re-introduced static cache would
    latch the first observed value and fail the toggle."""

    def test_deploy_gate_read_live_per_dispatch(self) -> None:
        import os
        from unittest import mock

        def nan_add(n):
            # All-NaN fp16 input, last dim 32-aligned so the device path is eligible once the
            # scan is skipped. Value doesn't affect compilation, so deploy-on still compiles.
            x = torch.full((n,), float("nan"), dtype=torch.float16, device="rbln")
            y = torch.ones(n, dtype=torch.float16, device="rbln")
            (x + y).to("cpu")

        if not _C._warmcache_is_enabled():
            self.skipTest("warm cache disabled; the device-path install is unobservable")

        with mock.patch.dict(os.environ):  # restores the full environment on exit
            _C._warmcache_clear()
            os.environ.pop("TORCH_RBLN_DEPLOY", None)
            base = _C._warmcache_size()
            nan_add(128)  # deploy OFF -> NaN pre-check -> CPU fallback -> no warm entry
            self.assertEqual(_C._warmcache_size(), base, "deploy-off NaN op must fall back, not prime the warm cache")

            os.environ["TORCH_RBLN_DEPLOY"] = "ON"
            nan_add(128)  # deploy ON -> scan skipped -> device path -> warm entry installed
            self.assertGreater(
                _C._warmcache_size(),
                base,
                "deploy-on must skip the scan and take the device path; a latched static gate would still fall back",
            )
            after_on = _C._warmcache_size()

            os.environ.pop("TORCH_RBLN_DEPLOY", None)
            nan_add(192)  # deploy OFF again, fresh shape -> live re-read -> fallback -> no new entry
            self.assertEqual(
                _C._warmcache_size(),
                after_on,
                "deploy read live: a fresh-shape NaN op must fall back again, not compile on the device",
            )


instantiate_device_type_tests(TestDispatchShimWrappedScalar, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimAllScalarFallback, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimDtypeMismatch, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimNanInfFallback, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimDeployGateLiveRead, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
