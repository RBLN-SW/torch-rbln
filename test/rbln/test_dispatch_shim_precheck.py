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


def _dispatch_counts() -> tuple[int, int, int, int]:
    """(n_total, n_fallback, n_warm_hit, n_miss) from the dispatch-shim diagnostics."""
    d = _C._dispatch_shim_diag_dump()
    return d[0], d[1], d[2], d[3]


@pytest.mark.test_set_ci
class TestDispatchShimWrappedScalar(TestCase):
    """A python-number operand is a wrapped 0-dim tensor at the dispatcher and a
    graph constant in the compiled program.

    Two things follow for the shim. The pre-check must not count it against the
    "all-scalar inputs" rule, or `add(tensor, 1.0)` would take the CPU fallback.
    And the warm cache must key it by *value* and keep it out of the runtime
    inputs: a hit that handed its host pointer to the runtime failed and erased
    the entry, so every python-scalar op ran the Python path forever, while a
    key that ignored the value would let `a + 2` return `a + 1`.
    """

    def setUp(self) -> None:
        _C._warmcache_clear()

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

    def test_python_scalar_warm_hits_and_keeps_entry(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                _C._warmcache_clear()
                a = torch.ones(1, 64, dtype=dtype, device="rbln")
                _ = a + 1  # compile + install
                self.assertEqual(_C._warmcache_size(), 1)
                before = _dispatch_counts()
                out = a + 1
                delta = tuple(b - c for c, b in zip(before, _dispatch_counts()))
                self.assertEqual(delta, (1, 0, 1, 0), "second `a + 1` must be a warm hit (total, fallback, hit, miss)")
                self.assertEqual(_C._warmcache_size(), 1, "the hit must not erase the entry")
                self.assertEqual(out.cpu(), torch.full((1, 64), 2.0, dtype=dtype))

    def test_distinct_python_scalars_are_distinct_entries(self) -> None:
        a = torch.ones(1, 64, dtype=torch.float16, device="rbln")
        r1 = (a + 1).cpu()
        r2 = (a + 2).cpu()
        self.assertEqual(_C._warmcache_size(), 2, "`a + 1` and `a + 2` must not share a key")
        before = _dispatch_counts()
        again1 = (a + 1).cpu()
        again2 = (a + 2).cpu()
        _, _, warm, miss = tuple(b - c for c, b in zip(before, _dispatch_counts()))
        self.assertEqual((warm, miss), (2, 0))
        self.assertEqual(again1, r1)
        self.assertEqual(again2, r2)
        self.assertEqual(r1[0, 0].item(), 2.0)
        self.assertEqual(r2[0, 0].item(), 3.0)

    def test_int_float_and_op_scalars_do_not_collide(self) -> None:
        a = torch.ones(1, 64, dtype=torch.float16, device="rbln")
        self.assertEqual((a + 1).cpu()[0, 0].item(), 2.0)
        self.assertEqual((a + 1.5).cpu()[0, 0].item(), 2.5)
        self.assertEqual((a * 3).cpu()[0, 0].item(), 3.0)
        self.assertEqual((a - 0.5).cpu()[0, 0].item(), 0.5)
        self.assertEqual(_C._warmcache_size(), 4)
        before = _dispatch_counts()
        self.assertEqual((a + 1.5).cpu()[0, 0].item(), 2.5)
        self.assertEqual((a * 3).cpu()[0, 0].item(), 3.0)
        _, _, warm, miss = tuple(b - c for c, b in zip(before, _dispatch_counts()))
        self.assertEqual((warm, miss), (2, 0))

    def test_inplace_python_scalar_add(self) -> None:
        a = torch.ones(1, 64, dtype=torch.float16, device="rbln")
        a += 1
        a += 1
        self.assertEqual(a.cpu(), torch.full((1, 64), 3.0, dtype=torch.float16))

    def test_zero_dim_device_tensor_stays_a_runtime_input(self) -> None:
        a = torch.ones(1, 64, dtype=torch.float16, device="rbln")
        s = torch.tensor(1.0, dtype=torch.float16, device="rbln")
        self.assertEqual((a + s).cpu()[0, 0].item(), 2.0)
        self.assertEqual((a + s).cpu()[0, 0].item(), 2.0)
        s.fill_(3.0)
        self.assertEqual((a + s).cpu()[0, 0].item(), 4.0, "a real 0-dim tensor is read at run time, not baked in")


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
class TestDispatchShimGateLiveRead(TestCase):
    """Both NaN/Inf-scan gates -- ``TORCH_RBLN_DEPLOY`` (``is_deploy_mode``) and
    ``TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK`` (``is_nan_inf_check_disabled``) in
    ``DispatchShim.cpp`` -- are read live per dispatch, not cached in a process-lifetime
    ``static``. A re-introduced static cache in either would latch the first value and fail
    its toggle. They are separate functions, so each is toggled separately."""

    _GATE_ENVS = ("TORCH_RBLN_DEPLOY", "TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK")

    def _nan_add(self, n):
        # All-NaN fp16 input with a 64-aligned last dim so the device path is eligible once the
        # scan is off (the NaN value doesn't affect compilation).
        x = torch.full((n,), float("nan"), dtype=torch.float16, device="rbln")
        y = torch.ones(n, dtype=torch.float16, device="rbln")
        (x + y).to("cpu")

    def _assert_gate_read_live(self, env_name, scan_off_value) -> None:
        """Toggle one gate off -> on -> off in a single process and assert the warm cache is
        primed only while the scan is off. Both gate envs are cleared at each baseline so
        ambient state can't skip the scan; ``env_name=scan_off_value`` disables it."""
        import os

        if not _C._warmcache_is_enabled():
            self.skipTest("warm cache disabled; the device-path install is unobservable")

        def scan_on_baseline():
            for env in self._GATE_ENVS:
                os.environ.pop(env, None)

        scan_on_baseline()
        _C._warmcache_clear()
        base = _C._warmcache_size()
        self._nan_add(128)
        self.assertEqual(_C._warmcache_size(), base, "scan on: NaN op must fall back, not prime the cache")

        os.environ[env_name] = scan_off_value
        self._nan_add(128)
        self.assertGreater(_C._warmcache_size(), base, f"{env_name} live: scan off must take the device path")
        after_off = _C._warmcache_size()

        scan_on_baseline()
        self._nan_add(192)  # fresh shape
        self.assertEqual(_C._warmcache_size(), after_off, f"{env_name} live: scan re-enabled must fall back again")

    def test_deploy_gate_read_live_per_dispatch(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ):  # restores the full environment on exit
            self._assert_gate_read_live("TORCH_RBLN_DEPLOY", "ON")

    def test_nan_inf_disable_gate_read_live_per_dispatch(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ):  # restores the full environment on exit
            self._assert_gate_read_live("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "nan_inf")


instantiate_device_type_tests(TestDispatchShimWrappedScalar, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimAllScalarFallback, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimDtypeMismatch, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimNanInfFallback, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDispatchShimGateLiveRead, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
