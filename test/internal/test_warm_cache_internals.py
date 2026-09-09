# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the C++ warm-cache internals (torch_rbln._C._warmcache_*).

The warm-cache caches the rebel-runtime handle for a (op, input-profile)
combination so that subsequent dispatches with the same profile bypass the
torch.compile / pybind round-trip and drive the runtime directly from C++.
This module verifies the small public surface that the dispatch shim and the
generated Python wrappers depend on:

  - enable / disable / size / clear  (state transitions)
  - thread-local "building" reentrancy guard
"""

import threading
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln
from torch_rbln import _C  # type: ignore[attr-defined]
from torch_rbln._internal import warm_cache


# These suites exercise the C++ warm-cache's process-wide / thread-local
# state via its pybind surface. No RBLN-device tensor work happens here, so
# ``instantiate_device_type_tests`` is intentionally NOT used — this matches
# the precedent set by ``test/rbln/test_file_offloading.py`` for tests that
# probe process-level flags rather than device-side ops.


@pytest.mark.test_set_ci
class TestWarmCacheEnableDisable(TestCase):
    """`_warmcache_set_enabled` round-trips and `clear()` empties the cache.

    These functions are the primitive on which the generated wrappers in
    register_ops.py rely; if they regress, every shim op silently bypasses
    the cache and we lose the warm-path speedup.
    """

    def setUp(self) -> None:
        self._was_enabled = _C._warmcache_is_enabled()

    def tearDown(self) -> None:
        _C._warmcache_set_enabled(self._was_enabled)

    def test_default_state_is_queryable(self) -> None:
        # Whether enabled or not by default, the query must succeed.
        v = _C._warmcache_is_enabled()
        self.assertIsInstance(v, bool)

    def test_set_enabled_round_trip(self) -> None:
        _C._warmcache_set_enabled(False)
        self.assertFalse(_C._warmcache_is_enabled())
        _C._warmcache_set_enabled(True)
        self.assertTrue(_C._warmcache_is_enabled())

    def test_size_is_non_negative_int(self) -> None:
        n = _C._warmcache_size()
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)

    def test_clear_returns_size_zero(self) -> None:
        _C._warmcache_clear()
        self.assertEqual(_C._warmcache_size(), 0)


@pytest.mark.test_set_ci
class TestWarmCacheBuildingGuard(TestCase):
    """Reentrancy guard set by the miss path.

    During the torch.compile compilation triggered by a shim cache miss,
    nested ATen dispatches must take the slow path so that they do not try
    to hit a partially-built entry. The guard is thread-local; verify
    enter / exit pair, idempotency on double-exit, and isolation across
    threads (one thread's flag does not leak into another).
    """

    def tearDown(self) -> None:
        # Always leave the flag cleared for subsequent tests.
        _C._warmcache_exit_building()

    def test_initial_state_is_not_building(self) -> None:
        _C._warmcache_exit_building()
        self.assertFalse(_C._warmcache_is_building())

    def test_enter_then_exit(self) -> None:
        _C._warmcache_enter_building()
        self.assertTrue(_C._warmcache_is_building())
        _C._warmcache_exit_building()
        self.assertFalse(_C._warmcache_is_building())

    def test_double_exit_is_no_op(self) -> None:
        _C._warmcache_exit_building()
        _C._warmcache_exit_building()
        self.assertFalse(_C._warmcache_is_building())

    def test_thread_local_isolation(self) -> None:
        """Setting the flag in one thread must not affect another.

        Without thread-local storage the flag would leak globally and the
        miss-path reentrancy guard would be unreliable in multi-worker
        scenarios.
        """
        _C._warmcache_enter_building()
        self.assertTrue(_C._warmcache_is_building())

        seen_in_thread: list[bool] = []
        ev = threading.Event()

        def worker() -> None:
            seen_in_thread.append(_C._warmcache_is_building())
            ev.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ev.wait(timeout=5.0)
        t.join(timeout=5.0)

        self.assertEqual(seen_in_thread, [False], f"Thread-local flag leaked across threads: {seen_in_thread}")
        # Original thread still has the flag set.
        self.assertTrue(_C._warmcache_is_building())


@pytest.mark.test_set_ci
class TestWarmCacheHitPath(TestCase):
    """Repeated dispatch of one input profile must keep taking the hit path.

    The hit path drives the rebel runtime through ``rbln_exec_api.h``, reached
    via the handle's ``native_handle()`` (see ``WarmCache.h``). If rebel drops
    either, no entry is installed and the shim falls back to the Python
    wrapper: results stay correct, hits stop. Nothing else in the suite can see
    that, since correctness is unaffected.
    """

    def test_repeated_same_profile_dispatch_takes_hit_path(self) -> None:
        x = torch.arange(64, dtype=torch.float16, device="rbln")
        y = torch.ones(64, dtype=torch.float16, device="rbln")
        expected = torch.arange(1, 65, dtype=torch.float16)

        # The first call installs the entry; the ones after it must hit. Read
        # the counter as a delta rather than resetting it, so the test leaves
        # no process-global state behind.
        self.assertEqual((x + y).to("cpu"), expected)
        hits_before = _C._dispatch_shim_warm_segments_dump()[0]
        for _ in range(3):
            self.assertEqual((x + y).to("cpu"), expected)

        hits_after = _C._dispatch_shim_warm_segments_dump()[0]
        self.assertGreater(hits_after, hits_before, "warm-cache hit path never ran")


@pytest.mark.test_set_ci
class TestWarmCacheHandleGate(TestCase):
    """Which runtime handles the hit path accepts.

    ``native_handle()`` is the whole contract on the Python side, so a handle
    is usable exactly when it carries one. The gate runs before an entry is
    cached, which is what keeps a rebel-side rename from reaching the C++ side
    at all.
    """

    class _Runtime:
        def native_handle(self):
            return 0xDEADBEEF

    def test_handle_with_native_handle_is_accepted(self) -> None:
        self.assertTrue(warm_cache._is_drivable_runtime_handle(self._Runtime()))

    def test_handle_without_callable_native_handle_is_refused(self) -> None:
        handle = self._Runtime()
        handle.native_handle = None  # shadows the class method
        self.assertFalse(warm_cache._is_drivable_runtime_handle(handle))

    def test_absent_handle_is_refused(self) -> None:
        self.assertFalse(warm_cache._is_drivable_runtime_handle(None))
        self.assertFalse(warm_cache._is_drivable_runtime_handle(object()))


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestWarmCacheContractBreak(TestCase):
    """A runtime the hit path cannot take a handle off must stay harmless.

    A runtime whose ``native_handle`` no longer answers has to cost only the
    fast path: correct results, on the Python wrapper path, for as long as it
    takes torch-rbln to catch up. That guarantee is what lets rebel change the
    surface without waiting for a matching torch-rbln, so assert it rather than
    assume it.
    """

    SHAPE = 192  # 64-aligned and unused elsewhere, so the first call compiles

    def setUp(self) -> None:
        self._orig_install = _C._warmcache_install_pending
        self.addCleanup(setattr, _C, "_warmcache_install_pending", self._orig_install)
        self.addCleanup(_C._warmcache_clear)

    def test_unusable_handle_keeps_values_and_stops_hitting(self) -> None:
        class RejectingHandle:
            """Delegates everything except the one call install makes."""

            def __init__(self, inner):
                self._inner = inner

            def native_handle(self):
                raise TypeError("native_handle(): incompatible function arguments")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        def install_with_rejecting_handle(*, dyn_runtime, runtime_handle, **kw):
            return self._orig_install(dyn_runtime=dyn_runtime, runtime_handle=RejectingHandle(runtime_handle), **kw)

        _C._warmcache_install_pending = install_with_rejecting_handle

        x = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        y = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")
        expected = torch.arange(1, self.SHAPE + 1, dtype=torch.float16)

        # The first call would have installed; the rest would have hit it.
        self.assertEqual((x + y).to("cpu"), expected)
        hits_before = _C._dispatch_shim_warm_segments_dump()[0]
        for _ in range(4):
            self.assertEqual((x + y).to("cpu"), expected, "a refused install changed the result")

        hits_after = _C._dispatch_shim_warm_segments_dump()[0]
        self.assertEqual(hits_after, hits_before, "an entry was cached off an unusable handle")


class _HitPathCase(TestCase):
    """Shared plumbing: a fresh 64-aligned profile per suite, hit-counter reads, miss/hit assertions."""

    SHAPE = 0  # 64-aligned and unused elsewhere, so the first call compiles

    def setUp(self) -> None:
        self.addCleanup(_C._warmcache_clear)

    @staticmethod
    def _hits() -> int:
        return _C._dispatch_shim_warm_segments_dump()[0]

    def _operands(self, device: str = "rbln"):
        x = torch.arange(self.SHAPE, dtype=torch.float16, device=device)
        y = torch.ones(self.SHAPE, dtype=torch.float16, device=device)
        expected = torch.arange(1, self.SHAPE + 1, dtype=torch.float16)
        return x, y, expected

    def _assert_miss(self, x, y, expected, msg: str) -> None:
        hits = self._hits()
        self.assertEqual(torch.add(x, y).to("cpu"), expected)
        self.assertEqual(self._hits(), hits, msg)

    def _assert_hit(self, x, y, expected, msg: str) -> None:
        hits = self._hits()
        self.assertEqual(torch.add(x, y).to("cpu"), expected)
        self.assertGreater(self._hits(), hits, msg)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestWarmCacheReinstall(_HitPathCase):
    """A profile whose entry was dropped must get one back on its next miss, from
    the runtime it already has.

    ``empty_cache`` drops the C++ entries; the Python compile cache keeps the
    callable, and Dynamo keeps its graph and runtime. The re-install has to come
    from that runtime. A re-install that compiled instead would leave the
    previous graph and runtime in Dynamo's cache on every cycle, until the
    recompile limit resets every compiled graph in the process.
    """

    SHAPE = 320
    CYCLES = 8

    def test_empty_cache_then_same_profile_reinstalls_the_same_runtime(self) -> None:
        x, y, expected = self._operands()
        self._assert_miss(x, y, expected, "first call of a fresh profile took the hit path")
        self._assert_hit(x, y, expected, "second call did not take the hit path")

        installed: list[int] = []
        orig_install = _C._warmcache_install_pending

        def recording_install(*, dyn_runtime, **kw):
            installed.append(id(dyn_runtime))
            return orig_install(dyn_runtime=dyn_runtime, **kw)

        compiles = 0
        orig_compile = warm_cache._compile_graph

        def counting_compile(*args):
            nonlocal compiles
            compiles += 1
            return orig_compile(*args)

        with (
            mock.patch.object(_C, "_warmcache_install_pending", recording_install),
            mock.patch.object(warm_cache, "_compile_graph", counting_compile),
        ):
            for cycle in range(self.CYCLES):
                size = _C._warmcache_size()
                torch.rbln.empty_cache()
                self.assertLess(_C._warmcache_size(), size, "empty_cache() left the entry in place")
                self._assert_miss(x, y, expected, f"cycle {cycle}: call after empty_cache() took the hit path")
                self._assert_hit(x, y, expected, f"cycle {cycle}: entry was not re-installed")

        self.assertEqual(len(installed), self.CYCLES, "expected one install per cycle")
        self.assertEqual(len(set(installed)), 1, "a re-install used a different runtime than the first install")
        self.assertEqual(compiles, 0, "a re-install ran the backend again")


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestWarmCacheDeviceScope(_HitPathCase):
    """``empty_cache(device)`` drops that device's entries and no other's.

    Serving several devices from one process, a flush on one of them must not
    put the others' ops back on the Python wrapper path.
    """

    SHAPE = 384

    def setUp(self) -> None:
        if torch_rbln._C.device_count() < 2:
            self.skipTest("needs two RBLN devices")
        super().setUp()

    def test_clearing_one_device_keeps_the_other_hitting(self) -> None:
        a = self._operands("rbln:0")
        self._assert_miss(*a, "rbln:0 first call took the hit path")
        self._assert_hit(*a, "rbln:0 second call did not hit")
        size_after_0 = _C._warmcache_size()
        b = self._operands("rbln:1")
        self._assert_miss(*b, "rbln:1 first call took the hit path")
        self._assert_hit(*b, "rbln:1 second call did not hit")
        size_after_1 = _C._warmcache_size()
        self.assertGreater(size_after_1, size_after_0)

        torch.rbln.empty_cache(1)

        self.assertEqual(_C._warmcache_size(), size_after_0, "empty_cache(1) did not drop exactly rbln:1's entries")
        self._assert_hit(*a, "rbln:0 stopped hitting after empty_cache(1)")
        self._assert_miss(*b, "rbln:1 hit after its entries were dropped")
        self._assert_hit(*b, "rbln:1 was not re-installed")


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestWarmCacheRetire(_HitPathCase):
    """A key whose hit failed at run time stays on the Python wrapper path.

    The runtime that failed is the one a re-install would use again, so the
    C++ side replaces the entry with a tombstone: values stay correct, the hit
    counter stops for that key, the next miss does not install, and
    ``empty_cache`` gives the key a fresh start.
    """

    SHAPE = 448

    def test_failed_hit_retires_the_key_until_cleared(self) -> None:
        x, y, expected = self._operands()
        self._assert_miss(x, y, expected, "first call took the hit path")
        self._assert_hit(x, y, expected, "second call did not hit")
        size = _C._warmcache_size()

        _C._warmcache_inject_hit_failure()
        self._assert_miss(x, y, expected, "a failed hit attempt was counted as a hit")
        self.assertEqual(_C._warmcache_size(), size, "the retired key was not kept as a tombstone")
        for i in range(3):
            self._assert_miss(x, y, expected, f"call {i} after the failure was re-installed and hit")

        torch.rbln.empty_cache()
        self._assert_miss(x, y, expected, "call after empty_cache() took the hit path")
        self._assert_hit(x, y, expected, "the key did not get a fresh start after empty_cache()")


if __name__ == "__main__":
    run_tests()
