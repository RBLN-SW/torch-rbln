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
  - the hit path still runs, and still degrades safely when it cannot
"""

import threading
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln import _C  # type: ignore[attr-defined]
from torch_rbln._internal import rebel_contract, warm_cache


# Most suites here exercise the C++ warm-cache's process-wide / thread-local
# state via its pybind surface. Those do no RBLN-device tensor work, so
# ``instantiate_device_type_tests`` is intentionally NOT used — this matches
# the precedent set by ``test/rbln/test_file_offloading.py`` for tests that
# probe process-level flags rather than device-side ops.
# ``TestWarmCacheHitPath`` and ``TestWarmCacheContractBreak`` are the
# exceptions: they need a real dispatch, and follow the device-test shape used
# by ``test_memory_stats.py``.
#
# Both belong here rather than under ``test/rbln/`` because of who runs what:
# rebel-compiler's CI runs ``test/rbln`` against a pinned torch-rbln, and
# rebel lands its interface changes before torch-rbln follows. A hit-path
# assertion placed there would block a rebel change that costs us only this
# optimization. Here, the signal lands on the side that has to follow.


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
class TestWarmCacheForceRecompileFlag(TestCase):
    """Thread-local force-recompile signal that pairs ``try_warmcache_hit``'s
    erase-on-failure with the next ``compile_rbln_cached`` invocation.

    The C++ shim sets the flag inside ``try_warmcache_hit`` when it has to
    ``erase`` a broken entry; ``compile_and_run_view_aware`` consumes the
    flag and passes ``force_recompile=True`` to ``compile_rbln_cached`` so
    the same op+shape gets a fresh ``torch.compile`` pass (which lets the
    rebel backend re-populate ``_runtime_holder`` and install succeed
    again). Without this pairing, the erased op+shape would stay
    permanently on the Python wrapper path because the Python compile
    cache returns the stale callable and the holder stays empty.
    """

    def setUp(self) -> None:
        _C._warmcache_consume_force_recompile()

    def tearDown(self) -> None:
        _C._warmcache_consume_force_recompile()

    def test_default_false(self) -> None:
        self.assertFalse(_C._warmcache_consume_force_recompile())

    def test_request_then_consume_once(self) -> None:
        _C._warmcache_request_force_recompile()
        self.assertTrue(_C._warmcache_consume_force_recompile())
        # Consume is single-shot.
        self.assertFalse(_C._warmcache_consume_force_recompile())

    def test_thread_local_isolation(self) -> None:
        """The flag must not leak across threads. If it did, an erase on
        thread A would force unnecessary recompiles on thread B's next
        unrelated op."""
        _C._warmcache_request_force_recompile()
        seen_in_thread: list[bool] = []
        ev = threading.Event()

        def worker() -> None:
            seen_in_thread.append(_C._warmcache_consume_force_recompile())
            ev.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ev.wait(timeout=5.0)
        t.join(timeout=5.0)

        self.assertEqual(seen_in_thread, [False], f"force-recompile flag leaked across threads: {seen_in_thread}")
        self.assertTrue(_C._warmcache_consume_force_recompile())


@pytest.mark.test_set_ci
class TestWarmCacheHitPath(TestCase):
    """Repeated dispatch of one input profile must keep taking the hit path.

    The hit path calls the rebel runtime's ``prepare_inputs`` /
    ``prepare_outputs`` / ``run`` by name (see ``WarmCache.h``). If rebel
    renames one or makes a parameter required, every call raises and the shim
    falls back to the Python wrapper: results stay correct, hits stop. Nothing
    else in the suite can see that, since correctness is unaffected.
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

    The hit path calls the methods ``rebel_contract`` declares, so a handle is usable exactly
    when it carries all of them. The gate runs before an entry is cached, which is what keeps a
    rebel-side rename from reaching the C++ side at all.
    """

    class _Runtime:
        def prepare_inputs(self, *a, **k): ...
        def prepare_outputs(self, *a, **k): ...
        def run(self, *a, **k): ...

    def test_handle_with_every_declared_method_is_accepted(self) -> None:
        methods = rebel_contract.runtime_methods(self._Runtime())
        self.assertIsNotNone(methods)
        self.assertEqual(len(methods), len(rebel_contract.RUNTIME_METHODS))

    def test_handle_missing_any_method_is_refused(self) -> None:
        for name in rebel_contract.RUNTIME_METHODS:
            handle = self._Runtime()
            setattr(handle, name, None)  # shadows the class method
            self.assertIsNone(
                rebel_contract.runtime_methods(handle),
                f"a handle whose {name} is not callable must be refused",
            )

    def test_absent_handle_is_refused(self) -> None:
        self.assertIsNone(rebel_contract.runtime_methods(None))
        self.assertIsNone(rebel_contract.runtime_methods(object()))


@pytest.mark.test_set_ci
class TestWarmCacheContractBreak(TestCase):
    """A runtime that rejects the hit path's call must cost only the fast path.

    A runtime whose ``prepare_inputs`` no longer accepts this call shape has to stay harmless:
    correct results, on the Python wrapper path, for as long as it takes torch-rbln to catch up.
    That guarantee is what lets rebel merge without waiting for a matching torch-rbln, so assert
    it rather than assume it.

    ``TypeError`` is what pybind raises when the argument count no longer matches, and it is the
    signal the shim reads as "this build's call no longer fits", so the rejection uses it.
    """

    SHAPE = 192  # 64-aligned and unused elsewhere, so the first call compiles

    def setUp(self) -> None:
        self._orig_install = _C._warmcache_install_pending
        self.addCleanup(setattr, _C, "_warmcache_install_pending", self._orig_install)
        self.addCleanup(_C._warmcache_clear)
        # A break disables the cache for the process; later tests need it back.
        self.addCleanup(_C._warmcache_set_enabled, True)
        self.addCleanup(_C._warmcache_take_contract_break)

    def _install_a_rejecting_prepare_inputs(self) -> None:
        def rejecting(*a, **k):
            raise TypeError("prepare_inputs(): incompatible function arguments")

        def install(*, dyn_runtime, prepare_inputs, **kw):
            return self._orig_install(dyn_runtime=dyn_runtime, prepare_inputs=rejecting, **kw)

        _C._warmcache_install_pending = install

    def test_rejected_call_keeps_values_and_stops_hitting(self) -> None:
        self._install_a_rejecting_prepare_inputs()

        x = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        y = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")
        expected = torch.arange(1, self.SHAPE + 1, dtype=torch.float16)

        # First call installs the rejecting method; the rest would hit it.
        self.assertEqual((x + y).to("cpu"), expected)
        hits_before = _C._dispatch_shim_warm_segments_dump()[0]
        for _ in range(4):
            self.assertEqual((x + y).to("cpu"), expected, "a rejected hit path changed the result")

        hits_after = _C._dispatch_shim_warm_segments_dump()[0]
        self.assertEqual(hits_after, hits_before, "a rejected call was counted as a hit")

    def test_rejected_call_stops_the_fast_path_instead_of_retrying(self) -> None:
        # Without this, every dispatch erases the entry and forces a recompile that reinstalls
        # the same rejected call -- correct results at the cost of a compile per dispatch.
        self._install_a_rejecting_prepare_inputs()

        x = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        y = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")

        self.assertTrue(_C._warmcache_is_enabled())
        for _ in range(3):
            (x + y).to("cpu")
        self.assertFalse(_C._warmcache_is_enabled(), "a rejected call left the fast path armed")

    def test_the_break_reaches_the_user_through_the_dispatch_path(self) -> None:
        # The whole handoff: the C++ hit path flags the break, install_pending consumes the flag
        # on the miss that follows, and the report comes out of Python -- which is where the
        # contract declaration is, and so the only side that can name what changed.
        self._install_a_rejecting_prepare_inputs()

        x = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        y = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")

        (x + y).to("cpu")  # installs the rejecting method
        with self.assertWarns(RuntimeWarning) as caught:
            for _ in range(3):
                (x + y).to("cpu")
        self.assertIn("warm cache is off", str(caught.warning))

    def test_the_report_names_the_divergence(self) -> None:
        divergence = rebel_contract.Divergence("rebel.made_up:entry", rebel_contract.BROKEN, "not there")
        with mock.patch.object(rebel_contract, "verify", return_value=[divergence]):
            with self.assertWarns(RuntimeWarning) as caught:
                warm_cache._warn_contract_break()
        self.assertIn("rebel.made_up:entry", str(caught.warning))


if __name__ == "__main__":
    run_tests()
