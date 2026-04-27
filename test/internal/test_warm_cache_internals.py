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

import pytest
import torch  # noqa: F401  (needed to load torch_rbln._C)

from torch_rbln import _C  # type: ignore[attr-defined]


@pytest.mark.test_set_ci
class TestWarmCacheEnableDisable:
    """`_warmcache_set_enabled` round-trips and `clear()` empties the cache.

    These functions are the primitive on which the generated wrappers in
    register_ops.py rely; if they regress, every shim op silently bypasses
    the cache and we lose the warm-path speedup.
    """

    def setup_method(self) -> None:
        self._was_enabled = _C._warmcache_is_enabled()

    def teardown_method(self) -> None:
        _C._warmcache_set_enabled(self._was_enabled)

    def test_default_state_is_queryable(self) -> None:
        # Whether enabled or not by default, the query must succeed.
        v = _C._warmcache_is_enabled()
        assert isinstance(v, bool)

    def test_set_enabled_round_trip(self) -> None:
        _C._warmcache_set_enabled(False)
        assert _C._warmcache_is_enabled() is False
        _C._warmcache_set_enabled(True)
        assert _C._warmcache_is_enabled() is True

    def test_size_is_non_negative_int(self) -> None:
        n = _C._warmcache_size()
        assert isinstance(n, int)
        assert n >= 0

    def test_clear_returns_size_zero(self) -> None:
        _C._warmcache_clear()
        assert _C._warmcache_size() == 0


@pytest.mark.test_set_ci
class TestWarmCacheBuildingGuard:
    """Reentrancy guard set by the miss path.

    During the torch.compile compilation triggered by a shim cache miss,
    nested ATen dispatches must take the slow path so that they do not try
    to hit a partially-built entry. The guard is thread-local; verify
    enter / exit pair, idempotency on double-exit, and isolation across
    threads (one thread's flag does not leak into another).
    """

    def teardown_method(self) -> None:
        # Always leave the flag cleared for subsequent tests.
        _C._warmcache_exit_building()

    def test_initial_state_is_not_building(self) -> None:
        _C._warmcache_exit_building()
        assert _C._warmcache_is_building() is False

    def test_enter_then_exit(self) -> None:
        _C._warmcache_enter_building()
        assert _C._warmcache_is_building() is True
        _C._warmcache_exit_building()
        assert _C._warmcache_is_building() is False

    def test_double_exit_is_no_op(self) -> None:
        _C._warmcache_exit_building()
        _C._warmcache_exit_building()
        assert _C._warmcache_is_building() is False

    def test_thread_local_isolation(self) -> None:
        """Setting the flag in one thread must not affect another.

        Without thread-local storage the flag would leak globally and the
        miss-path reentrancy guard would be unreliable in multi-worker
        scenarios.
        """
        _C._warmcache_enter_building()
        assert _C._warmcache_is_building() is True

        seen_in_thread: list[bool] = []
        ev = threading.Event()

        def worker() -> None:
            seen_in_thread.append(_C._warmcache_is_building())
            ev.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        ev.wait(timeout=5.0)
        t.join(timeout=5.0)

        assert seen_in_thread == [False], (
            f"Thread-local flag leaked across threads: {seen_in_thread}"
        )
        # Original thread still has the flag set.
        assert _C._warmcache_is_building() is True
