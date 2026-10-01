# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the C++ warm-cache internals (torch_rbln._C._warmcache_*).

The warm-cache keeps the compiled function of an (op, input-profile)
combination, with the positions of the call's tensors it takes, so that
subsequent dispatches with the same profile skip the Python wrapper and run the
function from C++. This module verifies the small public surface that the
dispatch shim and the generated Python wrappers depend on, and the install
rules:

  - enable / disable / size / clear  (state transitions)
  - an install takes only the call's own tensors, each at its position
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln
from torch_rbln import _C  # type: ignore[attr-defined]


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
class TestWarmCacheHitPath(TestCase):
    """Repeated dispatch of one input profile must keep taking the hit path.

    If no entry is installed the shim falls back to the Python wrapper: results
    stay correct, hits stop. Nothing else in the suite can see that, since
    correctness is unaffected.
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
@pytest.mark.single_worker
class TestWarmCacheInstall(TestCase):
    """What an install takes from the call it follows."""

    SHAPE = 192  # unused elsewhere, so the first call compiles

    def setUp(self) -> None:
        self._orig_install = _C._warmcache_install_pending
        self.addCleanup(setattr, _C, "_warmcache_install_pending", self._orig_install)
        self.addCleanup(_C._warmcache_clear)

    @staticmethod
    def _hits() -> int:
        return _C._dispatch_shim_warm_segments_dump()[0]

    def test_tensors_that_are_not_the_calls_install_nothing(self) -> None:
        """A wrapper that ran over copies of the call's tensors installs no entry:
        the hit path binds the call's tensors as they are."""

        def install_over_copies(function, inputs):
            return self._orig_install(function, [t.clone() for t in inputs])

        _C._warmcache_install_pending = install_over_copies
        x = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        y = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")
        expected = torch.arange(1, self.SHAPE + 1, dtype=torch.float16)

        self.assertEqual((x + y).to("cpu"), expected)
        hits_before = self._hits()
        for _ in range(4):
            self.assertEqual((x + y).to("cpu"), expected, "a refused install changed the result")
        self.assertEqual(self._hits(), hits_before, "an entry was cached over tensors of the wrapper's own")

    def test_a_tensor_passed_twice_does_not_bind_its_other_position(self) -> None:
        """``add(a, a)`` and ``add(a, b)`` share a key; each must add its own operands."""
        a = torch.arange(self.SHAPE, dtype=torch.float16, device="rbln")
        b = torch.ones(self.SHAPE, dtype=torch.float16, device="rbln")
        doubled = torch.arange(self.SHAPE, dtype=torch.float16) * 2
        plus_one = torch.arange(1, self.SHAPE + 1, dtype=torch.float16)

        self.assertEqual(torch.add(a, a).to("cpu"), doubled)
        self.assertEqual(torch.add(a, b).to("cpu"), plus_one)
        hits_before = self._hits()
        self.assertEqual(torch.add(a, b).to("cpu"), plus_one)
        self.assertEqual(torch.add(a, a).to("cpu"), doubled)
        self.assertGreater(self._hits(), hits_before, "the shared key never hit")


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestWarmCacheDeviceScope(TestCase):
    """``empty_cache(device)`` drops that device's entries and no other's.

    Serving several devices from one process, a flush on one of them must not
    put the others' ops back on the Python wrapper path.
    """

    SHAPE = 384  # unused elsewhere, so the first call compiles

    def setUp(self) -> None:
        if torch_rbln._C.device_count() < 2:
            self.skipTest("needs two RBLN devices")
        self.addCleanup(_C._warmcache_clear)

    @staticmethod
    def _hits() -> int:
        return _C._dispatch_shim_warm_segments_dump()[0]

    def _operands(self, device: str):
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

    def test_clearing_one_device_keeps_the_other_hitting(self) -> None:
        a = self._operands("rbln:0")
        self._assert_miss(*a, "rbln:0 first call took the hit path")
        self._assert_hit(*a, "rbln:0 second call did not hit")
        size_after_0 = _C._warmcache_size()
        b = self._operands("rbln:1")
        self._assert_miss(*b, "rbln:1 first call took the hit path")
        self._assert_hit(*b, "rbln:1 second call did not hit")
        self.assertGreater(_C._warmcache_size(), size_after_0)

        torch.rbln.empty_cache(1)

        self.assertEqual(_C._warmcache_size(), size_after_0, "empty_cache(1) did not drop exactly rbln:1's entries")
        self._assert_hit(*a, "rbln:0 stopped hitting after empty_cache(1)")
        self._assert_miss(*b, "rbln:1 hit after its entries were dropped")
        self._assert_hit(*b, "rbln:1 was not re-installed")


if __name__ == "__main__":
    run_tests()
