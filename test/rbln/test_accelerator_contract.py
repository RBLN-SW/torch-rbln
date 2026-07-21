# Owner(s): ["module: PrivateUse1"]

"""Regression tests for torch.accelerator / torch.rbln contract gaps.

Covers:
- ``torch.rbln.memory_*`` reading the RBLN allocator's dotted stat keys
  (``allocated.current`` etc.), not the CUDA-style ``allocated_bytes.all.current``.
- Device-argument normalization (accept None/int/str/torch.device, reject
  non-rbln devices and out-of-range indices).
- ``memory_stats()`` returning zero (not raising) for a valid but uninitialized
  device index (CUDA parity).
- ``torch.accelerator.get_device_capability()`` advertising the dtypes RBLN can
  allocate and type-convert on device (allocation + conversion, not native-op
  dispatch).
- PrivateUse1 storage resize-to-zero.
- ``torch.accelerator.empty_cache()`` (no device arg) actually releasing cached
  memory on the current device.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 -- registers the rbln device + torch.rbln namespace


@pytest.mark.single_worker
class TestRblnMemoryHelpers(TestCase):
    """``torch.rbln.memory_*`` must read the RBLN allocator's dotted stat keys.

    Single-worker: mutates the global allocator (empty_cache / peak reset) and
    compares stats across calls, so it must not race other workers' allocations."""

    @pytest.mark.test_set_ci
    def test_memory_helpers_match_stats_and_are_nonzero(self):
        # Pin to device 0 explicitly. The helpers/reset default to the *current*
        # device, which is not guaranteed to be 0 (another test may have changed it),
        # so allocating on rbln:0 but querying no-arg would be flaky.
        torch.rbln.empty_cache(0)
        torch.rbln.reset_peak_memory_stats(0)

        keep = torch.randn(1024, 1024, device="rbln:0", dtype=torch.float16)
        _ = keep + keep

        stats = torch.rbln.memory_stats(0)
        self.assertEqual(torch.rbln.memory_allocated(0), stats.get("allocated.current", 0))
        self.assertEqual(torch.rbln.max_memory_allocated(0), stats.get("allocated.peak", 0))
        self.assertEqual(torch.rbln.memory_reserved(0), stats.get("reserved.current", 0))
        self.assertEqual(torch.rbln.max_memory_reserved(0), stats.get("reserved.peak", 0))
        # Regression: the CUDA-style key made these return 0 despite a live allocation.
        self.assertGreater(torch.rbln.memory_allocated(0), 0)
        del keep

    @pytest.mark.test_set_ci
    def test_memory_stats_rejects_non_rbln_devices(self):
        for bad in ("cpu", "cuda:0", torch.device("cpu")):
            with self.assertRaises(ValueError):
                torch.rbln.memory_stats(bad)

    @pytest.mark.test_set_ci
    def test_memory_stats_accepts_all_device_forms(self):
        current = torch.accelerator.current_device_index()
        for dev in (None, 0, "rbln:0", "rbln", torch.device("rbln", 0), torch.device("rbln", current)):
            self.assertIsInstance(torch.rbln.memory_stats(dev), dict)

    @pytest.mark.test_set_ci
    def test_memory_stats_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            torch.rbln.memory_stats(torch.rbln.device_count())  # first out-of-range index

    @pytest.mark.test_set_ci
    def test_memory_stats_rejects_int8_wrapped_index(self):
        # torch.device's index is an int8_t: raw 256 wraps to 0, 257 to 1, 255 to -1
        # (current device). Such values must be rejected from the original int/str, not
        # normalized to an in-range device by the silent wrap.
        count = torch.rbln.device_count()
        wrapped = [255, 256, 257, "rbln:255", "rbln:256", 256 + max(count - 1, 0)]
        for bad in wrapped:
            with self.assertRaises(ValueError, msg=f"{bad!r} slipped past the range check"):
                torch.rbln.memory_stats(bad)

    @pytest.mark.test_set_ci
    def test_memory_stats_uninitialized_device_reports_zero_not_raise(self):
        # CUDA parity: memory_stats() must not throw for a *valid* device index this
        # process has not allocated on. It reports zero (like torch.cuda.memory_stats
        # on an uninitialized device) via the device_context_initialized gate, instead
        # of hitting the runtime, whose per-node stats query rejects such a device
        # (INIT_INVALID_ARGUMENT). (An *initialized* device index > 0 is a separate,
        # runtime-limited case that still surfaces its error -- the runtime supports
        # per-node stats for node 0 only -- so it is intentionally not asserted here.)
        count = torch.rbln.device_count()
        if count < 2:
            self.skipTest("needs a second, never-allocated device index")
        # Highest index: not touched by other tests in this (single) worker process.
        idx = count - 1
        self.assertIsInstance(torch.rbln.memory_stats(idx), dict)  # must not raise
        self.assertEqual(torch.rbln.memory_allocated(idx), 0)
        self.assertEqual(torch.rbln.memory_reserved(idx), 0)

    @pytest.mark.test_set_ci
    def test_accelerator_memory_stats_uninitialized_device_reports_zero_not_raise(self):
        # The generic torch.accelerator path routes to the C10 getDeviceStats() hook --
        # a separate code path from torch.rbln.memory_stats(). It must report zero (not
        # raise) for a valid, uninitialized device index.
        count = torch.rbln.device_count()
        if count < 2:
            self.skipTest("needs a second, never-allocated device index")
        # Initialize the allocator with a live device-0 allocation first: otherwise
        # torch.accelerator short-circuits to an empty dict *before* reaching
        # getDeviceStats(), making the check below vacuous. With it initialized, querying
        # an uninitialized index actually exercises getDeviceStats() (populated zeros).
        keep = torch.randn(1024, 1024, device="rbln:0", dtype=torch.float16)
        _ = keep + keep
        idx = count - 1  # highest index: not touched by other tests in this worker
        stats = torch.accelerator.memory_stats(idx)  # must not raise (regression: INIT_INVALID_ARGUMENT)
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0, "getDeviceStats() was not exercised (accelerator returned an empty dict)")
        self.assertTrue(
            all(v == 0 for v in stats.values() if isinstance(v, int)),
            msg=f"expected zero stats for uninitialized rbln:{idx}, got nonzero entries",
        )
        del keep


@pytest.mark.single_worker
class TestDeviceCapability(TestCase):
    """``get_device_capability()`` reports dtypes resident in device memory
    (fp16/bf16); other dtypes are CPU-backed even under device="rbln".
    Single-worker: measures global device memory."""

    @pytest.mark.test_set_ci
    def test_capability_matches_device_resident_dtypes(self):
        # Probe a candidate set spanning both advertised (fp16/bf16) and unadvertised
        # (fp32/int) dtypes, classify each by whether a compute op materializes it in
        # device memory, and require the advertised set to equal the resident set — so a
        # dtype that is resident but missing from the advertisement is caught too, not
        # just the reverse.
        advertised = set(torch.accelerator.get_device_capability()["supported_dtypes"])
        candidates = [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64]
        resident = set()
        for dtype in candidates:
            torch.rbln.empty_cache()
            before = torch.rbln.memory_allocated(0)
            scratch = torch.empty(1024 * 1024, dtype=dtype, device="rbln:0")
            scratch.add_(1)
            if torch.rbln.memory_allocated(0) - before > 0:
                resident.add(dtype)
            del scratch
        self.assertEqual(
            advertised,
            resident,
            msg=f"advertised dtypes {advertised} disagree with device-resident dtypes {resident}",
        )


class TestStorageResizeToZero(TestCase):
    """PrivateUse1 storage must accept ``resize_(0)``."""

    @pytest.mark.test_set_ci
    def test_resize_storage_to_zero(self):
        x = torch.empty(16, device="rbln:0", dtype=torch.float16)
        storage = x.untyped_storage()
        storage.resize_(0)
        self.assertEqual(storage.nbytes(), 0)


@pytest.mark.single_worker
class TestAcceleratorEmptyCache(TestCase):
    """Device-less ``torch.accelerator.empty_cache()`` actually releases cached
    memory (it is not a no-op or a query-only stub).

    Single-worker: asserts on global reserved-memory changes, so it must not race
    other workers' allocations.

    Note: the all-devices span — ``empty_cache()`` walking *every* initialized
    device, not just the current one — is not asserted at the Python level.
    Confirming a non-current device was released means reading its reserved bytes,
    but the rbln runtime's per-node memory-stats query supports node 0 only, so on a
    device index > 0 ``memory_reserved()`` either raises (an initialized device, the
    runtime rejects the query) or reads 0 (an uninitialized device, the gate) — never
    the real figure. The all-device selection is instead covered in C++ by
    ``RBLNAllocatorTest.EmptyCacheSpansNonCurrentInitializedDevice`` (emptyCache()
    iterates ``initialized_device_indices()``, asserted to include a non-current
    device)."""

    @staticmethod
    def _hold_then_free_reserved(device):
        # Allocate then free a large buffer. The caching allocator keeps the
        # freed block as *reserved* (freeing alone does not return it to the
        # runtime), so reserved stays high until empty_cache() releases it.
        buf = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.float16)  # 32 MiB
        buf.add_(1.0)
        del buf
        return torch.rbln.memory_reserved(device)

    @pytest.mark.test_set_ci
    def test_empty_cache_releases_current_device(self):
        reserved_cached = self._hold_then_free_reserved("rbln:0")
        torch.accelerator.empty_cache()
        # Must actually release the cached block; a no-op or query-only
        # implementation would leave reserved unchanged.
        self.assertLess(torch.rbln.memory_reserved("rbln:0"), reserved_cached)


if __name__ == "__main__":
    run_tests()
