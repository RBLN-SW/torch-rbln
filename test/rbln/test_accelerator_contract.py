# Owner(s): ["module: PrivateUse1"]

"""Regression tests for torch.accelerator / torch.rbln contract gaps.

Covers:
- ``torch.rbln.memory_*`` reading the ``torch.cuda``-style stat keys
  (``allocated_bytes.all.current`` etc.) that ``torch.rbln.memory_stats()`` reports.
- Device-argument normalization (accept None/int/str/torch.device, reject
  non-rbln devices and out-of-range indices).
- ``memory_stats()`` returning zero (not raising) for a valid but uninitialized
  device index (CUDA parity).
- ``torch.accelerator.get_device_capability()`` advertising the dtypes the device
  computes on, while a tensor of any dtype takes device memory when it is created.
- PrivateUse1 storage resize-to-zero.
- ``torch.accelerator.empty_cache()`` (no device arg) actually releasing cached
  memory on every device this process has allocated on.
- ``torch.accelerator.empty_host_cache()`` returning once the device is in use.
- ``torch.rbln.mem_get_info()`` / ``torch.accelerator.get_memory_info()`` reporting
  the driver's device-wide figures, and raising -- not guessing -- where there is no NPU
  to ask.
"""

import os
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 -- registers the rbln device + torch.rbln namespace
from test.utils import requires_logical_devices, run_in_isolated_process, SUPPORTED_DTYPES


@pytest.mark.single_worker
class TestRblnMemoryHelpers(TestCase):
    """``torch.rbln.memory_*`` must read the stat keys ``torch.rbln.memory_stats()`` reports.

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
        self.assertEqual(torch.rbln.memory_allocated(0), stats["allocated_bytes.all.current"])
        self.assertEqual(torch.rbln.max_memory_allocated(0), stats["allocated_bytes.all.peak"])
        self.assertEqual(torch.rbln.memory_reserved(0), stats["reserved_bytes.all.current"])
        self.assertEqual(torch.rbln.max_memory_reserved(0), stats["reserved_bytes.all.peak"])
        # A helper reading a key the stats lack would return 0 despite the live allocation.
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
    def test_memory_stats_rejects_malformed_device_strings(self):
        # torch's canonical device grammar rejects these; a bare partition(":") + int()
        # would accept them (int() tolerates "+0"/"00"/" 0"/"0_0"), so "rbln:" typos must
        # not silently resolve to the current device.
        for bad in ("rbln:", "rbln:+0", "rbln:00", "rbln: 0", "rbln:0_0", "rbln:0:0"):
            with self.assertRaises(ValueError, msg=f"{bad!r} was accepted as a valid device"):
                torch.rbln.memory_stats(bad)

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
    """``get_device_capability()`` advertises the dtypes the device computes on: fp16/bf16, the
    ones eager dispatch sends to the device. A tensor of any dtype takes device memory when it
    is created; ops on the other dtypes run on the host through the CPU fallback.

    Single-worker: measures global device memory."""

    @pytest.mark.test_set_ci
    def test_capability_is_the_dispatch_dtypes(self):
        advertised = set(torch.accelerator.get_device_capability()["supported_dtypes"])
        self.assertEqual(advertised, set(SUPPORTED_DTYPES))

    @pytest.mark.test_set_ci
    def test_every_dtype_takes_device_memory(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                before = torch.rbln.memory_allocated(0)
                scratch = torch.empty(1024 * 1024, dtype=dtype, device="rbln:0")
                self.assertEqual(torch.rbln.memory_allocated(0) - before, scratch.nbytes)
                del scratch


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
    memory (it is not a no-op or a query-only stub), on every device this process has
    allocated on, not just the current one (CUDA/XPU parity).

    Single-worker: asserts on global reserved-memory changes, so it must not race
    other workers' allocations."""

    @staticmethod
    def _hold_then_free_reserved(device):
        # Allocate then free a large buffer. The caching allocator keeps the
        # freed block as *reserved* (freeing alone does not return it to the
        # runtime), so reserved stays high until empty_cache() releases it.
        buf = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.float16)  # 32 MiB
        del buf
        return torch.rbln.memory_reserved(device)

    @pytest.mark.test_set_ci
    def test_empty_cache_releases_current_device(self):
        reserved_cached = self._hold_then_free_reserved("rbln:0")
        torch.accelerator.empty_cache()
        # Must actually release the cached block; a no-op or query-only
        # implementation would leave reserved unchanged.
        self.assertLess(torch.rbln.memory_reserved("rbln:0"), reserved_cached)

    @pytest.mark.test_set_ci
    @requires_logical_devices(2)
    def test_empty_cache_releases_every_initialized_device(self):
        reserved_cached = {device: self._hold_then_free_reserved(device) for device in ("rbln:0", "rbln:1")}
        with torch.rbln.device(0):
            torch.accelerator.empty_cache()
        for device, cached in reserved_cached.items():
            self.assertLess(torch.rbln.memory_reserved(device), cached, device)


def _empty_host_cache_worker():
    # The binding returns early until the device is initialized, so use it first.
    torch.randn(16, 16).to("rbln:0")
    pinned = torch.arange(64, dtype=torch.float32).pin_memory()
    torch.accelerator.empty_host_cache()
    assert pinned.is_pinned()
    assert torch.equal(pinned, torch.arange(64, dtype=torch.float32))


class TestAcceleratorEmptyHostCache(TestCase):
    """``torch.accelerator.empty_host_cache()`` reads the PrivateUse1 host allocator
    slot; with it empty, torch 2.13 segfaults (pytorch/pytorch#197593)."""

    @pytest.mark.test_set_ci
    def test_empty_host_cache_after_device_use(self):
        # A spawned process, so a segfault fails this test instead of the worker.
        run_in_isolated_process(_empty_host_cache_worker)


def _mem_get_info_dummy_worker():
    for query in (torch.rbln.mem_get_info, torch.rbln.mem_get_info_per_chiplet, torch.accelerator.get_memory_info):
        try:
            query(0)
        except RuntimeError as err:
            assert "RBLN_DUMMY_DEVICE" in str(err), err
        else:
            raise AssertionError(f"{query.__name__} reported figures for a dummy device")


@pytest.mark.test_set_ci
class TestMemGetInfo(TestCase):
    """``mem_get_info()`` is the driver's reading of the logical device's DRAM, summed over its
    physical NPUs and their chiplets; ``mem_get_info_per_chiplet()`` is the same reading
    before the sums. ``total`` is static; ``free`` is a reading another process may move
    between two queries, so it is bounded rather than matched across calls."""

    def setUp(self):
        super().setUp()
        if torch.rbln.device_count() == 0:
            self.skipTest("no rbln device available")
        if torch.rbln.is_dummy_device():
            self.skipTest("RBLN_DUMMY_DEVICE has no NPU to report on")

    def test_total_is_the_npus_memory(self):
        free, total = torch.rbln.mem_get_info(0)
        self.assertIsInstance(free, int)
        self.assertIsInstance(total, int)
        self.assertEqual(total, torch.rbln.get_device_properties(0).total_memory)
        self.assertLessEqual(free, total)
        # Same figures through the DeviceAllocator path torch.accelerator uses.
        accelerator_free, accelerator_total = torch.accelerator.get_memory_info(0)
        self.assertEqual(accelerator_total, total)
        self.assertLessEqual(accelerator_free, accelerator_total)

    def test_per_chiplet_breaks_the_total_down(self):
        properties = torch.rbln.get_device_properties(0)
        per_chiplet = torch.rbln.mem_get_info_per_chiplet(0)
        npus = [f"npu.{npu}" for npu in range(properties.npu_count)]
        chiplets = {npu: [f"{npu}.chiplet.{chiplet}" for chiplet in range(properties.num_chiplet)] for npu in npus}
        scopes = npus + [chiplet for npu in npus for chiplet in chiplets[npu]]
        self.assertEqual(
            set(per_chiplet), {f"{scope}.{field}" for scope in scopes for field in ("total", "used", "free")}
        )
        for scope in scopes:
            self.assertEqual(per_chiplet[f"{scope}.used"] + per_chiplet[f"{scope}.free"], per_chiplet[f"{scope}.total"])
        for npu in npus:
            for field in ("total", "used", "free"):
                self.assertEqual(
                    per_chiplet[f"{npu}.{field}"], sum(per_chiplet[f"{chiplet}.{field}"] for chiplet in chiplets[npu])
                )
        self.assertEqual(sum(per_chiplet[f"{npu}.total"] for npu in npus), torch.rbln.mem_get_info(0)[1])

    def test_used_covers_this_process_reserved_memory(self):
        live = torch.empty(32 * 1024 * 1024, dtype=torch.float16, device="rbln:0")
        free, total = torch.rbln.mem_get_info(0)
        self.assertGreaterEqual(total - free, torch.rbln.memory_reserved(0))
        self.assertGreaterEqual(torch.rbln.memory_reserved(0), live.nbytes)
        del live

    def test_accepts_device_forms_and_rejects_non_rbln(self):
        for bad in ("cpu", "cuda:0", torch.device("cpu")):
            with self.assertRaises(ValueError):
                torch.rbln.mem_get_info(bad)
        with self.assertRaises(ValueError):
            torch.rbln.mem_get_info(torch.rbln.device_count())
        total = torch.rbln.mem_get_info(0)[1]
        for dev in (0, "rbln:0", torch.device("rbln", 0)):
            self.assertEqual(torch.rbln.mem_get_info(dev)[1], total)

    def test_dummy_device_raises(self):
        """RBLN_DUMMY_DEVICE has no NPU behind it, so there is no figure to report."""
        with mock.patch.dict(os.environ, {"RBLN_DUMMY_DEVICE": "1"}):
            run_in_isolated_process(_mem_get_info_dummy_worker)


if __name__ == "__main__":
    run_tests()
