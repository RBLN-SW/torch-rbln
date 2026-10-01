# Owner(s): ["module: PrivateUse1"]

"""Statistics of the RBLN caching allocator, through ``torch.rbln`` and ``torch.accelerator``.

The allocator (``c10/rbln/RBLNCachingAllocator.h``) rounds a request of up to 1 MiB up to
512 B and carves it out of a 2 MiB segment; a larger request is rounded up to a multiple of
2 MiB. A freed block stays cached for the next request on its stream, ``empty_cache()``
releases the segments no live block is in, and an allocation the device cannot hold is
retried once after releasing the cache.
"""

import collections
import gc
import os
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 -- registers the rbln device + torch.rbln namespace


KiB = 1024
MiB = 1024 * KiB

# The allocator's sizes, as c10/rbln/RBLNCachingAllocator.h defines them.
SMALL_SIZE = 1 * MiB
SMALL_ROUND = 512
SMALL_SEGMENT = 2 * MiB
LARGE_ROUND = 2 * MiB

STATS = (
    "allocation",
    "segment",
    "active",
    "inactive_split",
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "inactive_split_bytes",
    "requested_bytes",
)
POOLS = ("all", "small_pool", "large_pool")
METRICS = ("current", "peak", "allocated", "freed")
COUNTERS = ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free")
STAT_KEYS = frozenset(COUNTERS).union(
    f"{stat}.{pool}.{metric}" for stat in STATS for pool in POOLS for metric in METRICS
)
# DeviceStats fields torch.accelerator.memory_stats() reports that the RBLN allocator leaves at zero.
UNTRACKED_KEYS = frozenset({"max_split_size", "num_sync_all_streams"}).union(
    f"{stat}.{metric}" for stat in ("oversize_allocations", "oversize_segments") for metric in METRICS
)


def _alloc(nbytes, device):
    """A float16 tensor of ``nbytes`` (an even count) bytes on ``device``."""
    return torch.empty(nbytes // 2, dtype=torch.float16, device=device)


def _settle(device):
    """Release every cached segment of ``device`` and restart its peak and accumulated stats.

    ``memory_stats()`` reports nothing for a device this process has not allocated on, so a
    first block is taken (and dropped) before anything is measured.
    """
    _alloc(SMALL_ROUND, device)
    gc.collect()
    torch.rbln.synchronize(device)
    torch.rbln.empty_cache(device)
    torch.rbln.reset_accumulated_memory_stats(device)
    torch.rbln.reset_peak_memory_stats(device)


def _changes(before, after):
    """The stats that differ between two snapshots, as ``after - before``."""
    return {key: after[key] - before[key] for key in after if after[key] != before[key]}


def _grown(pool, **amounts):
    """The stat changes that adding ``amounts`` to ``pool`` makes, starting from peaks equal to their currents."""
    return {
        f"{stat}.{p}.{metric}": amount
        for stat, amount in amounts.items()
        for p in ("all", pool)
        for metric in ("current", "peak", "allocated")
    }


def _shrunk(pool, **amounts):
    """The stat changes that taking ``amounts`` from ``pool`` makes."""
    changes = {}
    for stat, amount in amounts.items():
        for p in ("all", pool):
            changes[f"{stat}.{p}.current"] = -amount
            changes[f"{stat}.{p}.freed"] = amount
    return changes


def _check_peaks_reset(test, before, after):
    """Every peak of ``after`` is its current; every other stat is as in ``before``."""
    for key, value in after.items():
        if key.endswith(".peak"):
            test.assertEqual(value, after[key.removesuffix("peak") + "current"], key)
        else:
            test.assertEqual(value, before[key], key)


def _check_accumulated_reset(test, before, after):
    """Every accumulated stat of ``after`` is zero; currents and peaks are as in ``before``."""
    for key, value in after.items():
        if key in COUNTERS or key.endswith((".allocated", ".freed")):
            test.assertEqual(value, 0, key)
        else:
            test.assertEqual(value, before[key], key)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestMemoryStats(TestCase):
    """The ``torch.rbln`` memory API: what it reports and which device it asks."""

    def setUp(self):
        super().setUp()
        if torch.rbln.device_count() == 0:
            self.skipTest("no rbln device available")
        self.device = torch.device("rbln", 0)
        _settle(self.device)

    def _stats(self):
        return torch.rbln.memory_stats(self.device)

    def test_api_surface(self):
        for name in (
            "empty_cache",
            "memory_stats",
            "memory_stats_per_chiplet",
            "memory_summary",
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
            "reset_accumulated_memory_stats",
            "reset_peak_memory_stats",
            "mem_get_info",
            "mem_get_info_per_chiplet",
        ):
            self.assertTrue(callable(getattr(torch.rbln, name, None)), name)

    def test_stats_keys(self):
        """memory_stats() reports torch.cuda.memory_stats()'s keys for the allocator's stats."""
        stats = self._stats()
        self.assertEqual(set(stats), STAT_KEYS)
        for key, value in stats.items():
            self.assertIsInstance(value, int, key)
            self.assertGreaterEqual(value, 0, key)

    def test_return_types(self):
        self.assertIsInstance(torch.rbln.memory_stats(self.device), dict)
        self.assertIsInstance(torch.rbln.memory_stats_per_chiplet(self.device), dict)
        self.assertIsInstance(torch.rbln.memory_summary(self.device), str)
        self.assertIsInstance(torch.rbln.memory_allocated(self.device), int)
        self.assertIsInstance(torch.rbln.memory_reserved(self.device), int)
        self.assertIsInstance(torch.rbln.max_memory_allocated(self.device), int)
        self.assertIsInstance(torch.rbln.max_memory_reserved(self.device), int)
        self.assertIsNone(torch.rbln.empty_cache(self.device))
        self.assertIsNone(torch.rbln.reset_accumulated_memory_stats(self.device))
        self.assertIsNone(torch.rbln.reset_peak_memory_stats(self.device))

    def test_helpers_read_the_stats(self):
        live = _alloc(4 * MiB, self.device)
        stats = self._stats()
        self.assertEqual(torch.rbln.memory_allocated(self.device), stats["allocated_bytes.all.current"])
        self.assertEqual(torch.rbln.max_memory_allocated(self.device), stats["allocated_bytes.all.peak"])
        self.assertEqual(torch.rbln.memory_reserved(self.device), stats["reserved_bytes.all.current"])
        self.assertEqual(torch.rbln.max_memory_reserved(self.device), stats["reserved_bytes.all.peak"])
        self.assertGreaterEqual(torch.rbln.memory_allocated(self.device), 4 * MiB)
        self.assertGreaterEqual(torch.rbln.memory_reserved(self.device), torch.rbln.memory_allocated(self.device))
        del live

    def test_device_forms_agree(self):
        live = _alloc(4 * MiB, self.device)
        expected = self._stats()
        for form in (0, "rbln:0", torch.device("rbln", 0)):
            self.assertEqual(torch.rbln.memory_stats(form), expected, form)
        with torch.rbln.device(0):
            for form in (None, "rbln", torch.device("rbln")):
                self.assertEqual(torch.rbln.memory_stats(form), expected, form)
        del live

    def test_none_device_uses_current_device(self):
        """Every entry point asks the current device when given none."""
        current = torch.device("rbln", 1)
        stats = {
            "allocated_bytes.all.current": 512,
            "allocated_bytes.all.peak": 1024,
            "reserved_bytes.all.current": 2 * MiB,
            "reserved_bytes.all.peak": 4 * MiB,
        }
        per_chiplet = {f"npu.0.chiplet.0.{key}": value for key, value in stats.items()}
        cases = (
            ("memory_stats", "memory_stats", stats, stats),
            ("memory_allocated", "memory_stats", stats, 512),
            ("max_memory_allocated", "memory_stats", stats, 1024),
            ("memory_reserved", "memory_stats", stats, 2 * MiB),
            ("max_memory_reserved", "memory_stats", stats, 4 * MiB),
            ("memory_stats_per_chiplet", "memory_stats_per_chiplet", per_chiplet, per_chiplet),
            ("mem_get_info", "mem_get_info", (MiB, 2 * MiB), (MiB, 2 * MiB)),
            ("mem_get_info_per_chiplet", "mem_get_info_per_chiplet", {"npu.0.free": MiB}, {"npu.0.free": MiB}),
            ("empty_cache", "empty_cache", None, None),
            ("reset_peak_memory_stats", "reset_peak_memory_stats", None, None),
            ("reset_accumulated_memory_stats", "reset_accumulated_memory_stats", None, None),
        )
        for api, binding, returned, expected in cases:
            with (
                self.subTest(api=api),
                patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=current.index),
                patch("torch_rbln.memory._no_rbln_device", return_value=False),
                patch("torch_rbln.memory.torch_rbln._C._warmcache_clear") as warmcache_clear,
                patch(f"torch_rbln.memory.torch_rbln._C.{binding}", return_value=returned) as bound,
            ):
                self.assertEqual(getattr(torch.rbln, api)(), expected)
                bound.assert_called_once_with(current)
                if api == "empty_cache":
                    warmcache_clear.assert_called_once_with(current.index)

    def test_invalid_device_raises(self):
        count = torch.rbln.device_count()
        apis = (
            torch.rbln.memory_stats,
            torch.rbln.memory_stats_per_chiplet,
            torch.rbln.memory_allocated,
            torch.rbln.memory_reserved,
            torch.rbln.max_memory_allocated,
            torch.rbln.max_memory_reserved,
            torch.rbln.empty_cache,
            torch.rbln.reset_peak_memory_stats,
            torch.rbln.reset_accumulated_memory_stats,
            torch.rbln.mem_get_info,
            torch.rbln.mem_get_info_per_chiplet,
        )
        for api in apis:
            for bad in (-1, count, f"rbln:{count}", "cpu", torch.device("cpu")):
                with self.subTest(api=api.__name__, device=bad), self.assertRaises(ValueError):
                    api(bad)
        with self.assertRaises(RuntimeError):
            torch.device("rbln:-1")

    def test_peak_tracking(self):
        base = self._stats()["allocated_bytes.all.current"]
        first = _alloc(4 * KiB, self.device)
        second = _alloc(8 * KiB, self.device)
        self.assertEqual(torch.rbln.max_memory_allocated(self.device), base + 12 * KiB)

        del first
        self.assertEqual(torch.rbln.memory_allocated(self.device), base + 8 * KiB)
        self.assertEqual(torch.rbln.max_memory_allocated(self.device), base + 12 * KiB)

        del second
        torch.rbln.empty_cache(self.device)
        self.assertEqual(torch.rbln.memory_allocated(self.device), base)
        self.assertEqual(torch.rbln.max_memory_allocated(self.device), base + 12 * KiB)
        self.assertGreaterEqual(torch.rbln.max_memory_reserved(self.device), torch.rbln.memory_reserved(self.device))

    def test_reset_peak_memory_stats(self):
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        before = self._stats()
        self.assertGreater(before["allocated_bytes.all.peak"], before["allocated_bytes.all.current"])

        torch.rbln.reset_peak_memory_stats(self.device)
        _check_peaks_reset(self, before, self._stats())
        del live

    def test_reset_accumulated_memory_stats(self):
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        before = self._stats()
        self.assertEqual(before["allocation.all.allocated"], 2)
        self.assertEqual(before["allocation.all.freed"], 1)
        self.assertEqual(before["allocated_bytes.all.allocated"], 4 * KiB + 8 * MiB)

        torch.rbln.reset_accumulated_memory_stats(self.device)
        _check_accumulated_reset(self, before, self._stats())
        del live

    def test_empty_cache_keeps_live_memory(self):
        before = self._stats()
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        cached = self._stats()

        torch.rbln.empty_cache(self.device)
        flushed = self._stats()
        self.assertEqual(flushed["allocated_bytes.all.current"], cached["allocated_bytes.all.current"])
        self.assertEqual(flushed["allocation.all.current"], before["allocation.all.current"] + 1)
        self.assertLessEqual(flushed["reserved_bytes.all.current"], cached["reserved_bytes.all.current"])
        self.assertGreaterEqual(flushed["reserved_bytes.all.current"], flushed["allocated_bytes.all.current"])

        del live
        torch.rbln.empty_cache(self.device)
        emptied = self._stats()
        self.assertEqual(emptied["allocated_bytes.all.current"], before["allocated_bytes.all.current"])
        self.assertEqual(emptied["reserved_bytes.all.current"], before["reserved_bytes.all.current"])

    def test_malloc_free_cycles(self):
        """Allocating and freeing one size over and over takes device memory only once."""
        cycles = 8
        base = self._stats()["allocated_bytes.all.current"]
        for nbytes in (4 * KiB, 4 * MiB):
            with self.subTest(nbytes=nbytes):
                _alloc(nbytes, self.device)
                warm = self._stats()
                for _ in range(cycles):
                    _alloc(nbytes, self.device)
                after = self._stats()
                self.assertEqual(after["num_device_alloc"], warm["num_device_alloc"])
                self.assertEqual(after["segment.all.current"], warm["segment.all.current"])
                self.assertEqual(after["reserved_bytes.all.current"], warm["reserved_bytes.all.current"])
                self.assertEqual(after["allocation.all.allocated"] - warm["allocation.all.allocated"], cycles)
                self.assertEqual(after["allocation.all.freed"] - warm["allocation.all.freed"], cycles)
                self.assertEqual(
                    after["allocated_bytes.all.allocated"] - warm["allocated_bytes.all.allocated"], cycles * nbytes
                )
                self.assertEqual(after["allocated_bytes.all.current"], base)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestCachingAllocator(TestCase):
    """How the allocator lays blocks out, as its stats show.

    Each test starts with no free block in any segment, so the first request of each pool
    opens a segment of its own.
    """

    def setUp(self):
        super().setUp()
        if torch.rbln.device_count() == 0:
            self.skipTest("no rbln device available")
        self.device = torch.device("rbln", 0)
        _settle(self.device)
        self.assertEqual(
            self._stats()["inactive_split_bytes.all.current"],
            0,
            "a live block left by an earlier test shares a segment with free space the requests below would take",
        )

    def _stats(self):
        return torch.rbln.memory_stats(self.device)

    def test_small_block_lifecycle(self):
        """A small block opens a 2 MiB segment, rejoins it when freed, and empty_cache() releases it."""
        start = self._stats()
        block = _alloc(1000, self.device)
        allocated = self._stats()
        expected = _grown(
            "small_pool",
            allocation=1,
            active=1,
            allocated_bytes=1024,
            active_bytes=1024,
            requested_bytes=1000,
            segment=1,
            reserved_bytes=SMALL_SEGMENT,
            inactive_split=1,
            inactive_split_bytes=SMALL_SEGMENT - 1024,
        )
        expected["num_device_alloc"] = 1
        self.assertEqual(_changes(start, allocated), expected)

        del block
        freed = self._stats()
        expected = _shrunk(
            "small_pool",
            allocation=1,
            active=1,
            allocated_bytes=1024,
            active_bytes=1024,
            requested_bytes=1000,
            inactive_split=1,
            inactive_split_bytes=SMALL_SEGMENT - 1024,
        )
        self.assertEqual(_changes(allocated, freed), expected)

        torch.rbln.empty_cache(self.device)
        expected = _shrunk("small_pool", segment=1, reserved_bytes=SMALL_SEGMENT)
        expected["num_device_free"] = 1
        self.assertEqual(_changes(freed, self._stats()), expected)

    def test_small_requests_round_up_to_512_bytes(self):
        held = []
        for nbytes, size in (
            (2, 512),
            (510, 512),
            (512, 512),
            (514, 1024),
            (1000, 1024),
            (64 * KiB + 2, 64 * KiB + 512),
            (SMALL_SIZE, SMALL_SIZE),
        ):
            with self.subTest(nbytes=nbytes):
                before = self._stats()
                held.append(_alloc(nbytes, self.device))
                changes = _changes(before, self._stats())
                self.assertEqual(changes["allocated_bytes.small_pool.current"], size)
                self.assertEqual(changes["requested_bytes.small_pool.current"], nbytes)
                self.assertEqual(changes["allocation.small_pool.current"], 1)
                self.assertEqual([key for key in changes if ".large_pool." in key], [])

    def test_small_segment_serves_many_blocks(self):
        """32 requests of just under 64 KiB fill one 2 MiB segment back to back."""
        nbytes = 64 * KiB - 100
        before = self._stats()
        blocks = [_alloc(nbytes, self.device) for _ in range(32)]
        full = self._stats()
        self.assertEqual(full["segment.small_pool.current"] - before["segment.small_pool.current"], 1)
        self.assertEqual(full["num_device_alloc"] - before["num_device_alloc"], 1)
        self.assertEqual(
            full["reserved_bytes.small_pool.current"] - before["reserved_bytes.small_pool.current"], SMALL_SEGMENT
        )
        self.assertEqual(
            full["allocated_bytes.small_pool.current"] - before["allocated_bytes.small_pool.current"], SMALL_SEGMENT
        )
        self.assertEqual(
            full["requested_bytes.small_pool.current"] - before["requested_bytes.small_pool.current"], 32 * nbytes
        )
        self.assertEqual(full["inactive_split_bytes.small_pool.current"], 0)
        start = min(block.data_ptr() for block in blocks)
        self.assertEqual(sorted(block.data_ptr() - start for block in blocks), [i * 64 * KiB for i in range(32)])

        extra = _alloc(SMALL_ROUND, self.device)
        after = self._stats()
        self.assertEqual(after["segment.small_pool.current"] - full["segment.small_pool.current"], 1)
        self.assertEqual(after["num_device_alloc"] - full["num_device_alloc"], 1)
        del blocks, extra

    def test_large_requests_round_up_to_2_mib(self):
        """A request over 1 MiB gets a segment of its own, a whole number of 2 MiB long."""
        held = []
        for nbytes, size in ((SMALL_SIZE + 2, 2 * MiB), (2 * MiB, 2 * MiB), (3 * MiB, 4 * MiB), (5 * MiB + 2, 6 * MiB)):
            with self.subTest(nbytes=nbytes):
                before = self._stats()
                held.append(_alloc(nbytes, self.device))
                expected = _grown(
                    "large_pool",
                    allocation=1,
                    active=1,
                    allocated_bytes=size,
                    active_bytes=size,
                    requested_bytes=nbytes,
                    segment=1,
                    reserved_bytes=size,
                )
                expected["num_device_alloc"] = 1
                self.assertEqual(_changes(before, self._stats()), expected)

    def test_freed_block_is_reused(self):
        """A request the size of a freed block gets that block back; reserved memory does not grow."""
        for nbytes in (256 * KiB, 8 * MiB):
            with self.subTest(nbytes=nbytes):
                block = _alloc(nbytes, self.device)
                ptr = block.data_ptr()
                del block
                before = self._stats()
                again = _alloc(nbytes, self.device)
                after = self._stats()
                self.assertEqual(again.data_ptr(), ptr)
                for key in ("segment.all.current", "reserved_bytes.all.current", "num_device_alloc"):
                    self.assertEqual(after[key], before[key], key)
                self.assertEqual(after["allocated_bytes.all.current"] - before["allocated_bytes.all.current"], nbytes)
                del again

    def test_cached_large_block_serves_a_smaller_request(self):
        """A cached large block serves any request it holds; the rest of it stays cached."""
        cached = _alloc(8 * MiB, self.device)
        start = cached.data_ptr()
        del cached
        before = self._stats()
        block = _alloc(2 * MiB, self.device)
        after = self._stats()
        self.assertEqual(block.data_ptr(), start)
        for key in ("segment.all.current", "reserved_bytes.all.current", "num_device_alloc"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after["inactive_split.large_pool.current"] - before["inactive_split.large_pool.current"], 1)
        self.assertEqual(
            after["inactive_split_bytes.large_pool.current"] - before["inactive_split_bytes.large_pool.current"],
            6 * MiB,
        )

        del block
        rejoined = self._stats()
        self.assertEqual(rejoined["inactive_split_bytes.large_pool.current"], 0)
        self.assertEqual(rejoined["reserved_bytes.all.current"], before["reserved_bytes.all.current"])

    def test_empty_cache_releases_only_empty_segments(self):
        """A segment with a live block stays; a wholly free one is released, on either entry point."""
        first = _alloc(KiB, self.device)
        second = _alloc(KiB, self.device)
        self.assertEqual(second.data_ptr(), first.data_ptr() + KiB)
        large = _alloc(4 * MiB, self.device)
        kept = _alloc(4 * MiB, self.device)
        del first, large
        before = self._stats()
        torch.rbln.empty_cache(self.device)
        expected = _shrunk("large_pool", segment=1, reserved_bytes=4 * MiB)
        expected["num_device_free"] = 1
        self.assertEqual(_changes(before, self._stats()), expected)

        del second, kept
        before = self._stats()
        torch.accelerator.empty_cache()
        changes = _changes(before, self._stats())
        self.assertEqual(changes["segment.small_pool.current"], -1)
        self.assertEqual(changes["segment.large_pool.current"], -1)
        self.assertEqual(changes["reserved_bytes.small_pool.current"], -SMALL_SEGMENT)
        self.assertEqual(changes["reserved_bytes.large_pool.current"], -4 * MiB)
        self.assertEqual(changes["num_device_free"], 2)

    def test_freed_block_is_reused_only_on_its_stream(self):
        side = torch.rbln.Stream(self.device)
        self.assertNotEqual(side.stream_id, torch.rbln.current_stream(self.device).stream_id)
        nbytes = 256 * KiB
        with torch.rbln.stream(side):
            block = _alloc(nbytes, self.device)
        ptr = block.data_ptr()
        del block

        before = self._stats()
        elsewhere = _alloc(nbytes, self.device)
        crossed = self._stats()
        self.assertNotEqual(elsewhere.data_ptr(), ptr)
        self.assertEqual(crossed["segment.small_pool.current"] - before["segment.small_pool.current"], 1)
        self.assertEqual(crossed["num_device_alloc"] - before["num_device_alloc"], 1)

        with torch.rbln.stream(side):
            again = _alloc(nbytes, self.device)
        after = self._stats()
        self.assertEqual(again.data_ptr(), ptr)
        self.assertEqual(after["segment.small_pool.current"], crossed["segment.small_pool.current"])
        self.assertEqual(after["num_device_alloc"], crossed["num_device_alloc"])
        del elsewhere, again

    def test_out_of_memory_retries_after_releasing_the_cache(self):
        """A request the device cannot hold releases the cached segments, retries once, and raises."""
        if torch.rbln.is_dummy_device():
            self.skipTest("RBLN_DUMMY_DEVICE takes device memory from the host")
        total = torch.rbln.mem_get_info(self.device)[1]
        cached = _alloc(4 * MiB, self.device)
        del cached
        before = self._stats()
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            _alloc(total + LARGE_ROUND, self.device)
        expected = _shrunk("large_pool", segment=1, reserved_bytes=4 * MiB)
        expected.update(num_alloc_retries=1, num_ooms=1, num_device_free=1)
        self.assertEqual(_changes(before, self._stats()), expected)

    def test_empty_tensor_takes_no_block(self):
        before = self._stats()
        empty = torch.empty(0, dtype=torch.float16, device=self.device)
        self.assertEqual(_changes(before, self._stats()), {})
        del empty


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestAcceleratorMemoryAPI(TestCase):
    """``torch.accelerator``'s memory API over the RBLN ``DeviceAllocator``.

    It reads the same allocator ``torch.rbln`` does, through ``getDeviceStats``,
    ``resetAccumulatedStats``, ``resetPeakStats``, ``emptyCache`` and ``initialized``.
    """

    def setUp(self):
        super().setUp()
        if not torch.rbln.is_available():
            self.skipTest("RBLN device not available")
        if torch.accelerator.current_accelerator().type != "rbln":
            self.skipTest("Current accelerator is not RBLN")
        self.device = torch.device("rbln", 0)
        _settle(self.device)

    def test_memory_stats_is_a_sorted_ordered_dict(self):
        stats = torch.accelerator.memory_stats(0)
        self.assertIsInstance(stats, collections.OrderedDict)
        self.assertEqual(list(stats), sorted(stats))

    def test_memory_stats_reports_122_keys(self):
        """Every DeviceStats field torch flattens: torch.rbln's 112 keys and 10 the allocator leaves at zero."""
        stats = torch.accelerator.memory_stats(0)
        self.assertEqual(len(stats), 122)
        self.assertEqual(set(stats), STAT_KEYS | UNTRACKED_KEYS)

    def test_memory_stats_values_are_int(self):
        for key, value in torch.accelerator.memory_stats(0).items():
            self.assertIsInstance(value, int, key)

    def test_untracked_stats_stay_zero(self):
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        torch.accelerator.empty_cache()
        stats = torch.accelerator.memory_stats(0)
        for key in UNTRACKED_KEYS:
            self.assertEqual(stats[key], 0, key)
        del live

    def test_helpers_read_the_stats(self):
        live = _alloc(4 * MiB, self.device)
        stats = torch.accelerator.memory_stats(0)
        helpers = (
            ("memory_allocated", "allocated_bytes.all.current"),
            ("max_memory_allocated", "allocated_bytes.all.peak"),
            ("memory_reserved", "reserved_bytes.all.current"),
            ("max_memory_reserved", "reserved_bytes.all.peak"),
        )
        for helper, key in helpers:
            value = getattr(torch.accelerator, helper)(0)
            self.assertIsInstance(value, int, helper)
            self.assertEqual(value, stats[key], helper)
            self.assertEqual(value, getattr(torch.rbln, helper)(self.device), helper)
        self.assertGreaterEqual(torch.accelerator.memory_allocated(0), 4 * MiB)
        self.assertGreaterEqual(torch.accelerator.max_memory_allocated(0), torch.accelerator.memory_allocated(0))
        self.assertGreaterEqual(torch.accelerator.max_memory_reserved(0), torch.accelerator.memory_reserved(0))
        del live

    def test_stats_match_torch_rbln(self):
        """Every key torch.rbln reports agrees, through an allocate/free/empty_cache cycle."""

        def agreed(stage):
            rbln = torch.rbln.memory_stats(self.device)
            accelerator = torch.accelerator.memory_stats(0)
            self.assertEqual({key: accelerator[key] for key in STAT_KEYS}, rbln, stage)
            return rbln

        start = agreed("start")
        small = _alloc(4 * KiB, self.device)
        large = _alloc(4 * MiB, self.device)
        allocated = agreed("allocated")
        self.assertEqual(
            allocated["allocated_bytes.all.current"] - start["allocated_bytes.all.current"], 4 * KiB + 4 * MiB
        )
        del small, large
        freed = agreed("freed")
        self.assertEqual(freed["allocated_bytes.all.current"], start["allocated_bytes.all.current"])
        torch.accelerator.empty_cache()
        emptied = agreed("emptied")
        self.assertEqual(emptied["reserved_bytes.all.current"], start["reserved_bytes.all.current"])

    def test_reset_peak_memory_stats(self):
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        before = torch.rbln.memory_stats(self.device)
        self.assertGreater(before["allocated_bytes.all.peak"], before["allocated_bytes.all.current"])

        torch.accelerator.reset_peak_memory_stats(0)
        _check_peaks_reset(self, before, torch.rbln.memory_stats(self.device))
        del live

    def test_reset_accumulated_memory_stats(self):
        live = _alloc(4 * KiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        before = torch.rbln.memory_stats(self.device)
        self.assertGreater(before["allocation.all.freed"], 0)

        torch.accelerator.reset_accumulated_memory_stats(0)
        _check_accumulated_reset(self, before, torch.rbln.memory_stats(self.device))
        del live

    def test_empty_cache_keeps_live_memory(self):
        live = _alloc(4 * MiB, self.device)
        gone = _alloc(8 * MiB, self.device)
        del gone
        before = torch.accelerator.memory_stats(0)
        torch.accelerator.empty_cache()
        after = torch.accelerator.memory_stats(0)
        self.assertEqual(after["allocated_bytes.all.current"], before["allocated_bytes.all.current"])
        self.assertLessEqual(after["reserved_bytes.all.current"], before["reserved_bytes.all.current"])
        self.assertGreaterEqual(after["reserved_bytes.all.current"], after["allocated_bytes.all.current"])
        del live

    def test_invalid_device_index_raises(self):
        with self.assertRaises(RuntimeError):
            torch.accelerator.memory_stats(torch.rbln.device_count())


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestPerChipletMemoryStats(TestCase):
    """memory_stats_per_chiplet() / memory_summary()."""

    def setUp(self):
        super().setUp()
        if torch.rbln.device_count() == 0:
            self.skipTest("no rbln device available")
        self.device = torch.device("rbln", 0)
        _settle(self.device)

    def _by_chiplet(self, per_chiplet):
        """``per_chiplet`` as ``{(npu, chiplet): stats}``."""
        chiplets = {}
        for key, value in per_chiplet.items():
            npu_tag, npu, chiplet_tag, chiplet, stat = key.split(".", 4)
            self.assertEqual((npu_tag, chiplet_tag), ("npu", "chiplet"), key)
            self.assertTrue(npu.isdigit() and chiplet.isdigit(), key)
            chiplets.setdefault((int(npu), int(chiplet)), {})[stat] = value
        return chiplets

    def test_key_schema(self):
        """Each chiplet carries memory_stats()'s keys under an npu.<n>.chiplet.<c>. prefix."""
        chiplets = self._by_chiplet(torch.rbln.memory_stats_per_chiplet(self.device))
        self.assertGreater(len(chiplets), 0)
        for location, stats in chiplets.items():
            self.assertEqual(set(stats), STAT_KEYS, location)
            for key, value in stats.items():
                self.assertIsInstance(value, int, key)

    def test_breakdown_sums_to_aggregate(self):
        """The chiplets account for exactly what memory_stats() aggregates.

        Peaks are per chiplet, so their sum bounds the joint peak from above; memory_summary()'s
        total row inherits that, and its docstring says so.
        """
        small = _alloc(4 * KiB, self.device)
        large = _alloc(4 * MiB, self.device)
        aggregate = torch.rbln.memory_stats(self.device)
        chiplets = self._by_chiplet(torch.rbln.memory_stats_per_chiplet(self.device))
        for key in STAT_KEYS:
            total = sum(stats[key] for stats in chiplets.values())
            if key.endswith(".peak"):
                self.assertGreaterEqual(total, aggregate[key], key)
            else:
                self.assertEqual(total, aggregate[key], key)
        self.assertGreaterEqual(aggregate["allocated_bytes.all.current"], 4 * KiB + 4 * MiB)
        del small, large

    def test_driver_usage_covers_reserved_segments(self):
        """The driver counts the segments the allocator reserved on a chiplet as used there."""
        if torch.rbln.is_dummy_device():
            self.skipTest("RBLN_DUMMY_DEVICE has no driver memory figures")
        live = _alloc(8 * MiB, self.device)
        driver = torch.rbln.mem_get_info_per_chiplet(self.device)
        for (npu, chiplet), stats in self._by_chiplet(torch.rbln.memory_stats_per_chiplet(self.device)).items():
            self.assertGreaterEqual(
                driver[f"npu.{npu}.chiplet.{chiplet}.used"], stats["reserved_bytes.all.current"], (npu, chiplet)
            )
        del live

    def test_summary_names_its_scope(self):
        """The table must say the numbers are one process's allocator, not the NPU's."""
        summary = torch.rbln.memory_summary(self.device)
        self.assertIn("device=rbln:0", summary)
        self.assertIn(f"pid {os.getpid()}", summary)
        self.assertIn("caching allocator only, this process only", summary)
        self.assertIn("npu: physical NPU id as in device_summary()", summary)
        self.assertIn("npu", summary)
        self.assertIn("chiplet", summary)
        self.assertIn("total", summary)

    def test_summary_totals_are_the_chiplet_sums(self):
        small = _alloc(4 * KiB, self.device)
        large = _alloc(4 * MiB, self.device)
        summary = torch.rbln.memory_summary(self.device)
        chiplets = self._by_chiplet(torch.rbln.memory_stats_per_chiplet(self.device)).values()
        columns = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
        )
        totals = next(line.split() for line in summary.splitlines() if line.split()[:1] == ["total"])
        self.assertEqual(totals[1:], [f"{sum(stats[key] for stats in chiplets) / MiB:.1f}" for key in columns])
        self.assertNotEqual(totals[1], "0.0")
        retries = sum(stats["num_alloc_retries"] for stats in chiplets)
        ooms = sum(stats["num_ooms"] for stats in chiplets)
        self.assertIn(f"alloc retries: {retries}   ooms: {ooms}", summary)
        del small, large

    def test_summary_rows_carry_physical_npu_ids(self):
        """Row `npu` is the id device_summary() prints, not the NPU's position in the device."""
        from torch_rbln._internal.rsd_utils import get_physical_device_ids

        physical = get_physical_device_ids(0)
        self.assertTrue(physical)
        summary = torch.rbln.memory_summary(self.device)
        table = summary.split("\n")[4:]
        row_ids = sorted({int(line.split()[0]) for line in table if line.strip() and line.split()[0].isdigit()})
        self.assertEqual(row_ids, sorted(set(physical)))

    def test_summary_without_device(self):
        """No RBLN device / uninitialized allocator degrades to a notice, not a raise."""
        with patch("torch_rbln.memory.memory_stats_per_chiplet", return_value={}):
            summary = torch.rbln.memory_summary(self.device)
        self.assertIn("no statistics", summary)

    def test_second_logical_device(self):
        """rbln:1 has an allocator of its own, which every entry point reaches."""
        if torch.rbln.device_count() < 2:
            self.skipTest("needs at least 2 logical rbln devices")
        second = torch.device("rbln", 1)
        first_before = torch.rbln.memory_stats(self.device)
        tensor = _alloc(4 * MiB, second)
        try:
            stats = torch.rbln.memory_stats(second)
            self.assertGreaterEqual(stats["allocated_bytes.all.current"], 4 * MiB)
            self.assertEqual(torch.rbln.memory_stats(self.device), first_before)
            with torch.rbln.device(second):
                self.assertEqual(torch.rbln.memory_stats(), stats)
            accelerator = torch.accelerator.memory_stats(1)
            self.assertEqual({key: accelerator[key] for key in STAT_KEYS}, stats)
            self.assertGreater(len(torch.rbln.memory_stats_per_chiplet(second)), 0)
            self.assertIn("device=rbln:1", torch.rbln.memory_summary(second))
        finally:
            del tensor
        torch.rbln.empty_cache(second)


instantiate_device_type_tests(TestMemoryStats, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCachingAllocator, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAcceleratorMemoryAPI, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestPerChipletMemoryStats, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
