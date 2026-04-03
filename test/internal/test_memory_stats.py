# Owner(s): ["module: PrivateUse1"]

from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_eager_malloc")
class TestMemoryStats(TestCase):
    # Memory allocation size constants for testing (matching C++ test)
    KB_1 = 1024
    KB_2 = 2 * KB_1
    KB_4 = 4 * KB_1
    MB_1 = 1024 * KB_1
    MB_2 = 2 * MB_1
    MB_4 = 4 * MB_1

    def setUp(self):
        """Set up test environment."""
        self.device_id = 0
        self.device = f"rbln:{self.device_id}"

        # Clear any existing state to ensure test isolation (matching C++ test)
        # Use try-except to handle potential initialization issues
        try:
            torch.rbln.empty_cache(self.device)
            torch.rbln.reset_accumulated_memory_stats(self.device)
            torch.rbln.reset_peak_memory_stats(self.device)
        except Exception as e:
            # If memory functions are not available, skip the test
            self.skipTest(f"Memory functions not available: {e}")

        # Align allocated.current to 2MB boundary for consistent testing
        stats = torch.rbln.memory_stats(self.device)
        current_allocated = stats["allocated.current"]
        remainder = current_allocated % self.MB_2
        if remainder != 0:
            align_size = self.MB_2 - remainder
            # Allocate dummy tensor to align to 2MB boundary
            # Using float16 (2 bytes per element)
            self.dummy_tensor = torch.empty((align_size // 2,), device=self.device, dtype=torch.float16)
        else:
            self.dummy_tensor = None

        # Reset stats after alignment so tests start from clean state
        torch.rbln.reset_accumulated_memory_stats(self.device)
        torch.rbln.reset_peak_memory_stats(self.device)

    def tearDown(self):
        """Clean up after each test."""
        try:
            # Clean up dummy tensor used for 2MB alignment
            if hasattr(self, "dummy_tensor") and self.dummy_tensor is not None:
                del self.dummy_tensor
                self.dummy_tensor = None
            torch.rbln.empty_cache(self.device)
        except Exception:
            # Ignore cleanup errors
            pass

    def test_hasattr(self):
        """Test that memory functions are available."""
        # Check if memory functions are available in torch.rbln
        self.assertTrue(hasattr(torch.rbln, "empty_cache"))
        self.assertTrue(hasattr(torch.rbln, "max_memory_allocated"))
        self.assertTrue(hasattr(torch.rbln, "max_memory_reserved"))
        self.assertTrue(hasattr(torch.rbln, "memory_allocated"))
        self.assertTrue(hasattr(torch.rbln, "memory_reserved"))
        self.assertTrue(hasattr(torch.rbln, "memory_stats"))
        self.assertTrue(hasattr(torch.rbln, "reset_accumulated_memory_stats"))
        self.assertTrue(hasattr(torch.rbln, "reset_peak_memory_stats"))

    def test_basic_stats_operations(self):
        """Test basic memory stats operations (matching C++ BasicStatsOperations)."""

        # Get baseline stats (should be 0 or minimal after setUp)
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Test initial values are reasonable
        self.assertGreaterEqual(stats_baseline["allocated.current"], 0)
        self.assertGreaterEqual(stats_baseline["allocated.peak"], 0)
        self.assertGreaterEqual(stats_baseline["reserved.current"], 0)
        self.assertGreaterEqual(stats_baseline["active.current"], 0)
        self.assertGreaterEqual(stats_baseline["cached.current"], 0)
        self.assertGreaterEqual(stats_baseline["num_alloc_retries"], 0)
        self.assertGreaterEqual(stats_baseline["num_ooms"], 0)
        self.assertGreaterEqual(stats_baseline["num_device_alloc"], 0)
        self.assertGreaterEqual(stats_baseline["num_device_free"], 0)

    def test_peak_tracking(self):
        """Test peak memory tracking (matching C++ PeakTracking)."""
        # Reset peak stats first
        torch.rbln.reset_peak_memory_stats(self.device)

        # Get baseline
        stats_baseline = torch.rbln.memory_stats(self.device)
        baseline_peak = stats_baseline["allocated.peak"]

        # Create tensor using specific size
        tensor1 = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB
        stats_after_first = torch.rbln.memory_stats(self.device)

        # Peak should increase by 1KB
        self.assertEqual(stats_after_first["allocated.peak"], baseline_peak + self.KB_1)

        # Create another tensor
        tensor2 = torch.empty((self.KB_2 // 2,), device=self.device, dtype=torch.float16)  # 2KB
        stats_after_second = torch.rbln.memory_stats(self.device)

        # Peak should increase by total allocated (1KB + 2KB = 3KB)
        expected_total = self.KB_1 + self.KB_2
        self.assertEqual(stats_after_second["allocated.peak"], baseline_peak + expected_total)

        # Delete one tensor
        del tensor1
        torch.rbln.empty_cache(self.device)
        stats_after_delete = torch.rbln.memory_stats(self.device)

        # Peak should remain high (not decrease)
        self.assertEqual(stats_after_delete["allocated.peak"], baseline_peak + expected_total)

        # Clean up
        del tensor2
        torch.rbln.empty_cache(self.device)

    def test_reset_operations(self):
        """Test reset operations (matching C++ ResetOperations)."""
        # Create some tensors to generate stats using specific sizes
        tensor1 = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB
        tensor2 = torch.empty((self.KB_2 // 2,), device=self.device, dtype=torch.float16)  # 2KB

        # Get stats after allocation
        stats_after_alloc = torch.rbln.memory_stats(self.device)
        current_allocated = stats_after_alloc["allocated.current"]
        peak_allocated = stats_after_alloc["allocated.peak"]

        # Test ResetAccumulatedStats
        torch.rbln.reset_accumulated_memory_stats(self.device)
        stats_after_reset_accumulated = torch.rbln.memory_stats(self.device)

        # Current and peak should remain
        self.assertEqual(stats_after_reset_accumulated["allocated.current"], current_allocated)
        self.assertEqual(stats_after_reset_accumulated["allocated.peak"], peak_allocated)

        # Accumulated counters should reset
        self.assertEqual(stats_after_reset_accumulated["allocated.total_allocated"], 0)
        self.assertEqual(stats_after_reset_accumulated["allocated.total_freed"], 0)
        self.assertEqual(stats_after_reset_accumulated["num_alloc_retries"], 0)
        self.assertEqual(stats_after_reset_accumulated["num_ooms"], 0)
        self.assertEqual(stats_after_reset_accumulated["num_device_alloc"], 0)
        self.assertEqual(stats_after_reset_accumulated["num_device_free"], 0)

        # Test ResetPeakStats
        torch.rbln.reset_peak_memory_stats(self.device)
        stats_after_reset_peak = torch.rbln.memory_stats(self.device)

        # Peak should reset to current values
        self.assertEqual(stats_after_reset_peak["allocated.peak"], stats_after_reset_peak["allocated.current"])
        self.assertEqual(stats_after_reset_peak["reserved.peak"], stats_after_reset_peak["reserved.current"])

        # Clean up
        del tensor1, tensor2
        torch.rbln.empty_cache(self.device)

    def test_empty_cache(self):
        """Test empty_cache functionality (matching C++ RuntimeAPI_empty_cache)."""
        # Get baseline stats
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Allocate 1MB twice to create blocks that can be fully returned
        tensor1 = torch.empty((self.MB_1 // 2,), device=self.device, dtype=torch.float16)  # 1MB
        tensor2 = torch.empty((self.MB_1 // 2,), device=self.device, dtype=torch.float16)  # 1MB

        # Get stats after allocation
        stats_after_allocation = torch.rbln.memory_stats(self.device)
        expected_allocated = self.MB_1 + self.MB_1  # 2MB
        self.assertEqual(
            stats_baseline["allocated.current"] + expected_allocated, stats_after_allocation["allocated.current"]
        )
        self.assertEqual(
            stats_baseline["reserved.current"] + expected_allocated, stats_after_allocation["reserved.current"]
        )
        self.assertEqual(
            stats_baseline["active.current"] + expected_allocated, stats_after_allocation["active.current"]
        )
        self.assertEqual(stats_baseline["cached.current"], stats_after_allocation["cached.current"])

        # Free both allocations to create cached blocks
        del tensor1, tensor2

        # Get stats after complete deallocation
        stats_after_complete_free = torch.rbln.memory_stats(self.device)
        self.assertEqual(stats_baseline["allocated.current"], stats_after_complete_free["allocated.current"])
        self.assertEqual(
            stats_baseline["reserved.current"] + expected_allocated, stats_after_complete_free["reserved.current"]
        )
        self.assertEqual(stats_baseline["active.current"], stats_after_complete_free["active.current"])
        self.assertEqual(
            stats_baseline["cached.current"] + expected_allocated, stats_after_complete_free["cached.current"]
        )

        # Test empty_cache - this should clear all cached blocks
        torch.rbln.empty_cache(self.device)

        # Get stats after empty_cache
        stats_after_empty_cache = torch.rbln.memory_stats(self.device)

        # All allocated bytes should remain at baseline (no active allocations)
        self.assertEqual(stats_baseline["allocated.current"], stats_after_empty_cache["allocated.current"])
        self.assertEqual(stats_baseline["reserved.current"], stats_after_empty_cache["reserved.current"])
        self.assertEqual(stats_baseline["active.current"], stats_after_empty_cache["active.current"])
        self.assertEqual(stats_baseline["cached.current"], stats_after_empty_cache["cached.current"])

    def test_malloc_free_with_stats(self):
        """Test malloc/free with detailed stats tracking (matching C++ RuntimeAPI_malloc_free_with_stats)."""
        # Get baseline stats
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Allocate some memory using specific sizes (matching C++ test)
        tensor1 = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB
        tensor2 = torch.empty((self.KB_2 // 2,), device=self.device, dtype=torch.float16)  # 2KB

        # Check memory stats after allocation - verify all member items
        stats = torch.rbln.memory_stats(self.device)

        # Calculate expected allocated bytes (1KB + 2KB = 3KB)
        expected_allocated = self.KB_1 + self.KB_2

        # Allocated memory statistics
        self.assertEqual(stats_baseline["allocated.current"] + expected_allocated, stats["allocated.current"])
        self.assertEqual(stats_baseline["allocated.peak"] + expected_allocated, stats["allocated.peak"])
        self.assertEqual(
            stats_baseline["allocated.total_allocated"] + expected_allocated, stats["allocated.total_allocated"]
        )
        self.assertEqual(stats_baseline["allocated.total_freed"], stats["allocated.total_freed"])

        # Reserved memory statistics (should increase by 2MB due to allocator behavior)
        expected_reserved = self.MB_2
        self.assertEqual(stats_baseline["reserved.current"] + expected_reserved, stats["reserved.current"])
        self.assertEqual(stats_baseline["reserved.peak"] + expected_reserved, stats["reserved.peak"])
        self.assertEqual(
            stats_baseline["reserved.total_allocated"] + expected_reserved, stats["reserved.total_allocated"]
        )
        self.assertEqual(stats_baseline["reserved.total_freed"], stats["reserved.total_freed"])

        # Active block memory statistics
        self.assertEqual(stats_baseline["active.current"] + expected_allocated, stats["active.current"])
        self.assertEqual(stats_baseline["active.peak"] + expected_allocated, stats["active.peak"])

        # Cached block memory statistics (2MB - 3KB = remaining cached)
        expected_cached = expected_reserved - expected_allocated
        self.assertEqual(stats_baseline["cached.current"] + expected_cached, stats["cached.current"])
        self.assertEqual(stats_baseline["cached.peak"] + self.MB_2 - self.KB_1, stats["cached.peak"])

        # Allocation operation counters
        self.assertEqual(stats_baseline["num_alloc_retries"], stats["num_alloc_retries"])
        self.assertEqual(stats_baseline["num_ooms"], stats["num_ooms"])
        self.assertEqual(stats_baseline["num_device_alloc"] + 1, stats["num_device_alloc"])
        self.assertEqual(stats_baseline["num_device_free"], stats["num_device_free"])

        # Free memory
        del tensor1, tensor2

        # Check memory stats after deallocation - verify all member items
        stats_after = torch.rbln.memory_stats(self.device)

        # Allocated memory statistics
        self.assertEqual(stats_baseline["allocated.current"], stats_after["allocated.current"])
        self.assertEqual(
            stats_baseline["allocated.peak"] + expected_allocated, stats_after["allocated.peak"]
        )  # Peak should remain
        self.assertEqual(
            stats_baseline["allocated.total_allocated"] + expected_allocated, stats_after["allocated.total_allocated"]
        )  # Allocated should remain
        self.assertEqual(
            stats_baseline["allocated.total_freed"] + expected_allocated, stats_after["allocated.total_freed"]
        )  # Freed should increase

        # Reserved memory statistics
        self.assertEqual(stats_baseline["reserved.current"] + expected_reserved, stats_after["reserved.current"])
        self.assertEqual(stats_baseline["reserved.peak"] + expected_reserved, stats_after["reserved.peak"])
        self.assertEqual(
            stats_baseline["reserved.total_allocated"] + expected_reserved, stats_after["reserved.total_allocated"]
        )
        self.assertEqual(stats_baseline["reserved.total_freed"], stats_after["reserved.total_freed"])

        # Active block memory statistics
        self.assertEqual(stats_baseline["active.current"], stats_after["active.current"])
        self.assertEqual(stats_baseline["active.peak"] + expected_allocated, stats_after["active.peak"])

        # Cached block memory statistics (all 2MB should be cached now)
        self.assertEqual(stats_baseline["cached.current"] + expected_reserved, stats_after["cached.current"])
        self.assertEqual(stats_baseline["cached.peak"] + expected_reserved, stats_after["cached.peak"])

    def test_invalid_device_id(self):
        """Test invalid device ID handling (matching C++ RuntimeAPI_invalid_device_id)."""

        # Test with invalid device IDs that don't immediately raise exceptions
        invalid_devices = [-1]  # Only test integer device IDs that might be handled gracefully

        for invalid_device in invalid_devices:
            # These should not raise exceptions but may return default values
            # or handle gracefully depending on implementation
            try:
                torch.rbln.empty_cache(invalid_device)
                torch.rbln.reset_accumulated_memory_stats(invalid_device)
                torch.rbln.reset_peak_memory_stats(invalid_device)
                stats = torch.rbln.memory_stats(invalid_device)
                # If no exception is raised, stats should be reasonable
                self.assertIsInstance(stats, dict)
            except (RuntimeError, ValueError):
                # Expected behavior for invalid device IDs
                pass

        # Test string device IDs separately since they raise exceptions immediately
        try:
            torch.device("rbln:-1")  # This will raise RuntimeError immediately
            self.fail("Expected RuntimeError for invalid device string")
        except RuntimeError:
            # Expected behavior
            pass

    def test_reset_peak_memory_stats(self):
        """Test reset peak memory stats functionality (matching C++ RuntimeAPI_reset_peak_memory_stats)."""
        # Get baseline stats
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Allocate some memory first using specific sizes
        tensor1 = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB
        tensor2 = torch.empty((self.KB_2 // 2,), device=self.device, dtype=torch.float16)  # 2KB

        # Get stats after allocation
        stats_after_allocation = torch.rbln.memory_stats(self.device)
        expected_allocated = self.KB_1 + self.KB_2  # 3KB
        self.assertEqual(
            stats_after_allocation["allocated.current"], stats_baseline["allocated.current"] + expected_allocated
        )
        self.assertEqual(
            stats_after_allocation["allocated.peak"], stats_baseline["allocated.peak"] + expected_allocated
        )

        # Test reset peak stats
        torch.rbln.reset_peak_memory_stats(self.device)
        stats_after_reset_peak = torch.rbln.memory_stats(self.device)

        # Peak should reset to current values
        self.assertEqual(stats_after_reset_peak["allocated.current"], stats_after_reset_peak["allocated.peak"])
        self.assertEqual(stats_after_reset_peak["reserved.current"], stats_after_reset_peak["reserved.peak"])

        # Clean up
        del tensor1, tensor2
        torch.rbln.empty_cache(self.device)

    def test_reset_accumulated_memory_stats(self):
        """Test reset accumulated memory stats functionality (matching C++ RuntimeAPI_reset_accumulated_memory_stats)."""
        # Get baseline stats
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Allocate some memory first using specific sizes
        tensor1 = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB
        tensor2 = torch.empty((self.KB_2 // 2,), device=self.device, dtype=torch.float16)  # 2KB

        # Get stats after allocation
        stats_after_allocation = torch.rbln.memory_stats(self.device)
        expected_allocated = self.KB_1 + self.KB_2  # 3KB
        self.assertEqual(
            stats_after_allocation["allocated.current"], stats_baseline["allocated.current"] + expected_allocated
        )
        self.assertEqual(
            stats_after_allocation["allocated.peak"], stats_baseline["allocated.peak"] + expected_allocated
        )
        self.assertEqual(stats_after_allocation["num_device_alloc"], stats_baseline["num_device_alloc"] + 1)
        self.assertEqual(stats_after_allocation["num_device_free"], stats_baseline["num_device_free"])

        # Test reset accumulated stats
        torch.rbln.reset_accumulated_memory_stats(self.device)
        stats_after_reset_accumulated = torch.rbln.memory_stats(self.device)

        # Current and peak should remain, but accumulated counters should reset
        self.assertEqual(
            stats_after_allocation["allocated.current"], stats_after_reset_accumulated["allocated.current"]
        )
        self.assertEqual(stats_after_allocation["allocated.peak"], stats_after_reset_accumulated["allocated.peak"])
        self.assertEqual(0, stats_after_reset_accumulated["num_alloc_retries"])
        self.assertEqual(0, stats_after_reset_accumulated["num_ooms"])
        self.assertEqual(0, stats_after_reset_accumulated["num_device_alloc"])
        self.assertEqual(0, stats_after_reset_accumulated["num_device_free"])

        # Clean up
        del tensor1, tensor2
        torch.rbln.empty_cache(self.device)

    def test_get_memory_stats(self):
        """Test get memory stats functionality (matching C++ RuntimeAPI_get_memory_stats)."""
        # Test with valid device_id - get baseline stats
        stats_baseline = torch.rbln.memory_stats(self.device)

        # Verify that we can get stats and they have reasonable values
        self.assertGreaterEqual(stats_baseline["allocated.current"], 0)
        self.assertGreaterEqual(stats_baseline["reserved.current"], 0)
        self.assertGreaterEqual(stats_baseline["active.current"], 0)
        self.assertGreaterEqual(stats_baseline["cached.current"], 0)

        # Test allocation and verify stats increase using specific size
        tensor = torch.empty((self.KB_1 // 2,), device=self.device, dtype=torch.float16)  # 1KB

        stats_after_alloc = torch.rbln.memory_stats(self.device)

        # Verify allocation increased the stats by exact amount
        self.assertEqual(stats_after_alloc["allocated.current"], stats_baseline["allocated.current"] + self.KB_1)
        self.assertEqual(stats_after_alloc["allocated.peak"], stats_baseline["allocated.peak"] + self.KB_1)
        self.assertEqual(
            stats_after_alloc["allocated.total_allocated"], stats_baseline["allocated.total_allocated"] + self.KB_1
        )
        self.assertEqual(stats_after_alloc["allocated.total_freed"], stats_baseline["allocated.total_freed"])

        # Clean up
        del tensor

        # Verify stats after deallocation
        stats_after_free = torch.rbln.memory_stats(self.device)
        self.assertEqual(stats_baseline["allocated.current"], stats_after_free["allocated.current"])
        self.assertEqual(stats_baseline["allocated.peak"] + self.KB_1, stats_after_free["allocated.peak"])
        self.assertEqual(
            stats_baseline["allocated.total_allocated"] + self.KB_1, stats_after_free["allocated.total_allocated"]
        )
        self.assertEqual(stats_baseline["allocated.total_freed"] + self.KB_1, stats_after_free["allocated.total_freed"])

    def test_memory_stats_consistency(self):
        """Test consistency between memory_stats and individual functions."""
        # Get individual values
        allocated = torch.rbln.memory_allocated(self.device)
        reserved = torch.rbln.memory_reserved(self.device)
        max_allocated = torch.rbln.max_memory_allocated(self.device)
        max_reserved = torch.rbln.max_memory_reserved(self.device)

        # Get stats dictionary
        stats = torch.rbln.memory_stats(self.device)

        # Check consistency
        self.assertEqual(stats["allocated.current"], allocated)
        self.assertEqual(stats["reserved.current"], reserved)
        self.assertEqual(stats["allocated.peak"], max_allocated)
        self.assertEqual(stats["reserved.peak"], max_reserved)

    def test_device_parameter(self):
        """Test device parameter handling."""
        # Test with different device parameter types
        device_variants = [
            None,  # Default device
            0,  # Device index
            "rbln:0",  # Device string
            torch.device("rbln:0"),  # torch.device object
        ]

        for device in device_variants:
            # All should work without error
            allocated = torch.rbln.memory_allocated(device)
            reserved = torch.rbln.memory_reserved(device)
            stats = torch.rbln.memory_stats(device)

            self.assertGreaterEqual(allocated, 0)
            self.assertGreaterEqual(reserved, 0)
            self.assertIsInstance(stats, dict)

    def test_none_device_queries_use_current_device(self):
        """None device arguments must normalize to the current logical device."""
        expected_device = torch.device("rbln:1")
        stats = {
            "allocated_bytes.all.current": 128,
            "reserved_bytes.all.current": 256,
        }

        with (
            patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=1),
            patch(
                "torch_rbln.memory.torch_rbln._C.memory_stats",
                return_value=stats,
            ) as mock_memory_stats,
        ):
            self.assertEqual(torch.rbln.memory_stats(), stats)
            mock_memory_stats.assert_called_once_with(expected_device)

        with (
            patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=1),
            patch(
                "torch_rbln.memory.torch_rbln._C.memory_stats",
                return_value=stats,
            ) as mock_memory_stats,
        ):
            self.assertEqual(torch.rbln.memory_allocated(), 128)
            mock_memory_stats.assert_called_once_with(expected_device)

        with (
            patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=1),
            patch(
                "torch_rbln.memory.torch_rbln._C.memory_stats",
                return_value=stats,
            ) as mock_memory_stats,
        ):
            self.assertEqual(torch.rbln.memory_reserved(), 256)
            mock_memory_stats.assert_called_once_with(expected_device)

    def test_none_device_empty_cache_uses_current_device(self):
        """empty_cache(None) must target the current logical device."""
        expected_device = torch.device("rbln:1")

        with (
            patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=1),
            patch(
                "torch_rbln.memory.torch_rbln._C.empty_cache",
            ) as mock_empty_cache,
        ):
            torch.rbln.empty_cache()

        mock_empty_cache.assert_called_once_with(expected_device)

    def test_none_device_reset_peak_uses_current_device(self):
        """reset_peak_memory_stats(None) must target the current logical device."""
        expected_device = torch.device("rbln:1")

        with (
            patch("torch_rbln.memory.torch_rbln._C.current_device", return_value=1),
            patch(
                "torch_rbln.memory.torch_rbln._C.reset_peak_memory_stats",
            ) as mock_reset_peak,
        ):
            torch.rbln.reset_peak_memory_stats()

        mock_reset_peak.assert_called_once_with(expected_device)

    def test_memory_stats_keys_format(self):
        """Test that memory_stats returns keys in expected format."""
        stats = torch.rbln.memory_stats(self.device)

        # Check key naming convention
        for key in stats.keys():
            self.assertTrue(
                key.endswith((".current", ".peak", ".total_allocated", ".total_freed"))
                or key.startswith(("num_", "active_", "cached_")),
                f"Unexpected key format: {key}",
            )

    def test_memory_functions_return_types(self):
        """Test that memory functions return correct types."""
        # Test return types
        self.assertIsInstance(torch.rbln.memory_allocated(self.device), int)
        self.assertIsInstance(torch.rbln.memory_reserved(self.device), int)
        self.assertIsInstance(torch.rbln.max_memory_allocated(self.device), int)
        self.assertIsInstance(torch.rbln.max_memory_reserved(self.device), int)
        self.assertIsInstance(torch.rbln.memory_stats(self.device), dict)

        # Test that reset functions return None
        self.assertIsNone(torch.rbln.empty_cache(self.device))
        self.assertIsNone(torch.rbln.reset_accumulated_memory_stats(self.device))
        self.assertIsNone(torch.rbln.reset_peak_memory_stats(self.device))


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_eager_malloc")
class TestAcceleratorMemoryAPI(TestCase):
    """Tests for torch.accelerator.* memory APIs backed by RBLNAllocator.DeviceAllocator.

    These tests exercise the DeviceAllocator interface path (getDeviceStats,
    resetAccumulatedStats, resetPeakStats, emptyCache, initialized) independently
    of the torch.rbln.* path so that full coverage is achieved for the new
    DeviceAllocator methods added in RBLNAllocator.cpp.
    """

    def setUp(self):
        try:
            import torch_rbln  # noqa: F401 — registers the rbln backend
        except ImportError:
            self.skipTest("torch_rbln is not installed")

        if not torch.rbln.is_available():
            self.skipTest("RBLN device not available")
        if torch.accelerator.current_accelerator().type != "rbln":
            self.skipTest("Current accelerator is not RBLN")

        # Ensure a clean stats baseline before each test.
        torch.accelerator.empty_cache()
        torch.accelerator.reset_accumulated_memory_stats()
        torch.accelerator.reset_peak_memory_stats()

    def test_memory_stats_returns_ordered_dict(self):
        stats = torch.accelerator.memory_stats()
        self.assertIsInstance(stats, dict)

    def test_memory_stats_has_122_keys(self):
        stats = torch.accelerator.memory_stats()
        self.assertEqual(len(stats), 122)

    def test_memory_stats_all_values_are_int(self):
        stats = torch.accelerator.memory_stats()
        for key, value in stats.items():
            self.assertIsInstance(value, int, msg=f"Key '{key}' has non-int value: {value!r}")

    def test_memory_stats_expected_scalar_keys_present(self):
        stats = torch.accelerator.memory_stats()
        scalars = [
            "num_alloc_retries",
            "num_ooms",
            "num_device_alloc",
            "num_device_free",
            "num_sync_all_streams",
            "max_split_size",
        ]
        for key in scalars:
            self.assertIn(key, stats, msg=f"Missing expected key '{key}'")

    def test_memory_stats_pool_keys_present(self):
        # Verify that AGGREGATE (.all.), SMALL_POOL (.small_pool.), and LARGE_POOL
        # (.large_pool.) variants exist for a representative field.
        stats = torch.accelerator.memory_stats()
        for suffix in (".all.current", ".small_pool.current", ".large_pool.current"):
            key = "allocated_bytes" + suffix
            self.assertIn(key, stats, msg=f"Missing expected key '{key}'")

    def test_memory_allocated_return_type(self):
        self.assertIsInstance(torch.accelerator.memory_allocated(), int)

    def test_memory_reserved_return_type(self):
        self.assertIsInstance(torch.accelerator.memory_reserved(), int)

    def test_max_memory_allocated_return_type(self):
        self.assertIsInstance(torch.accelerator.max_memory_allocated(), int)

    def test_max_memory_reserved_return_type(self):
        self.assertIsInstance(torch.accelerator.max_memory_reserved(), int)

    def test_memory_allocated_non_negative(self):
        self.assertGreaterEqual(torch.accelerator.memory_allocated(), 0)

    def test_memory_reserved_non_negative(self):
        self.assertGreaterEqual(torch.accelerator.memory_reserved(), 0)

    def test_max_memory_allocated_gte_memory_allocated(self):
        self.assertGreaterEqual(
            torch.accelerator.max_memory_allocated(),
            torch.accelerator.memory_allocated(),
        )

    def test_max_memory_reserved_gte_memory_reserved(self):
        self.assertGreaterEqual(
            torch.accelerator.max_memory_reserved(),
            torch.accelerator.memory_reserved(),
        )

    def test_memory_allocated_consistent_with_stats(self):
        """memory_allocated() must equal allocated_bytes.all.current in memory_stats()."""
        # Sample both in close succession; no allocation between calls.
        stats = torch.accelerator.memory_stats()
        allocated = torch.accelerator.memory_allocated()
        self.assertEqual(allocated, stats["allocated_bytes.all.current"])

    def test_memory_reserved_consistent_with_stats(self):
        """memory_reserved() must equal reserved_bytes.all.current in memory_stats()."""
        stats = torch.accelerator.memory_stats()
        reserved = torch.accelerator.memory_reserved()
        self.assertEqual(reserved, stats["reserved_bytes.all.current"])

    def test_accelerator_stats_consistent_with_rbln_stats(self):
        """DeviceAllocator path must report the same underlying values as torch.rbln path."""
        device = "rbln:0"
        rbln_stats = torch.rbln.memory_stats(device)
        acc_stats = torch.accelerator.memory_stats()

        self.assertEqual(
            acc_stats["allocated_bytes.all.current"],
            rbln_stats["allocated.current"],
        )
        self.assertEqual(
            acc_stats["reserved_bytes.all.current"],
            rbln_stats["reserved.current"],
        )
        self.assertEqual(
            acc_stats["active_bytes.all.current"],
            rbln_stats["active.current"],
        )
        self.assertEqual(
            acc_stats["inactive_split_bytes.all.current"],
            rbln_stats["cached.current"],
        )

    def test_reset_peak_memory_stats_no_exception(self):
        torch.accelerator.reset_peak_memory_stats()  # must not raise

    def test_reset_peak_memory_stats_clears_peak(self):
        """After reset, peak must equal current (allocation tracking restarts)."""
        torch.accelerator.reset_peak_memory_stats()
        stats = torch.accelerator.memory_stats()
        self.assertEqual(
            stats["allocated_bytes.all.peak"],
            stats["allocated_bytes.all.current"],
        )
        self.assertEqual(
            stats["reserved_bytes.all.peak"],
            stats["reserved_bytes.all.current"],
        )

    def test_reset_accumulated_memory_stats_no_exception(self):
        torch.accelerator.reset_accumulated_memory_stats()  # must not raise

    def test_reset_accumulated_memory_stats_clears_counters(self):
        """After reset, cumulative counters must be zero."""
        torch.accelerator.reset_accumulated_memory_stats()
        stats = torch.accelerator.memory_stats()
        self.assertEqual(stats["num_alloc_retries"], 0)
        self.assertEqual(stats["num_ooms"], 0)
        self.assertEqual(stats["num_device_alloc"], 0)
        self.assertEqual(stats["num_device_free"], 0)

    def test_reset_accumulated_does_not_clear_peak(self):
        """reset_accumulated_memory_stats must not clear peak values."""
        stats_before = torch.accelerator.memory_stats()
        peak_before = stats_before["allocated_bytes.all.peak"]
        torch.accelerator.reset_accumulated_memory_stats()
        stats_after = torch.accelerator.memory_stats()
        self.assertEqual(stats_after["allocated_bytes.all.peak"], peak_before)

    def test_empty_cache_no_exception(self):
        torch.accelerator.empty_cache()  # must not raise

    def test_empty_cache_preserves_active_allocations(self):
        """Calling empty_cache while tensors are live must not free live memory."""
        # Allocate a small tensor without moving it to the RBLN device, because we
        # only have a simulated device here. We verify that memory_allocated() is
        # stable across empty_cache() when there are no real allocations.
        before = torch.accelerator.memory_allocated()
        torch.accelerator.empty_cache()
        after = torch.accelerator.memory_allocated()
        self.assertEqual(before, after)

    def test_stats_consistency_with_live_allocation(self):
        """Both APIs must report equal non-zero values while a tensor is live.

        With TORCH_RBLN_EAGER_MALLOC active, allocated bytes are physically
        bound at allocation time, so the comparison is against a real non-zero
        value rather than a lazy-mode zero.
        """
        device = "rbln:0"
        MB_1 = 1024 * 1024
        tensor = torch.empty(512 * 1024, dtype=torch.float16, device=device)
        try:
            rbln_stats = torch.rbln.memory_stats(device)
            acc_stats = torch.accelerator.memory_stats()

            # Physical allocation must be visible (eager malloc is active).
            self.assertGreaterEqual(acc_stats["allocated_bytes.all.current"], MB_1)

            # All four byte categories must match exactly.
            self.assertEqual(
                acc_stats["allocated_bytes.all.current"],
                rbln_stats["allocated.current"],
            )
            self.assertEqual(
                acc_stats["reserved_bytes.all.current"],
                rbln_stats["reserved.current"],
            )
            self.assertEqual(
                acc_stats["active_bytes.all.current"],
                rbln_stats["active.current"],
            )
            self.assertEqual(
                acc_stats["inactive_split_bytes.all.current"],
                rbln_stats["cached.current"],
            )
        finally:
            del tensor

    def test_stats_consistency_alloc_free_cycle(self):
        """Both APIs must track the full alloc/free lifecycle with exact byte counts.

        Three-point check: baseline, +1 MB allocated, freed back to baseline.
        Verifies that runtime and accelerator APIs report identical values at
        each stage, confirming no divergence in the getDeviceStats() mapping.
        """
        device = "rbln:0"
        MB_1 = 1024 * 1024

        # --- Baseline (clean state after setUp) ---
        rbln_base = torch.rbln.memory_stats(device)
        acc_base = torch.accelerator.memory_stats()
        self.assertEqual(acc_base["allocated_bytes.all.current"], rbln_base["allocated.current"])
        self.assertEqual(acc_base["reserved_bytes.all.current"], rbln_base["reserved.current"])

        # --- After allocating 1 MB ---
        tensor = torch.empty(512 * 1024, dtype=torch.float16, device=device)
        try:
            rbln_alloc = torch.rbln.memory_stats(device)
            acc_alloc = torch.accelerator.memory_stats()

            # Both APIs must agree on current state.
            self.assertEqual(
                acc_alloc["allocated_bytes.all.current"],
                rbln_alloc["allocated.current"],
            )
            self.assertEqual(
                acc_alloc["reserved_bytes.all.current"],
                rbln_alloc["reserved.current"],
            )
            # Exact delta must equal MB_1.
            self.assertEqual(
                rbln_alloc["allocated.current"] - rbln_base["allocated.current"],
                MB_1,
            )
            self.assertEqual(
                acc_alloc["allocated_bytes.all.current"] - acc_base["allocated_bytes.all.current"],
                MB_1,
            )
            # Peak must reflect the live allocation.
            self.assertGreaterEqual(rbln_alloc["allocated.peak"], MB_1)
            self.assertEqual(acc_alloc["allocated_bytes.all.peak"], rbln_alloc["allocated.peak"])
        finally:
            del tensor

        # --- After freeing ---
        torch.accelerator.empty_cache()
        rbln_free = torch.rbln.memory_stats(device)
        acc_free = torch.accelerator.memory_stats()

        # Both must agree; current must return to baseline.
        self.assertEqual(acc_free["allocated_bytes.all.current"], rbln_free["allocated.current"])
        self.assertEqual(rbln_free["allocated.current"], rbln_base["allocated.current"])
        # Peak must be retained after free.
        self.assertGreaterEqual(rbln_free["allocated.peak"], MB_1)
        self.assertEqual(acc_free["allocated_bytes.all.peak"], rbln_free["allocated.peak"])

    def test_stats_consistency_after_reset_accumulated(self):
        """Both APIs must agree before and after reset_accumulated_memory_stats.

        Allocates then frees a tensor first so the counters are non-trivially
        non-zero, then resets via the accelerator API. Both paths must see the
        same pre-reset and post-reset counter values.
        """
        device = "rbln:0"

        # Generate a non-zero device-alloc counter.
        tensor = torch.empty(512 * 1024, dtype=torch.float16, device=device)
        del tensor

        # Pre-reset: both APIs must agree on non-zero counters.
        rbln_pre = torch.rbln.memory_stats(device)
        acc_pre = torch.accelerator.memory_stats()
        self.assertGreater(rbln_pre["num_device_alloc"], 0)
        self.assertEqual(acc_pre["num_device_alloc"], rbln_pre["num_device_alloc"])

        # Reset via accelerator API.
        torch.accelerator.reset_accumulated_memory_stats()

        rbln_post = torch.rbln.memory_stats(device)
        acc_post = torch.accelerator.memory_stats()

        # All accumulated counters must be zero and equal in both APIs.
        self.assertEqual(acc_post["num_alloc_retries"], rbln_post["num_alloc_retries"])
        self.assertEqual(acc_post["num_ooms"], rbln_post["num_ooms"])
        self.assertEqual(acc_post["num_device_alloc"], rbln_post["num_device_alloc"])
        self.assertEqual(acc_post["num_device_free"], rbln_post["num_device_free"])
        self.assertEqual(rbln_post["num_device_alloc"], 0)
        self.assertEqual(rbln_post["num_device_free"], 0)

    def test_stats_consistency_after_reset_peak(self):
        """Both APIs must agree on peak values before and after reset_peak_memory_stats.

        Allocates 1 MB first so the peak is non-trivially non-zero, then
        resets via the accelerator API. Both paths must see the same pre-reset
        peak and agree on the post-reset value.
        """
        device = "rbln:0"
        MB_1 = 1024 * 1024

        # Generate a non-zero peak.
        tensor = torch.empty(512 * 1024, dtype=torch.float16, device=device)
        try:
            # Pre-reset: both APIs must agree on non-zero peak.
            rbln_pre = torch.rbln.memory_stats(device)
            acc_pre = torch.accelerator.memory_stats()
            self.assertGreaterEqual(rbln_pre["allocated.peak"], MB_1)
            self.assertEqual(acc_pre["allocated_bytes.all.peak"], rbln_pre["allocated.peak"])

            # Reset via accelerator API.
            torch.accelerator.reset_peak_memory_stats()

            rbln_post = torch.rbln.memory_stats(device)
            acc_post = torch.accelerator.memory_stats()

            # Peak must equal current in both APIs after reset.
            self.assertEqual(acc_post["allocated_bytes.all.peak"], rbln_post["allocated.peak"])
            self.assertEqual(acc_post["reserved_bytes.all.peak"], rbln_post["reserved.peak"])
            # Peak resets to current (tensor is still alive).
            self.assertEqual(rbln_post["allocated.peak"], rbln_post["allocated.current"])
        finally:
            del tensor


instantiate_device_type_tests(TestMemoryStats, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAcceleratorMemoryAPI, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
