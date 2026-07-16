# Owner(s): ["module: PrivateUse1"]

"""Regression tests for torch.accelerator / torch.rbln contract gaps.

Covers:
- ``torch.rbln.memory_*`` reading the RBLN allocator's dotted stat keys
  (``allocated.current`` etc.), not the CUDA-style ``allocated_bytes.all.current``.
- Device-argument normalization (accept None/int/str/torch.device, reject
  non-rbln devices and out-of-range indices).
- ``torch.accelerator.get_device_capability()`` advertising the dtypes RBLN can
  allocate and type-convert on device (allocation + conversion, not native-op
  dispatch).
- PrivateUse1 storage resize-to-zero.
- ``torch.accelerator.empty_cache()`` (no device arg) spanning every
  initialized device.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401 -- registers the rbln device + torch.rbln namespace


class TestRblnMemoryHelpers(TestCase):
    """``torch.rbln.memory_*`` must read the RBLN allocator's dotted stat keys."""

    @pytest.mark.test_set_ci
    def test_memory_helpers_match_stats_and_are_nonzero(self):
        torch.rbln.empty_cache()
        torch.rbln.reset_peak_memory_stats()

        keep = torch.randn(1024, 1024, device="rbln:0", dtype=torch.float16)
        _ = keep + keep

        stats = torch.rbln.memory_stats()
        self.assertEqual(torch.rbln.memory_allocated(), stats.get("allocated.current", 0))
        self.assertEqual(torch.rbln.max_memory_allocated(), stats.get("allocated.peak", 0))
        self.assertEqual(torch.rbln.memory_reserved(), stats.get("reserved.current", 0))
        self.assertEqual(torch.rbln.max_memory_reserved(), stats.get("reserved.peak", 0))
        # Regression: the CUDA-style key made these return 0 despite a live allocation.
        self.assertGreater(torch.rbln.memory_allocated(), 0)
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


class TestDeviceCapability(TestCase):
    """``torch.accelerator.get_device_capability()`` reports allocation/conversion dtypes."""

    @pytest.mark.test_set_ci
    def test_supported_dtypes_are_allocatable_and_convertible(self):
        cap = torch.accelerator.get_device_capability()
        self.assertIn("supported_dtypes", cap)
        # Capability = dtypes RBLN can allocate and type-convert on device
        # (per the upstream contract this is independent of native-op dispatch),
        # matching the v2v/copy engine set (ENGINE_DTYPES in test/utils_v2v.py).
        self.assertEqual(
            set(cap["supported_dtypes"]),
            {torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64},
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
    """Device-less ``torch.accelerator.empty_cache()`` releases cached memory on
    every initialized device — not just the current one, and not a no-op."""

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

    @pytest.mark.test_set_ci
    def test_empty_cache_spans_all_initialized_devices(self):
        # The device-less form must walk *every* initialized device, so a
        # current-device-only regression is caught. Needs >= 2 usable devices
        # (multi-node hardware); degrades to skip on single-node hosts.
        if torch.rbln.device_count() < 2:
            self.skipTest("requires >= 2 rbln devices")
        indices = (0, 1)
        reserved_cached = {}
        for idx in indices:
            try:
                reserved_cached[idx] = self._hold_then_free_reserved(idx)
            except RuntimeError as exc:  # device not usable on this host
                self.skipTest(f"rbln:{idx} memory ops unavailable: {exc}")
        torch.accelerator.empty_cache()
        for idx in indices:
            self.assertLess(
                torch.rbln.memory_reserved(idx),
                reserved_cached[idx],
                msg=f"empty_cache() did not release device {idx} (current-device-only regression)",
            )


if __name__ == "__main__":
    run_tests()
