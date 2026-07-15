# Owner(s): ["module: PrivateUse1"]

"""Regression tests for torch.accelerator / torch.rbln contract gaps.

Covers:
- ``torch.rbln.memory_*`` reading the RBLN allocator's dotted stat keys
  (``allocated.current`` etc.), not the CUDA-style ``allocated_bytes.all.current``.
- Device-argument normalization (accept None/int/str/torch.device, reject
  non-rbln devices and out-of-range indices).
- ``torch.accelerator.get_device_capability()`` advertising the natively
  dispatched dtypes.
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
    """``torch.accelerator.get_device_capability()`` reports the native dtypes."""

    @pytest.mark.test_set_ci
    def test_supported_dtypes_are_native_floats(self):
        cap = torch.accelerator.get_device_capability()
        self.assertIn("supported_dtypes", cap)
        # RBLN dispatches fp16/bf16 natively; fallback-only dtypes are excluded.
        self.assertEqual(set(cap["supported_dtypes"]), {torch.float16, torch.bfloat16})


class TestStorageResizeToZero(TestCase):
    """PrivateUse1 storage must accept ``resize_(0)``."""

    @pytest.mark.test_set_ci
    def test_resize_storage_to_zero(self):
        x = torch.empty(16, device="rbln:0", dtype=torch.float16)
        storage = x.untyped_storage()
        storage.resize_(0)
        self.assertEqual(storage.nbytes(), 0)


class TestAcceleratorEmptyCache(TestCase):
    """Device-less ``torch.accelerator.empty_cache()`` iterates all initialized devices."""

    @pytest.mark.test_set_ci
    def test_generic_empty_cache_runs(self):
        # Materialize then free so the allocator holds cache, then call the
        # device-less generic empty_cache: it walks every initialized device and
        # must complete without error (with one device it iterates once).
        scratch = torch.randn(512, 512, device="rbln:0", dtype=torch.float16)
        _ = scratch + scratch
        del scratch
        torch.accelerator.empty_cache()
        self.assertGreaterEqual(torch.rbln.memory_reserved(), 0)


if __name__ == "__main__":
    run_tests()
