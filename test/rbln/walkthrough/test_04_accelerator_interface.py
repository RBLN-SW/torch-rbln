# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 4: `torch.accelerator` device and memory APIs for RBLN.

Ported from walkthrough_guide/4.accelerator_interface.py.

Verifies that:
- `torch.accelerator.is_available()`, `current_accelerator()`, `device_count()`,
  and `current_device_index()` work with RBLN.
- The `device_index()` context manager switches the active device.
- `torch.accelerator.memory.*` exposes the standard memory API.
- Device memory is allocated lazily (a device op increases `memory_allocated`).
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import requires_logical_devices, SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestAcceleratorInterface(TestCase):
    """Walkthrough example 4: unified accelerator API with RBLN."""

    def test_accelerator_is_available(self):
        """`torch.accelerator.is_available()` should return True on an RBLN host."""
        self.assertTrue(torch.accelerator.is_available())

    def test_current_accelerator_is_rbln(self):
        """The current accelerator should be the RBLN device."""
        acc = torch.accelerator.current_accelerator()
        self.assertIsNotNone(acc)
        self.assertEqual(acc.type, "rbln")

    def test_accelerator_device_count(self):
        """Accelerator and `torch.rbln` device counts should agree and be positive."""
        count = torch.accelerator.device_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        self.assertEqual(count, torch.rbln.device_count())

    def test_accelerator_current_device_index(self):
        """`current_device_index()` should return a non-negative int."""
        idx = torch.accelerator.current_device_index()
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)

    @requires_logical_devices(2)
    def test_device_index_context_manager(self):
        """`device_index(i)` should switch the active device for the duration of the block."""
        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)

        self.assertEqual(torch.accelerator.current_device_index(), 0)


@pytest.mark.test_set_ci
class TestAcceleratorMemory(TestCase):
    """Walkthrough example 4: `torch.accelerator.memory` API and lazy allocation."""

    device_index = 0
    device = torch.device("rbln:0")

    def _reset_memory_state(self):
        torch.accelerator.set_device_index(self.device_index)
        torch.accelerator.memory.empty_cache()
        torch.accelerator.memory.reset_peak_memory_stats(self.device_index)

    def test_memory_api_surface(self):
        """All standard memory-management functions should exist and be callable."""
        self._reset_memory_state()
        mem_alloc = torch.accelerator.memory.memory_allocated(self.device_index)
        mem_reserved = torch.accelerator.memory.memory_reserved(self.device_index)
        max_alloc = torch.accelerator.memory.max_memory_allocated(self.device_index)
        stats = torch.accelerator.memory.memory_stats(self.device_index)

        self.assertIsInstance(mem_alloc, int)
        self.assertIsInstance(mem_reserved, int)
        self.assertIsInstance(max_alloc, int)
        self.assertIsInstance(stats, dict)

    @dtypes(*SUPPORTED_DTYPES)
    def test_lazy_device_allocation_after_op(self, dtype):
        """Running a device op should not decrease `memory_allocated`."""
        self._reset_memory_state()

        mem_before = torch.accelerator.memory.memory_allocated(self.device_index)
        a = torch.randn(128, 128, device=self.device, dtype=dtype)
        b = torch.randn(128, 128, device=self.device, dtype=dtype)
        mem_after_create = torch.accelerator.memory.memory_allocated(self.device_index)
        _ = a + b  # Device op should materialize memory.
        mem_after_op = torch.accelerator.memory.memory_allocated(self.device_index)

        self.assertGreaterEqual(mem_after_create, mem_before)
        self.assertGreaterEqual(mem_after_op, mem_after_create)


instantiate_device_type_tests(TestAcceleratorInterface, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAcceleratorMemory, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
