# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 1: PyTorch recognizes RBLN as a device type.

Ported from walkthrough_guide/1.rbln_device_type.py.

Verifies that:
- `torch.device("rbln")` is accepted and reports type "rbln".
- `torch.rbln.device_count()`, `current_device()`, and `set_device()` work.
- Indexed devices such as "rbln:0" are accepted.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import requires_logical_devices


@pytest.mark.test_set_ci
class TestRBLNDeviceType(TestCase):
    """Walkthrough example 1: RBLN as a first-class PyTorch device type."""

    def test_device_is_rbln(self):
        """`torch.device('rbln')` should yield a device whose type is 'rbln'."""
        device = torch.device("rbln")
        self.assertEqual(device.type, "rbln")

    def test_device_count_positive(self):
        """device_count() should return a positive int when RBLN is available."""
        count = torch.rbln.device_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_set_and_current_device_on_zero(self):
        """set_device(0) should make current_device() return 0."""
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)

    def test_indexed_device(self):
        """Indexed devices like 'rbln:0' should round-trip through torch.device."""
        device = torch.device("rbln:0")
        self.assertEqual(device.type, "rbln")
        self.assertEqual(device.index, 0)

    @requires_logical_devices(2)
    def test_switch_between_devices(self):
        """With at least two logical devices, set_device(1) should be observable."""
        try:
            torch.rbln.set_device(1)
            self.assertEqual(torch.rbln.current_device(), 1)
        finally:
            # Restore for downstream tests.
            torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)


instantiate_device_type_tests(TestRBLNDeviceType, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
