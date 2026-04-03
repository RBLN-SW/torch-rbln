# Owner(s): ["module: PrivateUse1"]

"""
Test suite for RBLN Python APIs.

This test suite covers the following RBLN Python APIs:
1. Device management APIs: device_count, is_available, current_device, set_device
2. Tensor utilities: _create_tensor_from_ptr
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestDeviceManagementAPIs(TestCase):
    """Test RBLN device management APIs such as device_count, is_available, current_device, and set_device."""

    def test_device_management_api_existence(self):
        self.assertTrue(hasattr(torch.rbln, "device_count"))
        self.assertTrue(callable(torch.rbln.device_count))
        self.assertTrue(hasattr(torch.rbln, "is_available"))
        self.assertTrue(callable(torch.rbln.is_available))
        self.assertTrue(hasattr(torch.rbln, "current_device"))
        self.assertTrue(callable(torch.rbln.current_device))
        self.assertTrue(hasattr(torch.rbln, "set_device"))
        self.assertTrue(callable(torch.rbln.set_device))

    def test_device_count(self):
        count = torch.rbln.device_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_is_available(self):
        available = torch.rbln.is_available()
        self.assertIsInstance(available, bool)
        self.assertTrue(available)

    def test_current_device(self):
        device = torch.rbln.current_device()
        self.assertIsInstance(device, int)
        self.assertGreaterEqual(device, 0)


@pytest.mark.test_set_ci
class TestTensorUtils(TestCase):
    """Test RBLN tensor utilities such as _create_tensor_from_ptr."""

    @dtypes(*SUPPORTED_DTYPES)
    def test_create_tensor_from_ptr(self, dtype):
        for device_index in range(torch.rbln.device_count()):
            device = torch.device("rbln", device_index)

            # Create source tensor on RBLN device
            x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
            original_data_ptr = x.data_ptr()

            # Create tensor from data_ptr
            new_x = torch.rbln._create_tensor_from_ptr(original_data_ptr, x.shape, x.dtype)

            # Verify properties
            self.assertEqual(new_x.data_ptr(), x.data_ptr())
            self.assertEqual(new_x.device, x.device)
            self.assertEqual(new_x.dtype, x.dtype)
            self.assertEqual(new_x.shape, x.shape)

            # Verify data is shared (modify one and check the other)
            x[0] = 42.0
            self.assertAlmostEqual(new_x[0].item(), 42.0, places=5)
            self.assertAlmostEqual(new_x[0].item(), x[0].item(), places=5)


instantiate_device_type_tests(TestDeviceManagementAPIs, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTensorUtils, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
