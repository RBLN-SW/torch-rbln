# Owner(s): ["module: PrivateUse1"]

"""
Test suite for multi-device functionality of RBLN backend.
"""

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import requires_logical_devices, SUPPORTED_DTYPES


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestMultiDevice(TestCase):
    @requires_logical_devices(2)
    def test_device_count(self):
        """Test that we have multiple devices available"""
        num_devices = torch.rbln.device_count()
        self.assertGreaterEqual(num_devices, 2, "Need at least 2 logical devices for multi-device tests")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_tensor_to_different_device(self, dtype):
        """Test moving tensors between devices using to()"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Create tensor on device 0
        x = torch.randn([4, 4], dtype=dtype, device=device0)
        self.assertEqual(x.device, device0)

        # Move to device 1
        y = x.to(device1)
        self.assertEqual(y.device, device1)
        self.assertNotEqual(x.device, y.device)

        # Verify data is preserved
        self.assertEqual(x.cpu(), y.cpu())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_tensor_to_same_device(self, dtype):
        """Test that to() on same device returns same tensor"""
        device0 = torch.device("rbln:0")
        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.to(device0)
        # Should be the same tensor (no copy)
        self.assertEqual(x.data_ptr(), y.data_ptr())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_copy_between_devices(self, dtype):
        """Test copy_() between different devices"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Create source on device 0
        src = torch.randn([4, 4], dtype=dtype, device=device0)
        src_data = src.cpu().clone()

        # Create destination on device 1
        dst = torch.zeros([4, 4], dtype=dtype, device=device1)

        # Copy from device 0 to device 1
        dst.copy_(src)

        # Verify copy succeeded
        self.assertEqual(dst.device, device1)
        self.assertEqual(dst.cpu(), src_data)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_copy_same_device(self, dtype):
        """Test copy_() on same device"""
        device0 = torch.device("rbln:0")
        src = torch.randn([4, 4], dtype=dtype, device=device0)
        dst = torch.zeros([4, 4], dtype=dtype, device=device0)
        src_data = src.cpu().clone()

        dst.copy_(src)
        self.assertEqual(dst.cpu(), src_data)

    @requires_logical_devices(2)
    def test_set_device(self):
        """Test set_device() public API"""
        # Set device 0
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)

        # Set device 1
        torch.rbln.set_device(1)
        self.assertEqual(torch.rbln.current_device(), 1)

        # Set device using torch.device
        torch.rbln.set_device(torch.device("rbln:0"))
        self.assertEqual(torch.rbln.current_device(), 0)

        # Set device using string
        torch.rbln.set_device("rbln:1")
        self.assertEqual(torch.rbln.current_device(), 1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_device_context_manager(self, dtype):
        """
        Test device context manager (torch.rbln.device()) for switching devices.
        This is the PyTorch-style way to switch devices, similar to torch.cuda.device().
        """
        device1 = torch.device("rbln:1")

        # Set initial device
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)

        # Test torch.rbln.device() context manager
        with torch.rbln.device(device1):
            # current_device() should be changed to device1
            self.assertEqual(torch.rbln.current_device(), 1)
            x = torch.randn([2, 2], dtype=dtype, device=device1)
            self.assertEqual(x.device.index, 1)

        # Device should be restored to original device after context
        self.assertEqual(torch.rbln.current_device(), 0)

        # Test with device index (int)
        with torch.rbln.device(1):
            self.assertEqual(torch.rbln.current_device(), 1)

        self.assertEqual(torch.rbln.current_device(), 0)

        # Test nested context managers
        with torch.rbln.device(0):
            self.assertEqual(torch.rbln.current_device(), 0)
            with torch.rbln.device(1):
                self.assertEqual(torch.rbln.current_device(), 1)
            # Should restore to device 0
            self.assertEqual(torch.rbln.current_device(), 0)
        # Should restore to original device 0
        self.assertEqual(torch.rbln.current_device(), 0)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_device_of(self, dtype):
        """Test device_of() context manager that uses tensor's device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Set initial device to 0
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)

        # Create tensor on device 1
        x = torch.randn([2, 2], dtype=dtype, device=device1)

        # Use device_of to switch to tensor's device
        with torch.rbln.device_of(x):
            self.assertEqual(torch.rbln.current_device(), 1)
            # Create another tensor, should be on device 1
            y = torch.randn([2, 2], dtype=dtype, device=device1)
            self.assertEqual(y.device.index, 1)

        # Device should be restored to original device after context
        self.assertEqual(torch.rbln.current_device(), 0)

        # Test with tensor on device 0
        z = torch.randn([2, 2], dtype=dtype, device=device0)
        with torch.rbln.device_of(z):
            self.assertEqual(torch.rbln.current_device(), 0)

        self.assertEqual(torch.rbln.current_device(), 0)

        # Test that non-RBLN tensor is a no-op
        cpu_tensor = torch.randn([2, 2], dtype=dtype, device="cpu")
        original_device = torch.rbln.current_device()
        with torch.rbln.device_of(cpu_tensor):
            # Should remain unchanged (no-op for non-RBLN devices)
            self.assertEqual(torch.rbln.current_device(), original_device)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_current_device_tracking(self, dtype):
        """Test that current device is tracked correctly"""
        num_devices = torch.rbln.device_count()
        for i in range(min(num_devices, 4)):  # Test up to 4 devices
            device = torch.device(f"rbln:{i}")
            torch.rbln.set_device(i)
            self.assertEqual(torch.rbln.current_device(), i)

            # Create tensor should be on current device
            x = torch.randn([2, 2], dtype=dtype, device=device)
            self.assertEqual(x.device.index, i)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_operations_preserve_device(self, dtype):
        """Test that operations preserve device placement"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Create tensors on different devices
        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = torch.randn([4, 4], dtype=dtype, device=device1)

        # Operations on same device should preserve device
        z = x + x  # Both on device0
        self.assertEqual(z.device, device0)

        w = y * y  # Both on device1
        self.assertEqual(w.device, device1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_cross_device_operations_error(self, dtype):
        """Test that cross-device operations raise appropriate errors"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = torch.randn([4, 4], dtype=dtype, device=device1)

        # Cross-device operations should fail (unless explicitly supported)
        with self.assertRaises(RuntimeError):
            _ = x + y  # Different devices

        with self.assertRaises(RuntimeError):
            _ = x * y  # Different devices

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_clone_preserves_device(self, dtype):
        """Test that clone() preserves device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.clone()
        self.assertEqual(y.device, device0)

        # Clone to different device
        z = x.clone().to(device1)
        self.assertEqual(z.device, device1)
        self.assertEqual(x.cpu(), z.cpu())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_empty_like_preserves_device(self, dtype):
        """Test that empty_like preserves device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = torch.empty_like(x)
        self.assertEqual(y.device, device0)

        x1 = torch.randn([4, 4], dtype=dtype, device=device1)
        y1 = torch.empty_like(x1)
        self.assertEqual(y1.device, device1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_zeros_ones_like_preserves_device(self, dtype):
        """Test that zeros_like and ones_like preserve device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = torch.zeros_like(x)
        z = torch.ones_like(x)
        self.assertEqual(y.device, device0)
        self.assertEqual(z.device, device0)

        x1 = torch.randn([4, 4], dtype=dtype, device=device1)
        y1 = torch.zeros_like(x1)
        z1 = torch.ones_like(x1)
        self.assertEqual(y1.device, device1)
        self.assertEqual(z1.device, device1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_multi_device_memory_isolation(self, dtype):
        """
        Test that memory is isolated between devices.
        This ensures that modifying a tensor on device 0 does not affect
        a tensor on device 1, confirming proper device memory separation.
        """
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Create simple tensors with explicit values on different devices
        # Device 0: values [1.0, 2.0, 3.0, ...]
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype, device=device0)
        x_initial = x.cpu().clone()

        # Device 1: different values [10.0, 20.0, 30.0, ...]
        y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=dtype, device=device1)
        y_initial = y.cpu().clone()

        # Modify x on device 0 (add 100 to all elements)
        x.add_(100.0)

        # Verify y on device 1 is completely unchanged (memory isolation)
        # This is the key test: device 1's memory should not be affected
        self.assertEqual(
            y.cpu(),
            y_initial,
            "Device 1 tensor must not change when device 0 tensor is modified",
            atol=1e-3,
            rtol=1e-3,
        )

        # Verify x on device 0 was actually modified
        x_expected = x_initial + 100.0
        self.assertEqual(x.cpu(), x_expected, "Device 0 tensor should be modified", atol=1e-3, rtol=1e-3)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_to_with_dtype_and_device(self, dtype):
        """Test to() with both dtype and device conversion"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.to(device=device1, dtype=dtype)

        self.assertEqual(y.device, device1)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(x.cpu(), y.cpu())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_non_blocking_copy(self, dtype):
        """Test non_blocking parameter in to()"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.to(device1, non_blocking=True)

        self.assertEqual(y.device, device1)
        self.assertEqual(x.cpu(), y.cpu())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_cuda_like_device_string(self, dtype):
        """Test that device strings like 'rbln:0' work like 'cuda:0'"""
        device0_str = "rbln:0"
        device1_str = "rbln:1"

        x = torch.randn([4, 4], dtype=dtype, device=device0_str)
        self.assertEqual(x.device.index, 0)

        y = x.to(device1_str)
        self.assertEqual(y.device.index, 1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_device_index_access(self, dtype):
        """Test accessing device index"""
        num_devices = torch.rbln.device_count()
        for device_index in range(min(num_devices, 4)):
            device = torch.device("rbln", device_index)
            x = torch.randn([2, 2], dtype=dtype, device=device)
            self.assertEqual(x.device, device)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_sequential_device_operations(self, dtype):
        """Test sequential operations across multiple devices"""
        devices = [f"rbln:{i}" for i in range(torch.rbln.device_count())]
        tensors = []
        for device_str in devices:
            x = torch.randn([3, 3], dtype=dtype, device=device_str)
            y = x * 2.0  # Operation on same device
            tensors.append(y)

        # Verify each tensor is on correct device
        for i, t in enumerate(tensors):
            self.assertEqual(t.device, torch.device(devices[i]))

        # Copy between devices
        if len(devices) >= 2:
            copied = tensors[0].to(devices[1])
            self.assertEqual(copied.device, torch.device(devices[1]))
            self.assertEqual(copied.cpu(), tensors[0].cpu())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_view_operations_preserve_device(self, dtype):
        """Test that view operations preserve device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.view(2, 8)
        self.assertEqual(y.device, device0)

        x1 = torch.randn([4, 4], dtype=dtype, device=device1)
        y1 = x1.reshape(16)
        self.assertEqual(y1.device, device1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_inplace_operations_preserve_device(self, dtype):
        """Test that in-place operations preserve device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        x.add_(1.0)
        self.assertEqual(x.device, device0)

        y = torch.randn([4, 4], dtype=dtype, device=device1)
        y.mul_(2.0)
        self.assertEqual(y.device, device1)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_slice_operations_preserve_device(self, dtype):
        """Test that slicing operations preserve device"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x[1:3, 1:3]
        self.assertEqual(y.device, device0)

        x1 = torch.randn([4, 4], dtype=dtype, device=device1)
        y1 = x1[:, :2]
        self.assertEqual(y1.device, device1)


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestDeviceOutputPlacement(TestCase):
    """Test cases to ensure output tensors are placed on the correct device."""

    def _assert_device_match(self, tensor, expected_device, msg=""):
        """Helper to assert tensor device matches expected device."""
        self.assertEqual(
            tensor.device.type,
            expected_device.type,
            f"{msg}Device type mismatch: expected {expected_device.type}, got {tensor.device.type}",
        )
        if expected_device.index is not None:
            self.assertEqual(
                tensor.device.index,
                expected_device.index,
                f"{msg}Device index mismatch: expected {expected_device.index}, got {tensor.device.index}",
            )

    # =========================================================================
    # Basic Linear Layer Tests
    # =========================================================================

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("device_index", [0, 1])
    def test_linear_single_device(self, dtype, device_index):
        """Test nn.Linear on specified device."""
        device = torch.device(f"rbln:{device_index}")
        linear = nn.Linear(64, 64).to(device=device, dtype=dtype)
        x = torch.randn([3, 64], dtype=dtype, device=device)

        output = linear(x)

        self._assert_device_match(output, device, f"Linear output on device {device_index}: ")
        self.assertEqual(output.shape, (3, 64))
        self.assertEqual(output.dtype, dtype)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_multiple_devices(self, dtype):
        """Test nn.Linear on multiple devices to catch device placement bugs."""
        num_devices = torch.rbln.device_count()
        for device_id in range(min(num_devices, 4)):  # Test up to 4 devices
            device = torch.device(f"rbln:{device_id}")
            linear = nn.Linear(32, 32).to(device=device, dtype=dtype)
            x = torch.randn([2, 32], dtype=dtype, device=device)

            output = linear(x)

            self._assert_device_match(output, device, f"Linear output on device {device_id}: ")
            self.assertEqual(output.shape, (2, 32))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_with_bias_device_1(self, dtype):
        """Test Linear layer with bias on device 1."""
        device = torch.device("rbln:1")
        linear = nn.Linear(32, 64, bias=True).to(device=device, dtype=dtype)
        x = torch.randn([4, 32], dtype=dtype, device=device)

        output = linear(x)
        self._assert_device_match(output, device, "Linear with bias output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_without_bias_device_1(self, dtype):
        """Test Linear layer without bias on device 1."""
        device = torch.device("rbln:1")
        linear = nn.Linear(32, 64, bias=False).to(device=device, dtype=dtype)
        x = torch.randn([4, 32], dtype=dtype, device=device)

        output = linear(x)
        self._assert_device_match(output, device, "Linear without bias output: ")

    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_default_device(self, dtype):
        """Test Linear on default device (rbln:0 or rbln)."""
        # Use default device (should be rbln:0 or rbln)
        linear = nn.Linear(32, 32).to(device="rbln", dtype=dtype)
        x = torch.randn([4, 32], dtype=dtype, device="rbln")

        output = linear(x)

        # Output should be on rbln device
        self.assertEqual(output.device.type, "rbln")
        # Should match input device
        self.assertEqual(output.device.index, x.device.index)

    # =========================================================================
    # Multi-Device and Context Tests
    # =========================================================================

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_sequential_devices(self, dtype):
        """Test sequential Linear operations on different devices."""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Create models on different devices
        linear0 = nn.Linear(16, 32).to(device=device0, dtype=dtype)
        linear1 = nn.Linear(32, 16).to(device=device1, dtype=dtype)

        # Input on device 0
        x = torch.randn([4, 16], dtype=dtype, device=device0)
        out0 = linear0(x)
        self._assert_device_match(out0, device0, "First Linear output: ")

        # Move to device 1 and apply second linear
        x1 = out0.to(device1)
        out1 = linear1(x1)
        self._assert_device_match(out1, device1, "Second Linear output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_linear_with_current_device_context(self, dtype):
        """Test that Linear respects device context even when current_device is different."""
        device1 = torch.device("rbln:1")

        # Set current device to 0
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.current_device(), 0)

        # Create model and input on device 1
        linear = nn.Linear(32, 32).to(device=device1, dtype=dtype)
        x = torch.randn([2, 32], dtype=dtype, device=device1)

        # Output should be on device 1, not current device (0)
        output = linear(x)
        self._assert_device_match(output, device1, "Linear should use input device, not current device: ")
        self.assertNotEqual(output.device.index, torch.rbln.current_device())

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_device_0_vs_device_1_comparison(self, dtype):
        """Test that operations on device 0 and device 1 produce outputs on correct devices."""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Same operation on different devices
        linear0 = nn.Linear(16, 16).to(device=device0, dtype=dtype)
        linear1 = nn.Linear(16, 16).to(device=device1, dtype=dtype)

        x0 = torch.randn([2, 16], dtype=dtype, device=device0)
        x1 = torch.randn([2, 16], dtype=dtype, device=device1)

        out0 = linear0(x0)
        out1 = linear1(x1)

        self._assert_device_match(out0, device0, "Device 0 output: ")
        self._assert_device_match(out1, device1, "Device 1 output: ")

        # Ensure they're on different devices
        self.assertNotEqual(out0.device.index, out1.device.index)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_device_switching_sequence(self, dtype):
        """Test switching between devices in a sequence of operations."""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        # Start on device 0
        x = torch.randn([4, 16], dtype=dtype, device=device0)
        linear0 = nn.Linear(16, 32).to(device=device0, dtype=dtype)
        out0 = linear0(x)
        self._assert_device_match(out0, device0, "First operation: ")

        # Switch to device 1
        x1 = out0.to(device1)
        linear1 = nn.Linear(32, 16).to(device=device1, dtype=dtype)
        out1 = linear1(x1)
        self._assert_device_match(out1, device1, "Second operation: ")

        # Switch back to device 0
        x2 = out1.to(device0)
        linear2 = nn.Linear(16, 8).to(device=device0, dtype=dtype)
        out2 = linear2(x2)
        self._assert_device_match(out2, device0, "Third operation: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_mixed_device_operations_error(self, dtype):
        """Test that mixing devices in operations raises appropriate errors."""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 8], dtype=dtype, device=device0)
        y = torch.randn([4, 8], dtype=dtype, device=device1)

        # Cross-device operations should fail
        with self.assertRaises(RuntimeError):
            _ = torch.add(x, y)

    # =========================================================================
    # Model Combination Tests
    # =========================================================================

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_multiple_operations_device_1(self, dtype):
        """Test multiple operations in sequence on device 1."""
        device = torch.device("rbln:1")
        x = torch.randn([4, 8], dtype=dtype, device=device)
        y = torch.randn([4, 8], dtype=dtype, device=device)

        # Various operations
        add_out = torch.add(x, y)
        self._assert_device_match(add_out, device, "Add output: ")

        mul_out = torch.mul(add_out, 2.0)
        self._assert_device_match(mul_out, device, "Mul output: ")

        linear = nn.Linear(8, 8).to(device=device, dtype=dtype)
        linear_out = linear(mul_out)
        self._assert_device_match(linear_out, device, "Linear output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_nested_model_device_1(self, dtype):
        """Test nested model with multiple layers on device 1."""
        device = torch.device("rbln:1")

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 32)
                self.linear2 = nn.Linear(32, 16)
                self.linear3 = nn.Linear(16, 8)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        model = NestedModel().to(device=device, dtype=dtype)
        x = torch.randn([4, 16], dtype=dtype, device=device)

        output = model(x)

        self._assert_device_match(output, device, "Nested model output: ")
        self.assertEqual(output.shape, (4, 8))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_sequential_model_device_1(self, dtype):
        """Test Sequential model device placement."""
        device = torch.device("rbln:1")
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
        ).to(device=device, dtype=dtype)

        x = torch.randn([4, 32], dtype=dtype, device=device)
        output = model(x)

        self._assert_device_match(output, device, "Sequential model output: ")
        self.assertEqual(output.shape, (4, 16))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_multiple_outputs_device_1(self, dtype):
        """Test model with multiple outputs on device 1."""
        device = torch.device("rbln:1")

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 32)
                self.linear2 = nn.Linear(16, 32)

            def forward(self, x):
                out1 = self.linear1(x)
                out2 = self.linear2(x)
                return out1, out2

        model = MultiOutputModel().to(device=device, dtype=dtype)
        x = torch.randn([4, 16], dtype=dtype, device=device)

        out1, out2 = model(x)

        self._assert_device_match(out1, device, "First output: ")
        self._assert_device_match(out2, device, "Second output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_complex_model_device_1(self, dtype):
        """Test a complex model using only ops from native_functions.yaml on device 1."""
        device = torch.device("rbln:1")

        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 64)
                self.linear2 = nn.Linear(64, 32)
                self.linear3 = nn.Linear(32, 16)

            def forward(self, x):
                # Linear transformation (native_functions: linear)
                x = self.linear1(x)
                # Element-wise operations (native_functions: add, mul, sigmoid)
                x = torch.add(x, torch.mul(x, 0.1))  # residual-like connection
                x = torch.sigmoid(x)  # activation
                # Linear transformation
                x = self.linear2(x)
                # More operations (native_functions: abs, clamp, log)
                x = torch.abs(x)
                x = torch.clamp(x, min=-2.0, max=2.0)
                x = torch.log(torch.add(torch.abs(x), 1.0))  # log(|x| + 1)
                # Final linear transformation
                x = self.linear3(x)
                return x

        model = ComplexModel().to(device=device, dtype=dtype)
        x = torch.randn([4, 32], dtype=dtype, device=device)

        output = model(x)

        self._assert_device_match(output, device, "Complex model output: ")
        self.assertEqual(output.shape, (4, 16))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_complex_model_device_1_128aligned(self, dtype):
        """Test complex model with 128-byte aligned last dimension using native_functions ops."""
        device = torch.device("rbln:1")
        # 128 bytes = 64 elements for float16 (2 bytes per element)

        class ComplexModel128Aligned(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                self.linear3 = nn.Linear(128, 64)

            def forward(self, x):
                # Linear transformation (native_functions: linear)
                x = self.linear1(x)
                # Element-wise operations (native_functions: add, mul, sigmoid)
                x = torch.add(x, torch.mul(x, 0.1))  # residual-like connection
                x = torch.sigmoid(x)  # activation
                # Linear transformation
                x = self.linear2(x)
                # More operations (native_functions: abs, clamp, log)
                x = torch.abs(x)
                x = torch.clamp(x, min=-2.0, max=2.0)
                x = torch.log(torch.add(torch.abs(x), 1.0))  # log(|x| + 1)
                # Matrix multiplication (native_functions: mm)
                # x shape: (4, 128), weight shape: (128, 128) -> output: (4, 128)
                weight = torch.randn(128, 128, device=x.device, dtype=x.dtype)
                x = torch.mm(x, weight)
                # Additional operations (native_functions: mean, maximum)
                x_mean = torch.mean(x, dim=-1, keepdim=True)
                x = torch.maximum(x, x_mean)
                # Final linear transformation
                x = self.linear3(x)
                return x

        model = ComplexModel128Aligned().to(device=device, dtype=dtype)
        x = torch.randn([4, 128], dtype=dtype, device=device)

        output = model(x)

        self._assert_device_match(output, device, "Complex model output (128-aligned): ")
        self.assertEqual(output.shape, (4, 64))

    # =========================================================================
    # Native Functions Tests - Basic Operations (non-aligned, aligned)
    # =========================================================================

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", [torch.add, torch.mul, torch.sub])
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_binary_ops_device_1(self, dtype, binary_op, shape):
        """Test binary operation device placement (native_functions: add.out, mul.out, sub.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)

        output = binary_op(x, y)
        self._assert_device_match(output, device, f"{binary_op.__name__} output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_div_device_1(self, dtype, shape):
        """Test div operation device placement (native_functions: div.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device) + 0.1  # avoid division by zero

        output = torch.div(x, y)
        self._assert_device_match(output, device, "Div output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("unary_op", [torch.abs, torch.ceil, torch.floor, torch.neg, torch.sigmoid, torch.nn.functional.silu])
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_unary_ops_device_1(self, dtype, unary_op, shape):
        """Test unary operation device placement (native_functions: abs.out, ceil.out, neg.out, sigmoid.out, silu.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)

        output = unary_op(x)
        self._assert_device_match(output, device, "Sigmoid output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_clamp_device_1(self, dtype, shape):
        """Test clamp operation device placement (native_functions: clamp.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)

        output = torch.clamp(x, min=-1.0, max=1.0)
        self._assert_device_match(output, device, "Clamp output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_log_device_1(self, dtype, shape):
        """Test log operation device placement (native_functions: log.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device).abs() + 0.1  # ensure positive

        output = torch.log(x)
        self._assert_device_match(output, device, "Log output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_rsqrt_device_1(self, dtype, shape):
        """Test rsqrt operation device placement (native_functions: rsqrt.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device).abs() + 0.1  # ensure positive

        output = torch.rsqrt(x)
        self._assert_device_match(output, device, "Rsqrt output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_pow_tensor_scalar_device_1(self, dtype, shape):
        """Test pow operation device placement (native_functions: pow.Tensor_Scalar_out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device).abs() + 0.1  # ensure positive

        output = torch.pow(x, 2.0)
        self._assert_device_match(output, device, "Pow output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_mean_device_1(self, dtype, shape):
        """Test mean operation device placement (native_functions: mean.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)

        output = torch.mean(x, dim=-1, keepdim=True)
        self._assert_device_match(output, device, "Mean output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("reduction_op", [torch.maximum, torch.minimum])
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_reduction_ops_device_1(self, dtype, reduction_op, shape):
        """Test reduction operation device placement (native_functions: maximum.out, minimum.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)

        output = reduction_op(x, y)
        self._assert_device_match(output, device, f"{reduction_op.__name__} output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_where_device_1(self, dtype, shape):
        """Test where operation device placement (native_functions: where.self)."""
        device = torch.device("rbln:1")
        condition = torch.randn(shape, dtype=dtype, device=device) > 0
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)

        output = torch.where(condition, x, y)
        self._assert_device_match(output, device, "Where output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_cat_device_1(self, dtype, shape):
        """Test cat operation device placement (native_functions: cat.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        z = torch.randn(shape, dtype=dtype, device=device)

        output = torch.cat([x, y, z], dim=1)
        self._assert_device_match(output, device, "Cat output: ")
        self.assertEqual(output.shape, (shape[0], shape[1] * 3))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("comparison_op", [torch.eq, torch.ne, torch.gt, torch.ge, torch.lt, torch.le])
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_comparison_ops_device_1(self, dtype, comparison_op, shape):
        """Test comparison operations device placement (native_functions: eq, ne, gt, ge, lt, le)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)

        # Test various comparison ops
        out = comparison_op(x, y)
        self._assert_device_match(out, device, f"{comparison_op.__name__} output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(4, 8), (4, 64)])
    def test_logical_not_device_1(self, dtype, shape):
        """Test logical_not operation device placement (native_functions: logical_not.out)."""
        device = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device) > 0

        output = torch.logical_not(x)
        self._assert_device_match(output, device, "Logical_not output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(4, 8, 16), (4, 64, 128)])
    def test_mm_device_1(self, dtype, n, m, p):
        """Test mm operation device placement (native_functions: mm.out)."""
        device = torch.device("rbln:1")
        x = torch.randn([n, m], dtype=dtype, device=device)
        y = torch.randn([m, p], dtype=dtype, device=device)

        output = torch.mm(x, y)
        self._assert_device_match(output, device, "MM output: ")
        self.assertEqual(output.shape, (n, p))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("b,n,m,p", [(2, 4, 8, 16), (2, 4, 64, 128)])
    def test_bmm_device_1(self, dtype, b, n, m, p):
        """Test bmm operation device placement (native_functions: bmm.out)."""
        device = torch.device("rbln:1")
        x = torch.randn([b, n, m], dtype=dtype, device=device)
        y = torch.randn([b, m, p], dtype=dtype, device=device)

        output = torch.bmm(x, y)
        self._assert_device_match(output, device, "BMM output: ")
        self.assertEqual(output.shape, (b, n, p))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(4, 8, 16), (4, 64, 128)])
    def test_addmm_device_1(self, dtype, n, m, p):
        """Test addmm operation device placement (native_functions: addmm.out)."""
        device = torch.device("rbln:1")
        x = torch.randn([n, p], dtype=dtype, device=device)
        y = torch.randn([n, m], dtype=dtype, device=device)
        z = torch.randn([m, p], dtype=dtype, device=device)

        output = torch.addmm(x, y, z)
        self._assert_device_match(output, device, "Addmm output: ")
        self.assertEqual(output.shape, (n, p))

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(4, 8, 16), (4, 64, 128)])
    def test_matmul_device_1(self, dtype, n, m, p):
        """Test matrix multiplication device placement."""
        device = torch.device("rbln:1")
        x = torch.randn([n, m], dtype=dtype, device=device)
        y = torch.randn([m, p], dtype=dtype, device=device)

        matmul_out = torch.matmul(x, y)
        self._assert_device_match(matmul_out, device, "Matmul output: ")

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("in_features,out_features", [(64, 64), (128, 128)])
    def test_linear_device_1(self, dtype, in_features, out_features):
        """Test Linear layer with 64-byte aligned last dimension."""
        device = torch.device("rbln:1")
        linear = nn.Linear(in_features, out_features).to(device=device, dtype=dtype)
        x = torch.randn([4, in_features], dtype=dtype, device=device)

        output = linear(x)
        self._assert_device_match(output, device, "Linear output (64-aligned): ")
        self.assertEqual(output.shape, (4, out_features))


instantiate_device_type_tests(TestMultiDevice, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestDeviceOutputPlacement, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
