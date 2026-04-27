# Owner(s): ["module: PrivateUse1"]

"""
Test suite for RBLN device mapping functionality.
"""

import os
import subprocess
import sys
from io import StringIO
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln._C


@pytest.mark.test_set_ci
class TestDeviceMapping(TestCase):
    """Test device mapping functionality without environment variables.

    These tests run in the current process and test the default behavior
    (1:1 mapping) or test APIs that don't depend on environment variables.
    """

    def test_device_summary_exists(self):
        """Test that device_summary function exists."""
        self.assertTrue(hasattr(torch.rbln, "device_summary"))
        self.assertTrue(callable(torch.rbln.device_summary))

    def test_device_summary_basic(self):
        """Test basic device_summary output format."""
        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            torch.rbln.device_summary()
            output = fake_out.getvalue()
            self.assertIn("[RBLN] Device Topology Initialized:", output)
            self.assertIn("Logical Device", output)
            self.assertIn("Physical NPU IDs", output)
            self.assertIn("Status", output)

    def test_get_device_topology(self):
        """Test _get_device_topology function."""
        topology = torch_rbln._C._get_device_topology()

        # Check that topology has the expected structure
        self.assertTrue(hasattr(topology, "entries"))
        self.assertTrue(hasattr(topology, "unused_physical_device_ids"))

        # Check entries is a list
        self.assertIsInstance(topology.entries, list)

        # Check unused_physical_device_ids is a list
        self.assertIsInstance(topology.unused_physical_device_ids, list)

        # Check that number of entries matches logical device count
        logical_count = torch.rbln.device_count()
        self.assertEqual(len(topology.entries), logical_count)

        # Check each entry has the expected structure
        for entry in topology.entries:
            self.assertTrue(hasattr(entry, "logical_device_index"))
            self.assertTrue(hasattr(entry, "physical_device_ids"))
            self.assertTrue(hasattr(entry, "is_aggregated"))

            # Check logical_device_index is an int
            self.assertIsInstance(entry.logical_device_index, int)
            self.assertGreaterEqual(entry.logical_device_index, 0)

            # Check physical_device_ids is a list
            self.assertIsInstance(entry.physical_device_ids, list)
            self.assertGreater(len(entry.physical_device_ids), 0)

            # Check each physical device ID is valid
            for pid in entry.physical_device_ids:
                self.assertIsInstance(pid, int)
                self.assertGreaterEqual(pid, 0)

            # Check is_aggregated is a boolean
            self.assertIsInstance(entry.is_aggregated, bool)

            # Check is_aggregated matches the number of physical devices
            expected_aggregated = len(entry.physical_device_ids) > 1
            self.assertEqual(entry.is_aggregated, expected_aggregated)

        # Check unused_physical_device_ids contains valid IDs
        for pid in topology.unused_physical_device_ids:
            self.assertIsInstance(pid, int)
            self.assertGreaterEqual(pid, 0)

        # Check that all logical device indices are present and unique
        logical_indices = [entry.logical_device_index for entry in topology.entries]
        self.assertEqual(len(logical_indices), len(set(logical_indices)))
        self.assertEqual(set(logical_indices), set(range(logical_count)))

    def test_device_count(self):
        """Test getting logical device count."""
        logical_count = torch.rbln.device_count()
        self.assertIsInstance(logical_count, int)
        self.assertGreaterEqual(logical_count, 0)
        # Logical count should be <= physical count
        physical_count = torch.rbln.physical_device_count()
        self.assertLessEqual(logical_count, physical_count)

    def test_device_count_equals_physical_count_default(self):
        """Test that default mode (1:1 mapping) results in equal counts."""
        # Skip if environment variables are set (they affect device mapping)
        if os.getenv("RBLN_DEVICE_MAP") is not None or os.getenv("RBLN_NPUS_PER_DEVICE") is not None:
            self.skipTest("This test requires no RBLN device mapping environment variables to be set")

        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()
        # In default mode, logical count should equal physical count
        self.assertEqual(logical_count, physical_count)

    def test_physical_device_ids_default_mapping(self):
        """Test that default mapping is 1:1."""
        # Skip if environment variables are set (they affect device mapping)
        if os.getenv("RBLN_DEVICE_MAP") is not None or os.getenv("RBLN_NPUS_PER_DEVICE") is not None:
            self.skipTest("This test requires no RBLN device mapping environment variables to be set")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        for i in range(logical_count):
            # Find entry with matching logical_device_index
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            # Default mapping should be 1:1
            self.assertEqual(len(physical_ids), 1)
            self.assertEqual(physical_ids[0], i)


instantiate_device_type_tests(TestDeviceMapping, globals(), only_for="privateuse1")


def run_test_with_env(env_vars, impl_test_func):
    """Run a test in a subprocess with specific environment variables.

    This function creates a standalone Python script that runs the test without
    importing path-based test modules, avoiding module import issues in subprocess.

    Args:
        env_vars: Dictionary of environment variables to set
        impl_test_func: The implementation test function (without test_ prefix)

    Returns:
        subprocess.CompletedProcess result
    """
    test_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(test_file))))

    # Get the function name and class name
    func_name = impl_test_func.__name__
    class_name = impl_test_func.__self__.__class__.__name__

    # Create a standalone test script that doesn't import path-based test modules
    import textwrap

    # Build environment variable setup code - directly set env vars in the script
    env_setup_lines = []
    env_setup_lines.append("# Remove conflicting env vars first")
    env_setup_lines.append('os.environ.pop("RBLN_DEVICE_MAP", None)')
    env_setup_lines.append('os.environ.pop("RBLN_NPUS_PER_DEVICE", None)')
    env_setup_lines.append('os.environ.pop("RBLN_DEVICES", None)')
    env_setup_lines.append("# Set the new environment variables")
    for k, v in env_vars.items():
        # Escape quotes in the value if needed
        escaped_v = repr(str(v))
        env_setup_lines.append(f'os.environ["{k}"] = {escaped_v}')
    env_setup_code = "\n        ".join(env_setup_lines)

    # Build environment variable cleanup code - unset env vars after test
    env_cleanup_lines = []
    env_cleanup_lines.append("# Clean up environment variables")
    for k in env_vars.keys():
        env_cleanup_lines.append(f'os.environ.pop("{k}", None)')
    env_cleanup_code = "\n            ".join(env_cleanup_lines)

    test_script = textwrap.dedent(
        f"""
        import os
        import sys

        # Add project root to path
        sys.path.insert(0, {repr(project_root)})

        # Set environment variables BEFORE any imports that might trigger device initialization
        {env_setup_code}

        # Now import what we need (avoiding test.rbln import)
        import torch
        from torch.testing._internal.common_utils import TestCase
        import torch_rbln._C

        # Import the test module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_device_mapping", {repr(test_file)})
        test_module = importlib.util.module_from_spec(spec)
        # Temporarily patch test.rbln import to avoid it
        import builtins
        original_import = builtins.__import__
        def patched_import(name, *args, **kwargs):
            if name == "test.rbln":
                # Create a mock module
                class MockModule:
                    pass
                return MockModule()
            return original_import(name, *args, **kwargs)
        builtins.__import__ = patched_import
        spec.loader.exec_module(test_module)
        builtins.__import__ = original_import

        # Get the test class and run the test
        {class_name} = getattr(test_module, {repr(class_name)})
        import unittest

        test_instance = {class_name}()
        test_instance.setUp()
        try:
            test_instance.{func_name}()
            print("TEST_PASSED")
            sys.exit(0)
        except unittest.SkipTest as e:
            print(f"TEST_SKIPPED: {{e}}")
            sys.exit(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            test_instance.tearDown()
            # Clean up environment variables
            {env_cleanup_code}
    """
    )

    # Run the test script
    result = subprocess.run([sys.executable, "-c", test_script], cwd=project_root, capture_output=True, text=True)
    return result


@pytest.mark.test_set_ci
class TestDeviceMappingEnvVars(TestCase):
    """Tests for device mapping with environment variables.

    These tests run in subprocesses to ensure clean environment variable setup.
    The device mapping is initialized once at process startup, so we need separate
    processes for different environment variable configurations.

    Structure:
    - Methods starting with "_impl": Actual test implementations that run in subprocess
      (do NOT start with "test_" so unittest won't auto-discover them)
    - Methods starting with "test_": Wrapper tests that launch subprocess with env vars
    """

    # ========== Wrapper tests (run in main process, launch subprocess) ==========

    def test_npus_per_device_1(self):
        """Test RBLN_NPUS_PER_DEVICE=1 (1:1 mapping)."""
        env_vars = {"RBLN_NPUS_PER_DEVICE": "1"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_1_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_npus_per_device_2(self):
        """Test RBLN_NPUS_PER_DEVICE=2 (grouping by 2)."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 2:
            self.skipTest(
                f"This test requires at least 2 physical devices for aggregation, but only {physical_count} are available"
            )
        env_vars = {"RBLN_NPUS_PER_DEVICE": "2"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_2_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_npus_per_device_4(self):
        """Test RBLN_NPUS_PER_DEVICE=4 (grouping by 4)."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        env_vars = {"RBLN_NPUS_PER_DEVICE": "4"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_4_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_simple(self):
        """Test RBLN_DEVICE_MAP with simple mapping."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        env_vars = {"RBLN_DEVICE_MAP": "[0,1],[2,3]"}
        result = run_test_with_env(env_vars, self.run_device_map_simple_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_complex(self):
        """Test RBLN_DEVICE_MAP with complex mapping."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 6:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")

        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2,3],[4,5]"}
        result = run_test_with_env(env_vars, self.run_device_map_complex_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_with_unused(self):
        """Test RBLN_DEVICE_MAP with some devices unused."""
        # Use RBLN_DEVICES to limit available devices to 5 (0,1,2,3,4)
        # With RBLN_DEVICE_MAP="[0,1],[2,3]", we use 4 devices, leaving 1 unused
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 5:
            self.skipTest(
                f"This test requires at least 5 physical devices to have unused ones, but only {physical_count} are available"
            )

        # Verify that device IDs 0-4 are available
        available_devices = list(range(min(5, physical_count)))
        env_vars = {"RBLN_DEVICE_MAP": "[0,1],[2,3]", "RBLN_DEVICES": ",".join(str(d) for d in available_devices)}
        result = run_test_with_env(env_vars, self.run_device_map_with_unused_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_non_contiguous(self):
        """Test RBLN_DEVICE_MAP with non-contiguous device IDs."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        # Use non-contiguous mapping: [0,2],[1,3] - devices are not consecutive
        env_vars = {"RBLN_DEVICE_MAP": "[0,2],[1,3]"}
        result = run_test_with_env(env_vars, self.run_device_map_non_contiguous_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_scattered(self):
        """Test RBLN_DEVICE_MAP with scattered device IDs."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 6:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")

        # Use scattered mapping: [0,4],[1,5] - devices with gaps
        env_vars = {"RBLN_DEVICE_MAP": "[0,4],[1,5]"}
        result = run_test_with_env(env_vars, self.run_device_map_scattered_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_reversed_order(self):
        """Test RBLN_DEVICE_MAP with reversed order device IDs."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        # Use reversed order: [2,0],[3,1] - devices in reverse order
        env_vars = {"RBLN_DEVICE_MAP": "[2,0],[3,1]"}
        result = run_test_with_env(env_vars, self.run_device_map_reversed_order_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_mixed_patterns(self):
        """Test RBLN_DEVICE_MAP with mixed patterns (non-contiguous and different sizes)."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")

        # Mixed pattern: [0,2],[1,3,5,7] - different sizes (2 and 4) and non-contiguous
        env_vars = {"RBLN_DEVICE_MAP": "[0,2],[1,3,5,7]"}
        result = run_test_with_env(env_vars, self.run_device_map_mixed_patterns_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_complex_non_contiguous(self):
        """Test RBLN_DEVICE_MAP with complex non-contiguous mapping."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")

        # Complex non-contiguous: [0,2,4,6],[1,3,5,7] - alternating pattern
        env_vars = {"RBLN_DEVICE_MAP": "[0,2,4,6],[1,3,5,7]"}
        result = run_test_with_env(env_vars, self.run_device_map_complex_non_contiguous_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_npus_per_device_invalid_size_error(self):
        """Test that invalid sizes are rejected for RBLN_NPUS_PER_DEVICE."""
        env_vars = {"RBLN_NPUS_PER_DEVICE": "3"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_invalid_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid size")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_npus_per_device_invalid_size_error_5(self):
        """Test that invalid size 5 is rejected."""
        env_vars = {"RBLN_NPUS_PER_DEVICE": "5"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_invalid_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid size")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_npus_per_device_invalid_size_error_6(self):
        """Test that invalid size 6 is rejected."""
        env_vars = {"RBLN_NPUS_PER_DEVICE": "6"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_invalid_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid size")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_npus_per_device_invalid_size_error_7(self):
        """Test that invalid size 7 is rejected."""
        env_vars = {"RBLN_NPUS_PER_DEVICE": "7"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_invalid_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid size")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_npus_per_device_valid_size_8(self):
        """Test that valid size 8 is accepted for RBLN_NPUS_PER_DEVICE."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_NPUS_PER_DEVICE": "8"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_valid_size_8_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_npus_per_device_valid_size_16(self):
        """Test that valid size 16 is accepted for RBLN_NPUS_PER_DEVICE."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 16:
            self.skipTest(f"This test requires at least 16 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_NPUS_PER_DEVICE": "16"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_valid_size_16_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_npus_per_device_valid_size_32(self):
        """Test that valid size 32 is accepted for RBLN_NPUS_PER_DEVICE."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 32:
            self.skipTest(f"This test requires at least 32 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_NPUS_PER_DEVICE": "32"}
        result = run_test_with_env(env_vars, self.run_npus_per_device_valid_size_32_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_invalid_group_size_error(self):
        """Test that invalid-sized groups are rejected for RBLN_DEVICE_MAP."""
        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2]"}
        result = run_test_with_env(env_vars, self.run_device_map_invalid_group_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid-sized group")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_device_map_multiple_invalid_groups_error(self):
        """Test that multiple invalid-sized groups are rejected."""
        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2],[3,4,5]"}
        result = run_test_with_env(env_vars, self.run_device_map_invalid_group_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid-sized groups")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_device_map_invalid_size_6_error(self):
        """Test that size 6 group is rejected."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 6:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2,3,4,5]"}
        result = run_test_with_env(env_vars, self.run_device_map_invalid_group_size_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with invalid-sized group")
        self.assertIn("valid sizes", result.stderr.lower() or result.stdout.lower())

    def test_device_map_out_of_range_error_message(self):
        """Test that RBLN_DEVICE_MAP out-of-range shows improved error with env and device_summary hint."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 2:
            self.skipTest(f"This test requires at least 2 physical devices, but only {physical_count} are available")
        # Process sees only devices 0,1 but map references [2,3] -> out of range
        env_vars = {
            "RBLN_DEVICES": "0,1",
            "RBLN_DEVICE_MAP": "[0,1],[2,3]",
        }
        result = run_test_with_env(env_vars, self.run_device_map_out_of_range_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with device out of range")
        out = (result.stderr or "") + (result.stdout or "")
        self.assertIn("out of range", out.lower(), "Error should mention 'out of range'")
        self.assertIn("RBLN_DEVICE_MAP=", out, "Error should show RBLN_DEVICE_MAP env")
        self.assertIn("RBLN_NPUS_PER_DEVICE=", out, "Error should show RBLN_NPUS_PER_DEVICE env")

    def test_npus_per_device_no_logical_device_error_message(self):
        """Test that RBLN_NPUS_PER_DEVICE with too few devices shows improved error with env and device_summary."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 2:
            self.skipTest(f"This test requires at least 2 physical devices, but only {physical_count} are available")
        # Process sees only 2 devices but NPUS_PER_DEVICE=4 -> no logical device
        env_vars = {
            "RBLN_DEVICES": "0,1",
            "RBLN_NPUS_PER_DEVICE": "4",
        }
        result = run_test_with_env(env_vars, self.run_npus_per_device_no_logical_device_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with no logical device")
        out = (result.stderr or "") + (result.stdout or "")
        self.assertIn("no logical device", out.lower(), "Error should mention 'no logical device'")
        self.assertIn("RBLN_NPUS_PER_DEVICE=4", out, "Error should show RBLN_NPUS_PER_DEVICE value")
        self.assertIn("RBLN_DEVICE_MAP=", out, "Error should show RBLN_DEVICE_MAP env")

    def test_device_index_not_assigned_error_message(self):
        """Test that using an unassigned device index shows improved error with env and device_summary hint."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 2:
            self.skipTest(
                f"This test requires at least 2 physical devices (1 logical with NPUS=2), but only {physical_count} are available"
            )
        # 2 devices + NPUS_PER_DEVICE=2 -> 1 logical device (rbln:0). Using rbln:1 triggers "not assigned"
        env_vars = {"RBLN_DEVICES": "0,1", "RBLN_NPUS_PER_DEVICE": "2"}
        result = run_test_with_env(env_vars, self.run_device_index_not_assigned_error_impl)
        self.assertNotEqual(result.returncode, 0, "Should fail with device index not assigned")
        out = (result.stderr or "") + (result.stdout or "")
        self.assertIn("not assigned", out.lower(), "Error should mention 'not assigned'")
        self.assertIn("logical device(s)", out.lower(), "Error should show logical device count")
        self.assertIn("RBLN_DEVICE_MAP=", out, "Error should show RBLN_DEVICE_MAP env")
        self.assertIn("RBLN_NPUS_PER_DEVICE=", out, "Error should show RBLN_NPUS_PER_DEVICE env")

    def test_torch_ones_device_rbln_errors_with_nonexistent_rbln_devices_filter(self):
        """Invalid RBLN_DEVICES must fail with a clear Python error (subprocess), not abort.

        The harness imports ``torch.testing._internal.common_utils``, which probes PrivateUse1
        availability, so the failure often appears at import time rather than at ``torch.ones``.
        """
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 1:
            self.skipTest("This test requires at least one physical RBLN device in the system")
        # ID that does not exist on the host: runtime rejects RBLN_DEVICES before tensor creation.
        env_vars = {"RBLN_DEVICES": "99999"}
        result = run_test_with_env(env_vars, self.run_torch_ones_rbln_nonexistent_devices_filter_impl)
        self.assertNotEqual(result.returncode, 0, "Expected failure when RBLN_DEVICES is invalid")
        out = (result.stderr or "") + (result.stdout or "")
        lowered = out.lower()
        self.assertTrue(
            "no logical device" in lowered
            or "not assigned" in lowered
            or "out of range" in lowered
            or "not available" in lowered
            or "rbln_devices" in lowered,
            f"Expected mapping/device error in output; got stdout+stderr:\n{out}",
        )

    def test_device_map_valid_size_8(self):
        """Test that valid size 8 group is accepted for RBLN_DEVICE_MAP."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2,3,4,5,6,7]"}
        result = run_test_with_env(env_vars, self.run_device_map_valid_size_8_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_map_valid_size_16(self):
        """Test that valid size 16 group is accepted for RBLN_DEVICE_MAP."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 16:
            self.skipTest(f"This test requires at least 16 physical devices, but only {physical_count} are available")
        env_vars = {"RBLN_DEVICE_MAP": "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"}
        result = run_test_with_env(env_vars, self.run_device_map_valid_size_16_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_priority_device_map_over_npus_per_device(self):
        """Test that RBLN_DEVICE_MAP takes priority over RBLN_NPUS_PER_DEVICE."""
        # Skip if not enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        env_vars = {"RBLN_DEVICE_MAP": "[0,1],[2,3]", "RBLN_NPUS_PER_DEVICE": "4"}
        result = run_test_with_env(env_vars, self.run_priority_device_map_over_npus_per_device_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_summary_with_aggregation(self):
        """Test device_summary output with aggregated mapping."""
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 2:
            self.skipTest(
                f"This test requires at least 2 physical devices for aggregation, but only {physical_count} are available"
            )
        env_vars = {"RBLN_NPUS_PER_DEVICE": "2"}
        result = run_test_with_env(env_vars, self.run_device_summary_with_aggregation_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_summary_with_unused_devices(self):
        """Test device_summary output with unused devices."""
        # Use RBLN_DEVICES to limit available devices to 5 (0,1,2,3,4)
        # With RBLN_NPUS_PER_DEVICE=4, we group 4 devices, leaving 1 unused
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 5:
            self.skipTest(
                f"This test requires at least 5 physical devices to have unused ones, but only {physical_count} are available"
            )

        # Verify that device IDs 0-4 are available
        available_devices = list(range(min(5, physical_count)))
        env_vars = {"RBLN_NPUS_PER_DEVICE": "4", "RBLN_DEVICES": ",".join(str(d) for d in available_devices)}
        result = run_test_with_env(env_vars, self.run_device_summary_with_unused_devices_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_device_count_equals_physical_count_default(self):
        """Test that default mode (1:1 mapping) results in equal counts."""
        # Use empty env_vars to ensure default behavior (no env vars set)
        env_vars = {"RBLN_NPUS_PER_DEVICE": "1"}
        result = run_test_with_env(env_vars, self.run_device_count_equals_physical_count_default_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_physical_device_ids_default_mapping(self):
        """Test that default mapping is 1:1."""
        # Use empty env_vars to ensure default behavior (no env vars set)
        env_vars = {"RBLN_NPUS_PER_DEVICE": "1"}
        result = run_test_with_env(env_vars, self.run_physical_device_ids_default_mapping_impl)
        self.assertEqual(
            result.returncode, 0, f"Subprocess test failed. STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    # ========== Implementation tests (run in subprocess with env vars) ==========
    # Note: These methods start with "run_" so unittest can discover them when run
    # via subprocess. They are NOT run directly by unittest discovery in the maidn process
    # because they require specific environment variables set by wrapper tests.

    def run_npus_per_device_1_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=1."""
        import torch_rbln._C

        # Verify environment variable is set
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "1", "RBLN_NPUS_PER_DEVICE should be set to 1 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()
        # With NPUS_PER_DEVICE=1, logical count should equal physical count
        self.assertEqual(logical_count, physical_count)

        # Each logical device should map to exactly one physical device
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 1)
            self.assertEqual(physical_ids[0], i)

        # No unused devices
        unused = topology.unused_physical_device_ids
        self.assertEqual(len(unused), 0)

    def run_npus_per_device_2_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=2."""
        import torch_rbln._C

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "2", "RBLN_NPUS_PER_DEVICE should be set to 2 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()

        # With NPUS_PER_DEVICE=2, logical count should be approximately half
        expected_logical = physical_count // 2
        self.assertEqual(logical_count, expected_logical)

        # Each logical device should map to 2 physical devices
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 2)
            # Check that physical IDs are consecutive
            self.assertEqual(physical_ids[1] - physical_ids[0], 1)

        # Check unused devices (if any)
        unused = topology.unused_physical_device_ids
        expected_unused = physical_count % 2
        self.assertEqual(len(unused), expected_unused)

    def run_npus_per_device_4_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=4."""
        import torch_rbln._C

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "4", "RBLN_NPUS_PER_DEVICE should be set to 4 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()

        # Skip if not enough physical devices
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        # With NPUS_PER_DEVICE=4, logical count should be approximately quarter
        expected_logical = physical_count // 4
        self.assertEqual(logical_count, expected_logical)

        # Each logical device should map to 4 physical devices
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 4)
            # Check that physical IDs are consecutive
            for j in range(1, len(physical_ids)):
                self.assertEqual(physical_ids[j] - physical_ids[j - 1], 1)

        # Check unused devices (if any)
        unused = topology.unused_physical_device_ids
        expected_unused = physical_count % 4
        self.assertEqual(len(unused), expected_unused)

    def run_device_map_simple_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with simple mapping."""
        import torch_rbln._C

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,1],[2,3]", "RBLN_DEVICE_MAP should be set to '[0,1],[2,3]' in subprocess")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,1],[2,3]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 1]
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 1])

        # Check second logical device maps to [2, 3]
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [2, 3])

        # No unused devices (assuming 4 physical devices)
        unused = topology.unused_physical_device_ids
        if physical_count == 4:
            self.assertEqual(len(unused), 0)

    def run_device_map_complex_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with complex mapping."""
        import torch_rbln._C

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,1,2,3],[4,5]", "RBLN_DEVICE_MAP should be set correctly")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 6:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,1,2,3],[4,5]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 1, 2, 3]
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 1, 2, 3])

        # Check second logical device maps to [4, 5]
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [4, 5])

    def run_device_map_with_unused_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with unused devices."""
        import torch_rbln._C

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,1],[2,3]", "RBLN_DEVICE_MAP should be set correctly")

        topology = torch_rbln._C._get_device_topology()
        physical_count = torch.rbln.physical_device_count()

        # Skip if not enough physical devices (need at least 5 to have unused devices)
        if physical_count < 5:
            self.skipTest(
                f"This test requires at least 5 physical devices to have unused ones, but only {physical_count} are available"
            )

        # Check unused devices
        unused = topology.unused_physical_device_ids
        # If we have more than 4 physical devices, some should be unused
        self.assertGreater(len(unused), 0, "Should have unused devices when physical_count > 4")
        # Unused devices should be >= 4 (since we use 0,1,2,3)
        for unused_id in unused:
            self.assertGreaterEqual(unused_id, 4)

    def run_device_map_non_contiguous_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with non-contiguous device IDs."""
        import torch_rbln._C

        # Verify environment variable is set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,2],[1,3]", "RBLN_DEVICE_MAP should be set to '[0,2],[1,3]' in subprocess")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,2],[1,3]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 2] (non-contiguous)
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 2])

        # Check second logical device maps to [1, 3] (non-contiguous)
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [1, 3])

    def run_device_map_scattered_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with scattered device IDs."""
        import torch_rbln._C

        # Verify environment variable is set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,4],[1,5]", "RBLN_DEVICE_MAP should be set to '[0,4],[1,5]' in subprocess")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 6:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,4],[1,5]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 4] (scattered with gap)
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 4])

        # Check second logical device maps to [1, 5] (scattered with gap)
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [1, 5])

    def run_device_map_reversed_order_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with reversed order device IDs."""
        import torch_rbln._C

        # Verify environment variable is set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[2,0],[3,1]", "RBLN_DEVICE_MAP should be set to '[2,0],[3,1]' in subprocess")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[2,0],[3,1]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [2, 0] (reversed order)
        # The order might be preserved or sorted, so check the set of IDs
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(set(physical_ids_0), {0, 2}, f"Expected {{0, 2}}, got {physical_ids_0}")
        self.assertEqual(len(physical_ids_0), 2)

        # Check second logical device maps to [3, 1] (reversed order)
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(set(physical_ids_1), {1, 3}, f"Expected {{1, 3}}, got {physical_ids_1}")
        self.assertEqual(len(physical_ids_1), 2)

    def run_device_map_mixed_patterns_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with mixed patterns."""
        import torch_rbln._C

        # Verify environment variable is set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(
            device_map, "[0,2],[1,3,5,7]", "RBLN_DEVICE_MAP should be set to '[0,2],[1,3,5,7]' in subprocess"
        )

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 6 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,2],[1,3,5,7]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 2] (size 2, non-contiguous)
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 2])
        self.assertEqual(len(physical_ids_0), 2)

        # Check second logical device maps to [1, 3, 5, 7] (size 4, non-contiguous)
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [1, 3, 5, 7])
        self.assertEqual(len(physical_ids_1), 4)

    def run_device_map_complex_non_contiguous_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with complex non-contiguous mapping."""
        import torch_rbln._C

        # Verify environment variable is set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(
            device_map, "[0,2,4,6],[1,3,5,7]", "RBLN_DEVICE_MAP should be set to '[0,2,4,6],[1,3,5,7]' in subprocess"
        )

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,2,4,6],[1,3,5,7]", we should have 2 logical devices
        self.assertEqual(logical_count, 2)

        # Check first logical device maps to [0, 2, 4, 6] (alternating pattern, even numbers)
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 2, 4, 6])
        self.assertEqual(len(physical_ids_0), 4)

        # Check second logical device maps to [1, 3, 5, 7] (alternating pattern, odd numbers)
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [1, 3, 5, 7])
        self.assertEqual(len(physical_ids_1), 4)

    def run_npus_per_device_invalid_size_error_impl(self):
        """Implementation: Test that invalid sizes are rejected."""
        # This should fail during initialization in C++ code
        # If we reach here, the test failed (should have failed earlier)
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertIsNotNone(npus_per_device, "Environment variable should be set")
        # The actual error checking happens in C++ code during initialization
        # If initialization succeeded, that's a bug
        pass  # noqa: PIE790

    def run_npus_per_device_valid_size_8_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=8."""
        import torch_rbln._C

        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "8", "RBLN_NPUS_PER_DEVICE should be set to 8 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()

        # With NPUS_PER_DEVICE=8, logical count should be approximately one-eighth
        expected_logical = physical_count // 8
        self.assertEqual(logical_count, expected_logical)

        # Each logical device should map to 8 physical devices
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 8)
            # Check that physical IDs are consecutive
            for j in range(1, len(physical_ids)):
                self.assertEqual(physical_ids[j] - physical_ids[j - 1], 1)

        # Check unused devices (if any)
        unused = topology.unused_physical_device_ids
        expected_unused = physical_count % 8
        self.assertEqual(len(unused), expected_unused)

    def run_npus_per_device_valid_size_16_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=16."""
        import torch_rbln._C

        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "16", "RBLN_NPUS_PER_DEVICE should be set to 16 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()

        # With NPUS_PER_DEVICE=16, logical count should be approximately one-sixteenth
        expected_logical = physical_count // 16
        self.assertEqual(logical_count, expected_logical)

        # Each logical device should map to 16 physical devices
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 16)
            # Check that physical IDs are consecutive
            for j in range(1, len(physical_ids)):
                self.assertEqual(physical_ids[j] - physical_ids[j - 1], 1)

        # Check unused devices (if any)
        unused = topology.unused_physical_device_ids
        expected_unused = physical_count % 16
        self.assertEqual(len(unused), expected_unused)

    def run_npus_per_device_valid_size_32_impl(self):
        """Implementation: Test RBLN_NPUS_PER_DEVICE=32."""
        import torch_rbln._C

        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "32", "RBLN_NPUS_PER_DEVICE should be set to 32 in subprocess")

        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()

        # With NPUS_PER_DEVICE=32, logical count should be approximately one-thirty-second
        expected_logical = physical_count // 32
        self.assertEqual(logical_count, expected_logical)

        # Each logical device should map to 32 physical devices
        for i in range(logical_count):
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            self.assertEqual(len(physical_ids), 32)
            # Check that physical IDs are consecutive
            for j in range(1, len(physical_ids)):
                self.assertEqual(physical_ids[j] - physical_ids[j - 1], 1)

        # Check unused devices (if any)
        unused = topology.unused_physical_device_ids
        expected_unused = physical_count % 32
        self.assertEqual(len(unused), expected_unused)

    def run_device_map_invalid_group_size_error_impl(self):
        """Implementation: Test that invalid-sized groups are rejected."""
        # This should fail during initialization in C++ code
        # If we reach here, the test failed (should have failed earlier)
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertIsNotNone(device_map, "Environment variable should be set")
        # The actual error checking happens in C++ code during initialization
        # If initialization succeeded, that's a bug
        pass  # noqa: PIE790

    def run_device_map_out_of_range_error_impl(self):
        """Implementation: Trigger RBLN_DEVICE_MAP out-of-range error (fails at import/init)."""
        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        # Error happens during device mapping init when torch is imported; if we reach here, init
        # succeeded (bug) or we are only checking env was set
        pass  # noqa: PIE790

    def run_npus_per_device_no_logical_device_error_impl(self):
        """Implementation: Trigger no logical device error (fails at import/init)."""
        npus = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        # Error happens during device mapping init; if we reach here, init succeeded (bug)
        pass  # noqa: PIE790

    def run_device_index_not_assigned_error_impl(self):
        """Implementation: Trigger device index not assigned by using rbln:1 when only rbln:0 exists."""
        npus = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        # With NPUS_PER_DEVICE=2 and 2 devices we have 1 logical device (rbln:0). Use rbln:1.
        torch.tensor([1], device="rbln:1", dtype=torch.float32)

    def run_torch_ones_rbln_nonexistent_devices_filter_impl(self):
        """Implementation: invalid RBLN_DEVICES; may fail at import (TestCase) or at torch.ones."""
        if os.getenv("RBLN_DEVICES") != "99999":
            self.skipTest("This test must be run via wrapper with RBLN_DEVICES=99999")
        # If the subprocess survived imports, exercise the same path as a minimal user script.
        torch.ones([1], device="rbln")

    def run_device_map_valid_size_8_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with size 8 group."""
        import torch_rbln._C

        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(device_map, "[0,1,2,3,4,5,6,7]", "RBLN_DEVICE_MAP should be set correctly")

        topology = torch_rbln._C._get_device_topology()
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 8:
            self.skipTest(f"This test requires at least 8 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,1,2,3,4,5,6,7]", we should have 1 logical device
        self.assertEqual(logical_count, 1)

        # Check logical device maps to [0, 1, 2, 3, 4, 5, 6, 7]
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(physical_ids_0), 8)

    def run_device_map_valid_size_16_impl(self):
        """Implementation: Test RBLN_DEVICE_MAP with size 16 group."""
        import torch_rbln._C

        device_map = os.getenv("RBLN_DEVICE_MAP")
        if device_map is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(
            device_map, "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]", "RBLN_DEVICE_MAP should be set correctly"
        )

        topology = torch_rbln._C._get_device_topology()
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 16:
            self.skipTest(f"This test requires at least 16 physical devices, but only {physical_count} are available")

        logical_count = torch.rbln.device_count()
        # With "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]", we should have 1 logical device
        self.assertEqual(logical_count, 1)

        # Check logical device maps to [0, 1, 2, ..., 15]
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        expected_ids = list(range(16))
        self.assertEqual(physical_ids_0, expected_ids)
        self.assertEqual(len(physical_ids_0), 16)

    def run_priority_device_map_over_npus_per_device_impl(self):
        """Implementation: Test that RBLN_DEVICE_MAP takes priority."""
        import torch_rbln._C

        # Both environment variables are set
        device_map = os.getenv("RBLN_DEVICE_MAP")
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")

        # If running directly (not via subprocess), skip this test
        if device_map is None or npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")

        self.assertIsNotNone(device_map, "RBLN_DEVICE_MAP should be set")
        self.assertIsNotNone(npus_per_device, "RBLN_NPUS_PER_DEVICE should be set")

        topology = torch_rbln._C._get_device_topology()
        # Check if we have enough physical devices for this test
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 4:
            self.skipTest(f"This test requires at least 4 physical devices, but only {physical_count} are available")

        # RBLN_DEVICE_MAP should take priority
        logical_count = torch.rbln.device_count()
        # With "[0,1],[2,3]", we should have 2 logical devices (not 1 from NPUS_PER_DEVICE=4)
        self.assertEqual(logical_count, 2)

        # Verify the mapping matches RBLN_DEVICE_MAP, not RBLN_NPUS_PER_DEVICE
        entry_0 = next((e for e in topology.entries if e.logical_device_index == 0), None)
        self.assertIsNotNone(entry_0, "Entry for logical device 0 not found")
        physical_ids_0 = entry_0.physical_device_ids
        self.assertEqual(physical_ids_0, [0, 1])
        entry_1 = next((e for e in topology.entries if e.logical_device_index == 1), None)
        self.assertIsNotNone(entry_1, "Entry for logical device 1 not found")
        physical_ids_1 = entry_1.physical_device_ids
        self.assertEqual(physical_ids_1, [2, 3])

    def run_device_summary_with_aggregation_impl(self):
        """Implementation: Test device_summary output with aggregated mapping."""
        from io import StringIO
        from unittest.mock import patch

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "2", "RBLN_NPUS_PER_DEVICE should be set to 2")

        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            torch.rbln.device_summary()
            output = fake_out.getvalue()

        # Check output contains expected elements
        self.assertIn("[RBLN] Device Topology Initialized:", output)
        self.assertIn("Logical Device", output)
        self.assertIn("Physical NPU IDs", output)
        self.assertIn("Status", output)
        self.assertIn("Active (Aggregated)", output)

        # Check that we have aggregated devices
        logical_count = torch.rbln.device_count()
        if logical_count > 0:
            self.assertIn("rbln:0", output)

    def run_device_summary_with_unused_devices_impl(self):
        """Implementation: Test device_summary output with unused devices."""
        from io import StringIO
        from unittest.mock import patch

        # Verify environment variable is set
        # If running directly (not via subprocess), skip this test
        npus_per_device = os.getenv("RBLN_NPUS_PER_DEVICE")
        if npus_per_device is None:
            self.skipTest("This test must be run via wrapper test with environment variables set")
        self.assertEqual(npus_per_device, "4", "RBLN_NPUS_PER_DEVICE should be set to 4")

        # Check if we have enough physical devices (need at least 5 to have unused devices)
        physical_count = torch.rbln.physical_device_count()
        if physical_count < 5:
            self.skipTest(
                f"This test requires at least 5 physical devices to have unused ones, but only {physical_count} are available"
            )

        # Verify unused devices exist before checking output
        topology = torch_rbln._C._get_device_topology()
        unused = topology.unused_physical_device_ids
        self.assertGreater(len(unused), 0, "Should have unused devices")

        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            torch.rbln.device_summary()
            output = fake_out.getvalue()

        # Check output contains expected elements
        self.assertIn("[RBLN] Device Topology Initialized:", output)
        # With RBLN_DEVICES limiting to 5 devices and NPUS_PER_DEVICE=4, there should be unused devices
        self.assertIn("Unused", output)
        self.assertIn("[Warning]", output)
        self.assertIn("unused", output.lower())

    def run_device_count_equals_physical_count_default_impl(self):
        """Implementation: Test that default mode (1:1 mapping) results in equal counts."""
        logical_count = torch.rbln.device_count()
        physical_count = torch.rbln.physical_device_count()
        # In default mode, logical count should equal physical count
        self.assertEqual(logical_count, physical_count)

    def run_physical_device_ids_default_mapping_impl(self):
        """Implementation: Test that default mapping is 1:1."""
        topology = torch_rbln._C._get_device_topology()
        logical_count = torch.rbln.device_count()
        for i in range(logical_count):
            # Find entry with matching logical_device_index
            entry = next((e for e in topology.entries if e.logical_device_index == i), None)
            self.assertIsNotNone(entry, f"Entry for logical device {i} not found")
            physical_ids = entry.physical_device_ids
            # Default mapping should be 1:1
            self.assertEqual(len(physical_ids), 1)
            self.assertEqual(physical_ids[0], i)


# Note: TestDeviceMappingEnvVars does not use instantiate_device_type_tests
# because its tests run in subprocesses with specific environment variables.
# The wrapper tests (without _impl suffix) handle the subprocess execution.


if __name__ == "__main__":
    run_tests()
