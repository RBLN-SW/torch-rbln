# Owner(s): ["module: PrivateUse1"]

"""
Test suite for torch.rbln.get_device_properties / get_device_name.
"""

import os
import subprocess
import sys

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln._C
from test.utils import requires_physical_devices


@pytest.mark.test_set_ci
class TestDeviceProperties(TestCase):
    def test_properties_report_hardware(self):
        props = torch.rbln.get_device_properties(0)

        self.assertRegex(props.name, r"^RBLN-C[AIR]\d\d$")
        self.assertGreaterEqual(props.num_chiplet, 1)
        self.assertGreater(props.memory_per_chiplet, 0)
        # Reporting the per-chiplet figure as the total understates REBEL 4x, unnoticed on ATOM.
        self.assertEqual(props.total_memory, props.memory_per_chiplet * props.num_chiplet)

    def test_get_device_name_matches_properties(self):
        self.assertEqual(torch.rbln.get_device_name(0), torch.rbln.get_device_properties(0).name)

    def test_device_argument_forms(self):
        expected = torch.rbln.get_device_properties(0).total_memory
        for device in (0, "rbln:0", torch.device("rbln", 0)):
            with self.subTest(device=device):
                self.assertEqual(torch.rbln.get_device_properties(device).total_memory, expected)

    def test_omitted_device_is_the_current_device(self):
        torch.rbln.set_device(0)
        self.assertEqual(torch.rbln.get_device_properties().name, torch.rbln.get_device_properties(0).name)

    def test_rejects_a_non_rbln_device(self):
        with self.assertRaisesRegex(ValueError, "Expected rbln device"):
            torch.rbln.get_device_properties("cpu")

    def test_rejects_an_out_of_range_device(self):
        out_of_range = torch.rbln.device_count()
        with self.assertRaises(RuntimeError):
            torch.rbln.get_device_properties(out_of_range)


def _run_in_subprocess(env_vars, script):
    env = dict(os.environ)
    for key in ("RBLN_DEVICES", "RBLN_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(env_vars)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)


@pytest.mark.test_set_ci
class TestDevicePropertiesEnvVars(TestCase):
    """Cases that need the device mapping set before import, so they run in a subprocess."""

    @requires_physical_devices(1)
    def test_reports_the_npu_rbln_devices_selects(self):
        """rbln:0 is the NPU RBLN_DEVICES names, by system number, not the host's NPU 0."""
        npu = torch_rbln._C._device_npus()[-1]
        script = """
import torch
import torch_rbln

print(torch.rbln.device_count(), torch_rbln._C._device_npus(), torch.rbln.get_device_properties(0).name)
"""
        result = _run_in_subprocess({"RBLN_VISIBLE_DEVICES": str(npu)}, script)
        self.assertEqual(result.returncode, 0, result.stderr)

        count, npus, name = result.stdout.split()[-3:]
        self.assertEqual(count, "1")
        self.assertEqual(npus, f"[{npu}]")
        self.assertRegex(name, r"^RBLN-C[AIR]\d\d$")

    def test_dummy_device_is_rejected(self):
        """Dummy mode is host-backed with no NPU, so there is nothing to report."""
        script = """
import torch
import torch_rbln

try:
    torch.rbln.get_device_properties(0)
except RuntimeError as err:
    print("RAISED", err)
else:
    print("NO RAISE")
"""
        result = _run_in_subprocess({"RBLN_DUMMY_DEVICE": "1"}, script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RAISED", result.stdout)
        self.assertIn("RBLN_DUMMY_DEVICE", result.stdout)


if __name__ == "__main__":
    run_tests()
