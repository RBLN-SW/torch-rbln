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

from ..utils import requires_physical_devices


@pytest.mark.test_set_ci
class TestDeviceProperties(TestCase):
    def test_properties_report_hardware(self):
        props = torch.rbln.get_device_properties(0)

        self.assertRegex(props.name, r"^RBLN-C[AIR]\d\d$")
        self.assertGreaterEqual(props.num_chiplet, 1)
        self.assertGreaterEqual(props.npu_count, 1)
        self.assertGreater(props.memory_per_chiplet, 0)
        # Reporting the per-chiplet figure as the total understates REBEL 4x, unnoticed on ATOM.
        self.assertEqual(props.total_memory, props.memory_per_chiplet * props.num_chiplet * props.npu_count)

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
    for key in ("RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE", "RBLN_DEVICES"):
        env.pop(key, None)
    env.update(env_vars)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)


@pytest.mark.test_set_ci
class TestDevicePropertiesEnvVars(TestCase):
    """Cases that need the device mapping set before import, so they run in a subprocess."""

    @requires_physical_devices(2)
    def test_an_aggregated_device_sums_its_npus(self):
        """With two NPUs behind rbln:0, total_memory doubles while the per-chiplet figure does not."""
        script = """
import torch
import torch_rbln

props = torch.rbln.get_device_properties(0)
print(props.npu_count, props.total_memory, props.memory_per_chiplet, props.num_chiplet)
"""
        one = _run_in_subprocess({}, script)
        two = _run_in_subprocess({"RBLN_NPUS_PER_DEVICE": "2"}, script)
        self.assertEqual(one.returncode, 0, one.stderr)
        self.assertEqual(two.returncode, 0, two.stderr)

        one_count, one_total, one_per_chiplet, one_chiplets = map(int, one.stdout.split()[-4:])
        two_count, two_total, two_per_chiplet, two_chiplets = map(int, two.stdout.split()[-4:])

        self.assertEqual((one_count, two_count), (1, 2))
        self.assertEqual(two_total, one_total * 2)
        self.assertEqual(two_per_chiplet, one_per_chiplet)
        self.assertEqual(two_chiplets, one_chiplets)

    @requires_physical_devices(2)
    def test_a_narrowed_pool_is_not_remapped_twice(self):
        """RBLN_VISIBLE_DEVICES renumbers the pool from 0, and the query resolves that once.

        The topology reports pool ids while the runtime query resolves a pool id to a system
        id, so passing an already-resolved id would either read another card or fall off the
        end of the mapping. Hiding all but the last NPU makes the difference visible: the pool
        holds one device at index 0, whose system id is the highest on the host.
        """
        last = torch.rbln.physical_device_count() - 1
        script = """
import torch
import torch_rbln

entry = torch_rbln._C._get_device_topology().entries[0]
print(torch.rbln.device_count(), list(entry.physical_device_ids), torch.rbln.get_device_properties(0).name)
"""
        result = _run_in_subprocess({"RBLN_VISIBLE_DEVICES": str(last)}, script)
        self.assertEqual(result.returncode, 0, result.stderr)

        count, physical_ids, name = result.stdout.split()[-3:]
        self.assertEqual(count, "1")
        self.assertEqual(physical_ids, "[0]")
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
