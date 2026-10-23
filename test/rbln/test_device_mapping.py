# Owner(s): ["module: PrivateUse1"]

"""
rbln:N is the N-th NPU that RBLN_DEVICES selects. A torch device is one NPU, so the variables
that grouped several NPUs into one device, RBLN_NPUS_PER_DEVICE and RBLN_DEVICE_MAP, are refused.
"""

import os
import subprocess
import sys
import textwrap
from io import StringIO
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln._C
from test.utils import requires_physical_devices


_SELECTION_VARS = (
    "RBLN_DEVICES",
    "RBLN_VISIBLE_DEVICES",
    "RBLN_NPUS_PER_DEVICE",
    "RBLN_DEVICE_MAP",
    "RBLN_DUMMY_DEVICE",
)

_GROUPING = [("RBLN_NPUS_PER_DEVICE", "2"), ("RBLN_DEVICE_MAP", "[0,1]"), ("RBLN_NPUS_PER_DEVICE", "1")]

_REPORT_REFUSAL = """
count, available = torch.rbln.device_count(), torch.rbln.is_available()
try:
    torch.ones(4, device="rbln:0")
except RuntimeError as e:
    print("RAISED", count, available, str(e).splitlines()[0])
else:
    print("NO RAISE")
"""


def _run(snippet, env):
    """Runs ``snippet`` after ``import torch, torch_rbln`` in a process whose RBLN_* selection is ``env``."""
    full_env = {key: value for key, value in os.environ.items() if key not in _SELECTION_VARS}
    full_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", "import os, torch, torch_rbln\n" + textwrap.dedent(snippet)],
        env=full_env,
        capture_output=True,
        text=True,
        errors="replace",
    )


def _refusal(names_and_values):
    return f"{names_and_values} set, but a torch device is one NPU on rebel.v2"


@pytest.mark.test_set_ci
class TestDeviceMapping(TestCase):
    def test_each_device_is_one_npu(self):
        npus = torch_rbln._C._device_npus()
        self.assertEqual(len(npus), torch.rbln.device_count())
        self.assertEqual(len(set(npus)), len(npus))
        if not torch.rbln.is_dummy_device():
            self.assertEqual(len(npus), torch.rbln.physical_device_count())

    def test_device_summary_names_the_npu_of_each_device(self):
        with patch("sys.stdout", new=StringIO()) as out:
            torch.rbln.device_summary()
        rows = [line.split("|")[1:3] for line in out.getvalue().splitlines() if line.startswith("| rbln:")]
        self.assertEqual(
            [(device.strip(), int(npu)) for device, npu in rows],
            [(f"rbln:{index}", npu) for index, npu in enumerate(torch_rbln._C._device_npus())],
        )


@pytest.mark.test_set_ci
class TestDeviceMappingEnvVars(TestCase):
    """The mapping is planned per process from RBLN_*, so each case runs in a subprocess."""

    @requires_physical_devices(2)
    def test_rbln_devices_numbers_the_devices_in_its_order(self):
        first, second = torch_rbln._C._device_npus()[:2]
        proc = _run(
            "print('NPUS', torch_rbln._C._device_npus(), torch.rbln.get_device_properties(1).name)",
            {"RBLN_DEVICES": f"{second},{first}"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"NPUS [{second}, {first}] RBLN-", proc.stdout)

    @requires_physical_devices(1)
    def test_an_index_past_the_devices_is_refused(self):
        proc = _run(
            """
            try:
                torch.rbln.set_device(1)
            except RuntimeError as e:
                print("RAISED", str(e).splitlines()[0])
            """,
            {"RBLN_DEVICES": str(torch_rbln._C._device_npus()[0])},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("RAISED rbln:1 is not a device of this process, which has 1 device(s)", proc.stdout)

    def test_a_nonexistent_npu_is_left_out(self):
        proc = _run(_REPORT_REFUSAL, {"RBLN_DEVICES": "99999"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("RAISED 0 False Cannot use rbln:0: no device available", proc.stdout)

    def test_a_grouping_variable_is_refused_where_a_device_is_used(self):
        """Enumeration stays quiet (0 devices, not available); using a device raises the reason."""
        for name, value in _GROUPING:
            with self.subTest(name=name, value=value):
                proc = _run(_REPORT_REFUSAL, {name: value})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("RAISED 0 False", proc.stdout)
                self.assertIn(_refusal(f"{name}={value} is"), proc.stdout)
                self.assertIn("select the NPUs with RBLN_DEVICES", proc.stdout)

    def test_both_grouping_variables_are_named(self):
        proc = _run(_REPORT_REFUSAL, {"RBLN_NPUS_PER_DEVICE": "2", "RBLN_DEVICE_MAP": "[0,1]"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(_refusal("RBLN_NPUS_PER_DEVICE=2 and RBLN_DEVICE_MAP=[0,1] are"), proc.stdout)

    def test_a_grouping_variable_is_refused_under_dummy(self):
        proc = _run(
            _REPORT_REFUSAL,
            {"RBLN_DUMMY_DEVICE": "1", "RBLN_FORCE_NPU_NAME": "RBLN-CA25", "RBLN_NPUS_PER_DEVICE": "2"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("RAISED 0 False", proc.stdout)
        self.assertIn(_refusal("RBLN_NPUS_PER_DEVICE=2 is"), proc.stdout)

    @requires_physical_devices(1)
    def test_unsetting_a_grouping_variable_before_first_use_takes_effect(self):
        """The refusal is part of the plan, which follows RBLN_* until the first device use."""
        proc = _run(
            """
            os.environ["RBLN_NPUS_PER_DEVICE"] = "2"
            before = torch.rbln.device_count()
            del os.environ["RBLN_NPUS_PER_DEVICE"]
            print("COUNTS", before, torch.rbln.device_count(), torch.ones(4, device="rbln:0").device)
            """,
            {"RBLN_DEVICES": str(torch_rbln._C._device_npus()[0])},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("COUNTS 0 1 rbln:0", proc.stdout)


if __name__ == "__main__":
    run_tests()
