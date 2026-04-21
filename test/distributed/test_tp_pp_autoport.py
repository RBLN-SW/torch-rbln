# Owner(s): ["module: PrivateUse1"]

"""
Concurrency smoke test for the RCCL autoport init path.
Spawns two concurrent invocations of ``test_tp_pp.py`` on disjoint
``RBLN_DEVICES`` partitions with distinct ``MASTER_PORT`` values, all under
``RCCL_PORT_GEN=1``. Both subprocess runs must complete successfully --
this asserts that the autoport / unique-id init path lets two independent
process groups coexist on a single host without port or device collisions.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


_TEST_TP_PP = Path(__file__).resolve().parent / "test_tp_pp.py"
_MIN_DEVICES = 8

_PARTITIONS = (
    {"RCCL_PORT_GEN": "1", "RBLN_DEVICES": "0,1,2,3", "MASTER_PORT": "29604"},
    {"RCCL_PORT_GEN": "1", "RBLN_DEVICES": "4,5,6,7", "MASTER_PORT": "29605"},
)


def _invoke_test_tp_pp(env_overrides: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(_TEST_TP_PP)],
        env=env,
        check=False,
    )


@pytest.mark.ci
@pytest.mark.single_worker
class TestTPPPAutoportConcurrent(TestCase):
    """Run test_tp_pp.py concurrently on two disjoint device partitions."""

    def test_autoport_concurrent_device_split(self):
        if not torch.rbln.is_available() or torch.rbln.device_count() < _MIN_DEVICES:
            self.skipTest(
                f"requires at least {_MIN_DEVICES} RBLN devices "
                f"(available: {torch.rbln.device_count() if torch.rbln.is_available() else 0})",
            )
        with ThreadPoolExecutor(max_workers=len(_PARTITIONS)) as executor:
            results = list(executor.map(_invoke_test_tp_pp, _PARTITIONS))

        failures = [
            (env["RBLN_DEVICES"], r.returncode)
            for env, r in zip(_PARTITIONS, results)
            if r.returncode != 0
        ]
        self.assertFalse(
            failures,
            msg=f"test_tp_pp.py concurrent autoport runs failed: {failures}",
        )


if __name__ == "__main__":
    run_tests()