# Owner(s): ["module: PrivateUse1"]

"""Importing torch_rbln must not freeze the ``RBLN_DEVICES`` mapping.

torch-rbln fixes the ``RBLN_DEVICES`` mapping once a device is used, and a later change is
then ignored. A vLLM data-parallel worker inherits a partition-wide ``RBLN_DEVICES`` and
remaps it per rank *after* import, so import must not fix the mapping.

Scope: the runtime half -- that a remap after import is still accepted. The torch half, that
nothing on the import path resolves a device at all, is pinned by
``test_import_does_not_resolve_a_device`` in test_privateuse1_contract.py.
"""

import os
import subprocess
import sys
import textwrap

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.test_set_ci
class TestImportDoesNotSeal(TestCase):
    """After import (or a non-RBLN profiler), ``RBLN_DEVICES`` must still be remappable.

    Each subprocess sets ``RBLN_DEVICES``, runs the scenario, remaps it, then counts the
    devices; the count must follow the remap. ``"0"`` -> ``"0,1"`` is a real change keeping
    logical device 0 valid.
    """

    def _assert_no_seal(self, result: subprocess.CompletedProcess):
        self.assertTrue(
            result.returncode == 0 and "NO_SEAL_OK" in result.stdout,
            "RBLN_DEVICES was sealed before the remap\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )

    def _run(self, script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_import_does_not_seal(self):
        script = f"""
            import os, sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            os.environ["RBLN_DEVICES"] = "0"
            import torch_rbln  # noqa: F401
            os.environ["RBLN_DEVICES"] = "0,1"   # worker remaps per rank AFTER import
            import rebel.v2, torch
            want, got = min(2, rebel.v2.device_count()), torch.rbln.device_count()
            print("NO_SEAL_OK" if got == want else f"SEALED: {{got}} device(s), not {{want}}")
        """
        self._assert_no_seal(self._run(script))

    def test_cpu_only_profiler_does_not_seal(self):
        script = f"""
            import os, sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            os.environ["RBLN_DEVICES"] = "0"
            import torch, torch_rbln
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU]):
                pass
            os.environ["RBLN_DEVICES"] = "0,1"
            import rebel.v2, torch
            want, got = min(2, rebel.v2.device_count()), torch.rbln.device_count()
            print("NO_SEAL_OK" if got == want else f"SEALED: {{got}} device(s), not {{want}}")
        """
        self._assert_no_seal(self._run(script))


if __name__ == "__main__":
    run_tests()
