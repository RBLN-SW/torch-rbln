# Owner(s): ["module: PrivateUse1"]

"""Importing torch_rbln must not freeze the ``RBLN_DEVICES`` mapping.

The rbln runtime fixes the ``RBLN_DEVICES`` mapping once a device is acquired, and a
later change is then rejected. Against a runtime older than
rebellions-sw/rebel_compiler#12904 it fixed the mapping on its first *device-resolving
call* instead (e.g. an NPU-name query), raising ``RBLN_DEVICES environment variable
changed at runtime (Sealed)``. A vLLM data-parallel worker inherits a partition-wide
``RBLN_DEVICES`` and remaps it per rank *after* import, so import must not resolve a
device on either runtime -- which is what this file pins.

torch-rbln #151 gated the kineto profiler on ``is_atom_device()`` at import, which called
``get_npu_name(0)`` and sealed ``RBLN_DEVICES`` (0.3.0rc0 OK, rc2 failed). The bridge now
registers unconditionally; ATOM is gated by the runtime (``rbln_kineto_is_active()``
reports inactive on ATOM, rebel-compiler #12079), not by an import-time arch query.

Scope note: what this file measures is the *runtime* still accepting a remap after import.
On a current runtime a query no longer freezes the mapping, so that alone would also pass
for an import that did query. The torch-side half -- that nothing on the import path
resolves a device at all -- is pinned by ``test_import_does_not_resolve_a_device`` in
test_privateuse1_contract.py, which reads the arch cache directly.
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

    Each subprocess sets ``RBLN_DEVICES``, runs the scenario, remaps it, then makes the
    first device-resolving call; it must not raise "changed at runtime (Sealed)".
    ``"0"`` -> ``"0,1"`` is a real change keeping logical device 0 valid; a non-seal
    device error is unrelated.
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
            from rebel._C import get_npu_name
            try:
                get_npu_name(0)
                print("NO_SEAL_OK")
            except RuntimeError as e:
                if "changed at runtime" in str(e) or "Sealed" in str(e):
                    print("SEALED: " + str(e).strip())
                    sys.exit(1)
                print("NO_SEAL_OK")  # a device-topology error is unrelated to the seal invariant
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
            from rebel._C import get_npu_name
            try:
                get_npu_name(0)
                print("NO_SEAL_OK")
            except RuntimeError as e:
                if "changed at runtime" in str(e) or "Sealed" in str(e):
                    print("SEALED: " + str(e).strip())
                    sys.exit(1)
                print("NO_SEAL_OK")
        """
        self._assert_no_seal(self._run(script))


if __name__ == "__main__":
    run_tests()
