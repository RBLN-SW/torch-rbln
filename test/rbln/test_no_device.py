# Owner(s): ["module: PrivateUse1"]

"""Device-layer behavior when no RBLN device is available (``device_count() == 0``).

On a host with no physical NPU, torch-rbln must still import and behave like
``torch.cuda`` on a CPU-only host: enumeration and capability queries degrade
gracefully, memory-management calls are no-ops, but selecting or actually using
a device raises. This lets a model be traced / compiled with no hardware (the
distributed / DeviceMesh side of that story lives in
``test/distributed/test_no_device.py``).

The contracts that only manifest with zero devices skip when this host has NPUs.
"""

import os
import subprocess
import sys
import textwrap
import unittest

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from test.utils import requires_physical_devices


_HAS_NPU = torch.rbln.device_count() > 0
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NEEDS_NO_NPU = "requires a host with no physical NPU (device_count() == 0)"


@pytest.mark.test_set_ci
class TestNoDevice(TestCase):
    """``device_count() == 0`` must be a valid, non-fatal state (torch.cuda parity)."""

    def test_device_count_and_is_available_do_not_raise(self):
        """Enumeration / availability never raise; on a no-NPU host they report 0 / False."""
        n = torch.rbln.device_count()
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)
        self.assertEqual(torch.rbln.is_available(), n > 0)
        if not _HAS_NPU:
            self.assertEqual(n, 0)
            self.assertFalse(torch.rbln.is_available())

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_set_and_current_device_raise_without_device(self):
        """Selecting / querying the current device raises with no NPU (torch.cuda parity)."""
        with self.assertRaises(RuntimeError):
            torch.rbln.set_device(0)
        with self.assertRaises(RuntimeError):
            torch.rbln.current_device()
        # The device context manager selects a device too -> also raises.
        with self.assertRaises(RuntimeError):
            with torch.rbln.device(0):
                pass

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_use_ops_fail_at_point_of_use(self):
        """Operations that truly use a device must fail with 0 devices."""
        with self.assertRaises(Exception):
            t = torch.ones(8, dtype=torch.float16, device="rbln:0")
            _ = t + t  # force materialization / device access
        with self.assertRaises(Exception):
            torch.rbln.synchronize()

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_memory_management_is_graceful(self):
        """empty_cache / memory_stats / reset_* are no-ops / empty with 0 devices (torch.cuda parity)."""
        import torch_rbln.memory as rbln_memory

        rbln_memory.empty_cache()  # no-op, must not raise
        rbln_memory.reset_peak_memory_stats()  # no-op, must not raise
        rbln_memory.reset_accumulated_memory_stats()  # no-op, must not raise
        self.assertEqual(rbln_memory.memory_stats(), {})
        self.assertEqual(rbln_memory.memory_allocated(), 0)
        self.assertEqual(rbln_memory.memory_reserved(), 0)

    def test_is_initialized_tracks_set_device(self):
        """``is_initialized()`` exists (DeviceMesh requires it).

        With a device it starts False and flips True on ``set_device``; with no device
        it reports True so DeviceMesh skips its auto-select path. Run in a fresh
        subprocess so the process-wide flag starts from its default.
        """
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            import torch, torch_rbln
            assert isinstance(torch.rbln.is_initialized(), bool)
            if torch.rbln.device_count() > 0:
                assert torch.rbln.is_initialized() is False, "should start uninitialized with a device"
                torch.rbln.set_device(0)
                assert torch.rbln.is_initialized() is True, "set_device should initialize"
            else:
                # No device: is_initialized() reports True so DeviceMesh skips auto-select
                # (get_rank() % device_count()==0 / set_device both fail with no NPU).
                assert torch.rbln.is_initialized() is True, "no-device should report initialized"
            print("INIT_OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=90
        )
        self.assertTrue(
            result.returncode == 0 and "INIT_OK" in result.stdout,
            f"is_initialized() semantics wrong (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )

    @requires_physical_devices(1)
    def test_no_device_contract_forced(self):
        """Run the no-device contract on a host that *has* NPUs by hiding them with a
        nonexistent ``RBLN_DEVICES`` filter, so CI on real hardware covers this path."""
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {_PROJECT_ROOT!r})
            import torch, torch_rbln
            import torch_rbln.memory as m
            assert torch.rbln.device_count() == 0, torch.rbln.device_count()
            assert torch.rbln.is_available() is False
            for call in (lambda: torch.rbln.set_device(0), torch.rbln.current_device):
                try:
                    call(); raise AssertionError("expected RuntimeError with no device")
                except RuntimeError:
                    pass
            m.empty_cache()  # graceful, must not raise
            assert m.memory_stats() == {{}} and m.memory_allocated() == 0
            try:
                t = torch.ones(4, dtype=torch.float16, device="rbln:0"); _ = t + t
                raise AssertionError("expected device use to fail")
            except Exception:
                pass
            print("FORCED_OK")
            """
        )
        env = dict(os.environ, RBLN_DEVICES="99999")
        env.pop("LOCAL_RANK", None)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=_PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=90
        )
        self.assertTrue(
            result.returncode == 0 and "FORCED_OK" in result.stdout,
            f"forced no-device contract failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )


if __name__ == "__main__":
    run_tests()
