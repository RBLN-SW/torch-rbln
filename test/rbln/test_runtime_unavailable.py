# Owner(s): ["module: PrivateUse1"]

"""Device-runtime liveness gate: torch-rbln must degrade like ``torch.cuda`` when
the device runtime (``librbln-thunk.so``) is absent or torn down, and must NEVER
segfault.

Background: torch-rbln links ``librbln.so``, which lazily ``dlopen()``s
``librbln-thunk.so`` on the first device op. When the thunk is missing (compile /
CPU-only / CI nodes) or has been unmapped at interpreter shutdown, a raw thunk
call dereferences a null handle and SEGFAULTs -- unlike CUDA, where a missing
driver merely returns an error code. ``c10::rbln::runtime_available()`` is the
single source of truth that lets best-effort ops no-op, mandatory ops raise a
clean error, and availability probes return False without raising.

These tests exercise the whole gate WITHOUT removing the thunk by flipping the
process-wide "shutting down" flag (``_set_runtime_shutting_down``), which forces
``runtime_available()`` to False. That makes the contract testable on any host,
with or without an NPU. Each test runs in a fresh subprocess so the process-wide
flag never leaks into other tests.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from test.utils import requires_physical_devices


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_subprocess(body: str, timeout: int = 90, env_extra=None) -> subprocess.CompletedProcess:
    # Flat preamble (column 0) + dedented body, so the caller can pass an
    # indented triple-quoted block without breaking Python's indentation.
    preamble = f"import sys\nsys.path.insert(0, {_PROJECT_ROOT!r})\nimport torch, torch_rbln\nC = torch_rbln._C\n"
    script = preamble + textwrap.dedent(body)
    env = None
    if env_extra is not None:
        env = dict(os.environ)
        for key, value in env_extra.items():
            if value is None:
                env.pop(key, None)  # remove a var for a hermetic env
            else:
                env[key] = value
    return subprocess.run(
        [sys.executable, "-c", script], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, env=env
    )


def _assert_ok(self, result: subprocess.CompletedProcess, marker: str) -> None:
    self.assertTrue(
        result.returncode == 0 and marker in result.stdout,
        f"runtime-liveness contract failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )


@pytest.mark.test_set_ci
class TestRuntimeUnavailable(TestCase):
    """``runtime_available()`` gates every runtime-touching leaf (torch.cuda parity)."""

    def test_bindings_exist_and_never_raise(self):
        """The liveness predicate and shutdown hook are exposed and total (nothrow)."""
        self.assertTrue(hasattr(torch_rbln._C, "runtime_available"))
        self.assertTrue(hasattr(torch_rbln._C, "thunk_loadable"))
        self.assertTrue(hasattr(torch_rbln._C, "_set_runtime_shutting_down"))
        self.assertIsInstance(torch_rbln._C.runtime_available(), bool)
        self.assertIsInstance(torch_rbln._C.thunk_loadable(), bool)
        # device_count() / is_available() must never raise (CUDA contract).
        self.assertIsInstance(torch.rbln.device_count(), int)
        self.assertIsInstance(torch.rbln.is_available(), bool)

    def test_best_effort_ops_no_op_when_runtime_unavailable(self):
        """empty_cache / memory_stats / reset_* / synchronize no-op (never segfault)
        when the runtime is unavailable -- both the RBLN-direct leaves and the
        generic torch.accelerator surface (incl. reset_peak, which has no Python
        init-guard, so the C++ leaf gate is the only thing protecting it)."""
        result = _run_subprocess(
            """
            C._set_runtime_shutting_down(True)
            assert C.runtime_available() is False, "shutdown flag must force runtime_available False"
            # is_available() reflects runtime liveness, so it is False once shutting
            # down -- in dummy mode too (dummy is not exempt from the gate).
            assert torch.rbln.is_available() is False
            assert isinstance(torch.rbln.device_count(), int)  # still nothrow

            d = torch.device("rbln", 0)
            # RBLN-direct best-effort leaves: no-op, must not raise or crash.
            C.empty_cache(d)
            C.synchronize(0)
            C.reset_peak_memory_stats(d)
            C.reset_accumulated_memory_stats(d)
            assert C.memory_stats(d) == {}, "memory_stats must be empty when unavailable"

            # Generic torch.accelerator surface (the vLLM shutdown path). reset_peak
            # has NO Python init-guard, so reaching it proves the leaf gate works.
            torch.accelerator.empty_cache()
            torch.accelerator.reset_peak_memory_stats()

            print("GATE_OK")
            """
        )
        _assert_ok(self, result, "GATE_OK")

    def test_file_offloading_no_ops_when_runtime_unavailable(self):
        """set_file_offloading_enabled (torch.rbln.offload()) is a global toggle
        reachable with NO prior allocation -- unlike the copy/borrow leaves, which
        are downstream of a gated malloc -- so it is gated directly: no-op, never a
        SEGFAULT, when the runtime is unavailable."""
        result = _run_subprocess(
            """
            C._set_runtime_shutting_down(True)
            assert C.runtime_available() is False
            C._set_file_offloading_enabled(True)   # best-effort: no-op, must not raise/crash
            C._set_file_offloading_enabled(False)
            print("OFFLOAD_OK")
            """
        )
        _assert_ok(self, result, "OFFLOAD_OK")

    def test_flag_toggles_runtime_available(self):
        """Setting / clearing the shutdown flag flips runtime_available() and restores it."""
        result = _run_subprocess(
            """
            before = C.runtime_available()
            C._set_runtime_shutting_down(True)
            assert C.runtime_available() is False
            C._set_runtime_shutting_down(False)
            assert C.runtime_available() == before, "runtime_available must restore after clearing the flag"
            print("TOGGLE_OK")
            """
        )
        _assert_ok(self, result, "TOGGLE_OK")

    def test_thunk_absent_degrades_to_zero_devices(self):
        """When librbln-thunk.so is genuinely absent, device enumeration degrades to
        0 (nothrow) and is_available() is False -- never a SEGFAULT -- mirroring
        torch.cuda on a host with no driver. The fix is at the source
        (DeviceMappingManager gates the raw rbln_get_device_count() on thunk_loadable),
        so thunk-absent collapses into the well-tested no-device path. Skipped where
        the thunk is present (e.g. device-bearing CI); the shutdown-flag tests above
        cover the torn-down half of the gate hardware-free."""
        if torch_rbln._C.thunk_loadable() or torch_rbln._C.is_dummy_device():
            self.skipTest("requires a host with librbln-thunk.so absent")
        result = _run_subprocess(
            """
            assert C.thunk_loadable() is False
            assert torch.rbln.device_count() == 0, "thunk-absent must degrade to 0 devices, not segfault"
            assert torch.rbln.is_available() is False
            C.set_device_index(0)  # bookkeeping only: must not throw or segfault
            try:
                torch.empty(4, device="rbln:0")  # use fails cleanly at the point of use
                raise AssertionError("allocation must raise with no device")
            except RuntimeError:
                pass
            print("THUNK_ABSENT_OK")
            """
        )
        _assert_ok(self, result, "THUNK_ABSENT_OK")

    def test_dummy_with_runtime_proceeds(self):
        """Dummy mode (``RBLN_DUMMY_DEVICE``) delegates host-backing to the runtime, so
        with a loadable ``librbln-thunk.so`` device ops proceed and materialize -- the
        gate passes rather than no-ops."""
        result = _run_subprocess(
            """
            assert C.is_dummy_device() is True
            assert C.runtime_available() is True, "dummy with a loadable thunk must be available"
            assert torch.rbln.is_available() is True
            t = torch.zeros(4, device="rbln:0")
            assert t.cpu().tolist() == [0.0, 0.0, 0.0, 0.0]
            print("DUMMY_PROCEEDS_OK")
            """,
            env_extra={"RBLN_DUMMY_DEVICE": "1", "RBLN_DEVICE_MAP": None, "RBLN_NPUS_PER_DEVICE": None},
        )
        _assert_ok(self, result, "DUMMY_PROCEEDS_OK")

    def test_dummy_without_runtime_is_gated(self):
        """Dummy mode is NOT exempt from the gate: it host-backs via the runtime, so
        librbln-thunk.so is still required. When the runtime is unavailable (simulated
        by the shutdown flag, standing in for a missing thunk), the gate fires --
        best-effort ops no-op and allocation raises a clean error, never a SEGFAULT."""
        result = _run_subprocess(
            """
            assert C.is_dummy_device() is True
            C._set_runtime_shutting_down(True)  # stand-in for an unavailable runtime (e.g. no thunk)
            assert C.runtime_available() is False, "dummy must not bypass the runtime gate"
            assert torch.rbln.is_available() is False
            d = torch.device("rbln", 0)
            C.empty_cache(d); C.synchronize(0)  # best-effort: no-op, no crash
            try:
                torch.zeros(4, device="rbln:0")
                raise AssertionError("allocation must raise when the runtime is unavailable in dummy")
            except RuntimeError:
                pass
            print("DUMMY_GATED_OK")
            """,
            env_extra={"RBLN_DUMMY_DEVICE": "1", "RBLN_DEVICE_MAP": None, "RBLN_NPUS_PER_DEVICE": None},
        )
        _assert_ok(self, result, "DUMMY_GATED_OK")

    @requires_physical_devices(1)
    def test_mandatory_op_raises_clean_error_not_segfault(self):
        """Allocation is a mandatory op: when the runtime is unavailable it must raise
        a clean, catchable RuntimeError (not SEGFAULT). Needs a real device so the
        allocation would otherwise reach the thunk."""
        result = _run_subprocess(
            """
            assert torch.rbln.device_count() > 0 and not C.is_dummy_device()
            C._set_runtime_shutting_down(True)
            try:
                torch.empty(4, device="rbln:0")
                raise AssertionError("allocation must raise when the runtime is unavailable")
            except RuntimeError as e:
                assert "runtime" in str(e).lower(), str(e)
            print("MALLOC_OK")
            """
        )
        _assert_ok(self, result, "MALLOC_OK")

    @requires_physical_devices(1)
    def test_runtime_available_true_on_healthy_host(self):
        """With a device present and the thunk loaded, runtime_available() is True and
        best-effort ops actually run (not gated off)."""
        result = _run_subprocess(
            """
            assert torch.rbln.device_count() > 0
            assert C.runtime_available() is True, "healthy host with a device must be available"
            assert torch.rbln.is_available() is True
            d = torch.device("rbln", 0)
            C.empty_cache(d)  # real flush, must not raise
            assert isinstance(C.memory_stats(d), dict)
            print("HEALTHY_OK")
            """
        )
        _assert_ok(self, result, "HEALTHY_OK")


if __name__ == "__main__":
    run_tests()
