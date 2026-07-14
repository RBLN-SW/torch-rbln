# Owner(s): ["module: PrivateUse1"]

"""Device-runtime liveness gate: torch-rbln must degrade like ``torch.cuda`` when
the device runtime is absent or torn down, and must NEVER
segfault.

Background: torch-rbln links ``librbln.so``, which loads the device runtime lazily.
When the runtime is missing (compile / CPU-only / CI nodes) or has been unmapped at
interpreter shutdown, a raw ``rbln_*`` call dereferences a null handle and SEGFAULTs
-- unlike CUDA, where a missing driver merely returns an error code. ``c10::rbln::runtime_available()`` is the
single source of truth that lets best-effort ops no-op, mandatory ops raise a
clean error, and availability probes return False without raising.

These tests exercise the whole gate WITHOUT an actually-absent runtime by flipping the
process-wide "shutting down" flag (``_set_runtime_shutting_down``), which forces
``runtime_available()`` to False. That makes the contract testable on any host,
with or without an NPU. Each test runs in a fresh subprocess so the process-wide
flag never leaks into other tests.
"""

import os
import shutil
import subprocess
import sys
import tempfile
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


def _build_ld_preload_shim(tmp_dir: str, c_source: str):
    """Compile a tiny LD_PRELOAD shim that interposes librbln's C-linkage entry points
    (used to force / observe the runtime calls). Returns the .so path, or None if no C
    compiler is available (test skips)."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    src = os.path.join(tmp_dir, "shim.c")
    so = os.path.join(tmp_dir, "shim.so")
    with open(src, "w") as handle:
        handle.write(c_source)
    build = subprocess.run([cc, "-shared", "-fPIC", "-o", so, src], capture_output=True, text=True)
    return so if build.returncode == 0 else None


@pytest.mark.test_set_ci
class TestRuntimeUnavailable(TestCase):
    """``runtime_available()`` gates every runtime-touching leaf (torch.cuda parity)."""

    def test_bindings_exist_and_never_raise(self):
        """The liveness predicate and shutdown hook are exposed and total (nothrow)."""
        self.assertTrue(hasattr(torch_rbln._C, "runtime_available"))
        self.assertTrue(hasattr(torch_rbln._C, "runtime_loaded"))
        self.assertTrue(hasattr(torch_rbln._C, "_set_runtime_shutting_down"))
        self.assertIsInstance(torch_rbln._C.runtime_available(), bool)
        self.assertIsInstance(torch_rbln._C.runtime_loaded(), bool)
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
            # Best-effort leaves no-op; synchronize no-ops under teardown too (it throws
            # on a live no-device host -- see test_runtime_absent).
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

    def test_runtime_absent_degrades_to_zero_devices(self):
        """When the RBLN runtime is genuinely absent, device enumeration degrades to
        0 (nothrow) and is_available() is False -- never a SEGFAULT -- mirroring
        torch.cuda on a host with no driver. The fix is at the source
        (DeviceMappingManager gates the raw rbln_get_device_count() on rbln_runtime_available()),
        so a missing runtime collapses into the well-tested no-device path. Skipped where
        the runtime is present (e.g. device-bearing CI); the shutdown-flag tests above
        cover the torn-down half of the gate hardware-free."""
        if torch_rbln._C.runtime_loaded() or torch_rbln._C.is_dummy_device():
            self.skipTest("requires a host with the RBLN runtime absent")
        result = _run_subprocess(
            """
            assert C.runtime_loaded() is False
            assert torch.rbln.device_count() == 0, "runtime-absent must degrade to 0 devices, not segfault"
            assert torch.rbln.is_available() is False
            C.set_device_index(0)  # bookkeeping only: must not throw or segfault
            for use in (lambda: torch.empty(4, device="rbln:0"), lambda: C.synchronize(0)):
                try:
                    use()  # device use fails cleanly at the point of use (torch.cuda parity)
                    raise AssertionError("device use must raise with no device/runtime")
                except RuntimeError:
                    pass
            print("RUNTIME_ABSENT_OK")
            """
        )
        _assert_ok(self, result, "RUNTIME_ABSENT_OK")

        # Dummy is NOT exempt: it host-backs via the runtime (DeviceMappingManager's
        # rbln_register_device_id), so with the runtime absent enumeration must also
        # degrade to 0 -- not segfault -- at init, BEFORE any shutdown flag is set.
        # This covers the init/register path the flag-based tests cannot reach.
        result = _run_subprocess(
            """
            assert C.is_dummy_device() is True and C.runtime_loaded() is False
            assert torch.rbln.device_count() == 0, "dummy + absent runtime must degrade to 0, not segfault"
            assert torch.rbln.is_available() is False
            print("DUMMY_RUNTIME_ABSENT_OK")
            """,
            env_extra={"RBLN_DUMMY_DEVICE": "1"},
        )
        _assert_ok(self, result, "DUMMY_RUNTIME_ABSENT_OK")

    def test_dummy_with_runtime_proceeds(self):
        """Dummy mode (``RBLN_DUMMY_DEVICE``) delegates host-backing to the runtime, so
        with the runtime loaded, device ops proceed and materialize -- the
        gate passes rather than no-ops."""
        result = _run_subprocess(
            """
            assert C.is_dummy_device() is True
            assert C.runtime_available() is True, "dummy with a loaded runtime must be available"
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
        the runtime is still required. When the runtime is unavailable (simulated
        by the shutdown flag, standing in for a missing runtime), the gate fires --
        best-effort ops no-op and allocation raises a clean error, never a SEGFAULT."""
        result = _run_subprocess(
            """
            assert C.is_dummy_device() is True
            C._set_runtime_shutting_down(True)  # stand-in for an unavailable runtime (e.g. no runtime .so)
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
        allocation would otherwise reach the runtime."""
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
    def test_best_effort_ops_noop_without_live_context(self):
        """The reported regression: a process with the runtime + a device mapping but NO
        live context (no allocation yet — the vLLM EngineCore parent) must NOT dispatch
        the runtime call. empty_cache/reset_* are gated by the per-process context flag
        (initialized()/hasPrimaryContext()), so they no-op without fabricating a context.
        Proven with a shim that records whether rbln_empty_cache is actually invoked."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "called")
            # Shim records an invocation; if the gate works, it is never called.
            so = _build_ld_preload_shim(
                tmp,
                "#include <stdio.h>\n#include <stdlib.h>\n"
                'int rbln_empty_cache(int d){(void)d; const char*p=getenv("SHIM_MARKER");'
                ' if(p){FILE*f=fopen(p,"w"); if(f)fclose(f);} return 0;}\n',
            )
            if so is None:
                self.skipTest("needs a C compiler to build the LD_PRELOAD shim")
            result = _run_subprocess(
                """
                assert torch.rbln.device_count() > 0 and C.runtime_available() is True
                assert torch._C._accelerator_isAllocatorInitialized() is False, "no allocation yet -> not initialized"
                d = torch.device("rbln", 0)
                torch.accelerator.empty_cache()          # generic accelerator path (vLLM shutdown)
                torch.accelerator.reset_peak_memory_stats()
                C.empty_cache(d); C.reset_peak_memory_stats(d)   # direct C API too
                print("NOCTX_OK")
                """,
                env_extra={"LD_PRELOAD": so, "SHIM_MARKER": marker},
            )
            _assert_ok(self, result, "NOCTX_OK")
            self.assertFalse(
                os.path.exists(marker),
                "rbln_empty_cache was dispatched despite no live context — the context gate failed",
            )

    @requires_physical_devices(1)
    def test_best_effort_ops_propagate_live_context_failure(self):
        """Once this process has a live context (after an allocation), a genuine runtime
        failure in a best-effort op IS surfaced (CUDA parity — cudaFree failures in an
        initialized allocator propagate), not silently swallowed. Injected with an
        LD_PRELOAD shim forcing the C entry points to return non-zero."""
        with tempfile.TemporaryDirectory() as tmp:
            so = _build_ld_preload_shim(
                tmp,
                "int rbln_empty_cache(int d){(void)d;return 1;}\n"
                "int rbln_reset_peak_memory_stats(int d){(void)d;return 1;}\n"
                "int rbln_reset_accumulated_memory_stats(int d){(void)d;return 1;}\n",
            )
            if so is None:
                self.skipTest("needs a C compiler to build the LD_PRELOAD shim")
            result = _run_subprocess(
                """
                t = torch.ones(64, device="rbln:0"); _ = (t + t).sum().item()   # establish a live context
                d = torch.device("rbln", 0)
                ops = {
                    "empty_cache": lambda: torch.accelerator.empty_cache(),
                    "reset_peak": lambda: torch.accelerator.reset_peak_memory_stats(),
                    "reset_accum": lambda: torch.accelerator.reset_accumulated_memory_stats(0),
                    "C.empty_cache": lambda: C.empty_cache(d),
                    "C.reset_peak": lambda: C.reset_peak_memory_stats(d),
                    "C.reset_accum": lambda: C.reset_accumulated_memory_stats(d),
                }
                swallowed = []
                for name, fn in ops.items():
                    try:
                        fn(); swallowed.append(name)
                    except RuntimeError:
                        pass
                assert not swallowed, "runtime failure silently swallowed on a live context: " + ",".join(swallowed)
                print("PROPAGATE_OK")
                """,
                env_extra={"LD_PRELOAD": so},
            )
            _assert_ok(self, result, "PROPAGATE_OK")

    @requires_physical_devices(1)
    def test_memory_ops_nothrow_on_malformed_config(self):
        """The initialized()/hasPrimaryContext() predicates that gate torch.accelerator
        memory ops must stay total. A malformed RBLN_NPUS_PER_DEVICE makes the internal
        device-count lookup throw; the predicate swallows it and reports not-initialized,
        so empty_cache no-ops instead of aborting the caller. The misconfig still surfaces
        at real device use."""
        result = _run_subprocess(
            """
            assert torch._C._accelerator_isAllocatorInitialized() is False, "predicate must be nothrow + not-initialized"
            torch.accelerator.empty_cache()               # gated off -> no-op, must not raise
            torch.accelerator.reset_peak_memory_stats()
            try:
                torch.ones(4, device="rbln:0")
            except RuntimeError as exc:
                assert "valid sizes" in str(exc), str(exc)   # misconfig surfaces at real device use
            else:
                raise AssertionError("malformed config must raise at the point of real device use")
            print("MALFORMED_OK")
            """,
            env_extra={"RBLN_NPUS_PER_DEVICE": "3", "RBLN_DEVICE_MAP": None},
        )
        _assert_ok(self, result, "MALFORMED_OK")

    @requires_physical_devices(1)
    def test_dummy_malformed_config_memory_ops_noop(self):
        """Dummy mode must not short-circuit the context gate: with a malformed config and
        no allocation, the memory ops still no-op (not-initialized), and the misconfig
        surfaces only at real device use. Covered for both malformed-config env vars
        (RBLN_NPUS_PER_DEVICE and RBLN_DEVICE_MAP — same validateDeviceGroups path)."""
        script = """
            assert C.is_dummy_device() is True
            assert torch._C._accelerator_isAllocatorInitialized() is False
            torch.accelerator.empty_cache()               # no live context -> no-op, must not raise
            torch.accelerator.reset_peak_memory_stats()
            try:
                torch.zeros(4, device="rbln:0")
            except RuntimeError:
                pass
            else:
                raise AssertionError("dummy + malformed config must raise at real device use")
            print("DUMMY_MALFORMED_OK")
            """
        # A bad group size (3) via each of the two mapping env vars.
        for env_extra in (
            {"RBLN_DUMMY_DEVICE": "1", "RBLN_NPUS_PER_DEVICE": "3", "RBLN_DEVICE_MAP": None},
            {"RBLN_DUMMY_DEVICE": "1", "RBLN_DEVICE_MAP": "[0,1,2]", "RBLN_NPUS_PER_DEVICE": None},
        ):
            result = _run_subprocess(script, env_extra=env_extra)
            _assert_ok(self, result, "DUMMY_MALFORMED_OK")

    @requires_physical_devices(1)
    def test_runtime_available_true_on_healthy_host(self):
        """With a device present and the runtime loaded, runtime_available() is True and
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
