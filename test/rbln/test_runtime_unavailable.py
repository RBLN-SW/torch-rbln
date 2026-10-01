# Owner(s): ["module: PrivateUse1"]

"""Device-runtime liveness gate: torch-rbln must degrade like ``torch.cuda`` when
the device runtime is absent or torn down, and must NEVER
segfault.

Background: torch-rbln links ``librbln_rt.so``, which loads the NPU driver
(``librbln-thunk.so``) on first use. The driver is missing on compile / CPU-only / CI
nodes, and at interpreter shutdown the runtime may already be torn down; a call into
either must fail cleanly or not happen, never crash. ``c10::rbln::runtime_available()`` is
the single source of truth that lets best-effort ops no-op, mandatory ops raise a
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


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in pyproject.
# The gate under test is torch-rbln's own: these tests never run against an absent runtime, they
# flip the process-wide shutting-down flag (and interpose librbln) to force it.
pytestmark = pytest.mark.torch_rbln_only


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


# rbln::runtime::Device::open(uint32_t), declared as librbln_rt.so exports it. torch-rbln
# opens an NPU only through it, so an LD_PRELOAD definition sees (or fails) every open.
_DEVICE_OPEN_DECLARATION = """
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
namespace rbln::runtime {
class Device {
 public:
  static std::shared_ptr<Device> open(uint32_t id);
};
}
using rbln::runtime::Device;
"""


def _build_device_open_shim(tmp_dir: str, definition: str):
    """Compile an LD_PRELOAD shim that defines ``Device::open`` as ``definition`` does, to
    observe or fail the NPU opens. Returns the .so path, or None without a C++ compiler."""
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        return None
    src = os.path.join(tmp_dir, "shim.cpp")
    so = os.path.join(tmp_dir, "shim.so")
    with open(src, "w") as handle:
        handle.write(_DEVICE_OPEN_DECLARATION + definition)
    build = subprocess.run(
        [cxx, "-std=c++17", "-shared", "-fPIC", "-o", so, src, "-ldl"], capture_output=True, text=True
    )
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

    def test_release_offload_temp_storage_no_ops_when_runtime_unavailable(self):
        """torch.rbln.release_offload_temp_storage() runs on the shutdown path, after the
        runtime may already be gone: it reports 0 files removed and never reaches the runtime."""
        result = _run_subprocess(
            """
            C._set_runtime_shutting_down(True)
            assert C.runtime_available() is False
            assert torch.rbln.release_offload_temp_storage() == 0
            print("RELEASE_OK")
            """
        )
        _assert_ok(self, result, "RELEASE_OK")

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
        torch.cuda on a host with no driver. DeviceMappingManager asks the runtime for a
        count only once rbln::runtime::Device::available() says the driver loads, so a missing
        runtime collapses into the well-tested no-device path. Skipped where
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
        live context (no allocation yet — the vLLM EngineCore parent) must not open the NPU,
        which would hold it against the process meant to use it. empty_cache/reset_*/stats
        are gated by the per-process context flag (initialized()/hasPrimaryContext()), so
        they no-op without opening anything. A shim records every Device::open; a real
        allocation at the end proves the shim sees the opens it is there to catch."""
        with tempfile.TemporaryDirectory() as tmp:
            so = _build_device_open_shim(
                tmp,
                """
std::shared_ptr<Device> Device::open(uint32_t id) {
  if (const char* marker = std::getenv("SHIM_MARKER")) {
    if (FILE* f = std::fopen(marker, "a")) std::fclose(f);
  }
  // librbln_rt.so is loaded RTLD_LOCAL under the extension, so RTLD_NEXT cannot reach it.
  using Open = std::shared_ptr<Device> (*)(uint32_t);
  static const auto real = reinterpret_cast<Open>(
      dlsym(dlopen("librbln_rt.so", RTLD_LAZY | RTLD_NOLOAD), "_ZN4rbln7runtime6Device4openEj"));
  if (real == nullptr) {
    std::fprintf(stderr, "shim: no Device::open in a loaded librbln_rt.so\n");
    std::abort();
  }
  return real(id);
}
""",
            )
            if so is None:
                self.skipTest("needs a C++ compiler to build the LD_PRELOAD shim")
            result = _run_subprocess(
                """
                import os
                marker = os.environ["SHIM_MARKER"]
                assert torch.rbln.device_count() > 0 and C.runtime_available() is True
                assert torch._C._accelerator_isAllocatorInitialized() is False, "no allocation yet -> not initialized"
                d = torch.device("rbln", 0)
                torch.accelerator.empty_cache()               # generic accelerator path (vLLM shutdown)
                torch.accelerator.reset_peak_memory_stats()
                torch.accelerator.reset_accumulated_memory_stats(0)
                C.empty_cache(d); C.reset_peak_memory_stats(d); C.reset_accumulated_memory_stats(d)  # direct C API too
                assert C.memory_stats(d) == {} and len(torch.accelerator.memory_stats(0)) == 0
                assert not os.path.exists(marker), "a best-effort op opened the NPU without a live context"
                torch.empty(4, dtype=torch.float16, device="rbln:0")
                assert os.path.exists(marker), "the shim did not see the allocation's NPU open"
                print("NOCTX_OK")
                """,
                env_extra={"LD_PRELOAD": so, "SHIM_MARKER": os.path.join(tmp, "opened")},
            )
            _assert_ok(self, result, "NOCTX_OK")

    @requires_physical_devices(1)
    def test_failed_commit_reports_unavailable(self):
        """A commit that fails part-way leaves the backend unusable, so is_available() must
        say so. The commit opens one NPU at a time and the runtime has no way to close one,
        so a failure mid-loop is permanent: every later device use rethrows the stored error.
        Reporting available while nothing can be used sends a caller -- vLLM picking a
        platform, LMCache picking a backend -- down a path that cannot work.

        The device count may stay: it describes the planned topology, not usability.
        Injected with a shim forcing Device::open to fail."""
        with tempfile.TemporaryDirectory() as tmp:
            so = _build_device_open_shim(
                tmp,
                """
std::shared_ptr<Device> Device::open(uint32_t) { throw std::runtime_error("injected open failure"); }
""",
            )
            if so is None:
                self.skipTest("needs a C++ compiler to build the LD_PRELOAD shim")
            result = _run_subprocess(
                """
                assert torch.rbln.is_available() is True, "available before any device use"
                for attempt in range(2):
                    try:
                        torch.ones(4, dtype=torch.float16, device="rbln:0")
                    except RuntimeError as exc:
                        assert "cannot open NPU" in str(exc) and "injected open failure" in str(exc), str(exc)
                    else:
                        raise AssertionError("a failing open must surface at every point of use")
                    assert torch.rbln.is_available() is False, "unusable backend still reports available"
                    assert C.runtime_available() is False, "python and C++ availability disagree"
                assert torch.rbln.device_count() > 0
                print("FAILED_COMMIT_OK")
                """,
                env_extra={"LD_PRELOAD": so},
            )
            _assert_ok(self, result, "FAILED_COMMIT_OK")

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
        """With a device present, the runtime loaded, and this process having allocated,
        best-effort ops actually run (not gated off)."""
        result = _run_subprocess(
            """
            assert torch.rbln.device_count() > 0
            assert C.runtime_available() is True, "healthy host with a device must be available"
            assert torch.rbln.is_available() is True
            t = torch.ones(64, device="rbln:0"); _ = (t + t).sum().item()   # establish a live context
            d = torch.device("rbln", 0)
            C.empty_cache(d)  # real flush, must not raise
            assert isinstance(C.memory_stats(d), dict) and len(C.memory_stats(d)) > 0
            print("HEALTHY_OK")
            """
        )
        _assert_ok(self, result, "HEALTHY_OK")

    @requires_physical_devices(1)
    def test_first_allocation_initializes_context(self):
        """A single torch.empty() takes its device memory at once, which marks the process
        initialized; the best-effort ops then run and the stats count the block."""
        result = _run_subprocess(
            """
            assert torch._C._accelerator_isAllocatorInitialized() is False
            t = torch.empty(1, device="rbln:0")
            assert torch._C._accelerator_isAllocatorInitialized() is True, "an allocation must initialize the context"
            torch.accelerator.empty_cache()
            torch.accelerator.reset_peak_memory_stats()
            stats = torch.accelerator.memory.memory_stats(0)
            assert stats["allocated_bytes.all.current"] >= t.nbytes > 0, stats
            print("FIRST_ALLOC_OK")
            """
        )
        _assert_ok(self, result, "FIRST_ALLOC_OK")


if __name__ == "__main__":
    run_tests()
