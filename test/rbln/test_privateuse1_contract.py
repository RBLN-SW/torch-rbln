# Owner(s): ["module: PrivateUse1"]

"""PrivateUse1 backend contract conformance.

torch does not merely *offer* the ``torch.rbln`` module and the RBLN accelerator
hooks -- it *calls into them*, from paths that have nothing to do with wanting an
NPU: ``DataLoader(pin_memory=True)``, ``torch.load(map_location=...)``, importing
``torch.testing._internal.common_utils``, ``torch._utils._get_available_device_type()``.
Those call sites assume a contract that upstream states explicitly.

Every test here pins exactly ONE clause and cites its upstream source, so that a
torch upgrade or a newly-discovered call site fails on the clause rather than on
a downstream symptom. Historically each violation was found and patched one call
site at a time (#100, #107, #120, #130, #151); this file exists so that stops.

Three properties are measured per probe, each in a fresh subprocess (the runtime
state involved is process-global and one-shot):

``raised``  the probe propagated an exception
``ctx``     the probe opened an NPU context (``rbln-stat`` reports this pid)
``sealed``  the probe sealed ``RBLN_DEVICES`` (a later remap is rejected)

``sealed`` is load-bearing for vLLM: ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to
``fork`` and ``RBLNWorker._init_device_env()`` remaps ``RBLN_DEVICES`` *inside* the
forked worker. A seal taken in the parent is inherited across fork and breaks every
worker. torch-rbln #151 was one instance; a co-tenant availability probe (LMCache
imports ``lmcache.v1.platform`` on every start) is another.

A clause not satisfied yet is marked ``strict=True`` xfail naming the work that closes
it; an unexpected pass means the marker should be removed. Two groups remain:

- ``fsw-inference#475`` -- the runtime seals ``RBLN_DEVICES`` on its first
  device-resolving call, so any availability query freezes it for this process and every
  process it forks. Not fixable from this layer.
- ``phase 4`` -- the RNG / serialization / device-name surface of ``torch.rbln``.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import requires_physical_devices


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- scenarios ---------------------------------------------------------------
# Each is an ``RBLN_*`` environment a real deployment can be in. A contract clause
# has to hold in all of them, not just on a healthy host.
HEALTHY = {"RBLN_DEVICES": "0"}
NO_DEVICE = {"RBLN_DEVICES": "99999"}  # visible-device filter matches nothing
BAD_MAP = {"RBLN_DEVICES": "0", "RBLN_DEVICE_MAP": "[1],[99]"}  # map exceeds visible NPUs
DUMMY = {"RBLN_DUMMY_DEVICE": "1"}
DUMMY_BAD_MAP = {"RBLN_DUMMY_DEVICE": "1", "RBLN_DEVICE_MAP": "[0,1,1]"}  # group size 3 invalid

ALL_SCENARIOS = {
    "healthy": HEALTHY,
    "no_device": NO_DEVICE,
    "bad_map": BAD_MAP,
    "dummy": DUMMY,
    "dummy_bad_map": DUMMY_BAD_MAP,
}

# Probes must assign ``rec["result"]``. They run after ``import torch, torch_rbln``.
_RUNNER = '''\
import json, os, shutil, subprocess, sys

sys.path.insert(0, {root!r})
{env}
os.environ.setdefault("RBLN_LOG_LEVEL", "off")

# The probe may print; only this file descriptor carries the JSON result.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

rec = {{"result": None, "raised": None, "ctx": None, "sealed": None}}

import torch          # noqa: F401
import torch_rbln     # noqa: F401


def _ctx_opened():
    """True if THIS pid holds an NPU context. None when rbln-stat is unavailable."""
    if shutil.which("rbln-stat") is None:
        return None
    try:
        out = subprocess.run(["rbln-stat"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    return any(" {{}} ".format(os.getpid()) in line for line in out.splitlines())


def _sealed():
    """True if RBLN_DEVICES is already sealed, i.e. a later remap is rejected.

    Remaps RBLN_DEVICES and then makes a device-resolving call, exactly as a vLLM DP
    worker does after fork. The call goes through rebel (the runtime that owns the seal)
    rather than a torch_rbln API, so this stays a measurement of the runtime rather than
    of our own sealing behavior -- a torch_rbln entry point that stopped sealing would
    otherwise make these checks pass for the wrong reason. Must run last: it seals.
    """
    cur = os.environ.get("RBLN_DEVICES")
    os.environ["RBLN_DEVICES"] = "0" if cur == "0,1" else "0,1"
    try:
        from rebel._C import get_npu_name

        get_npu_name(0)
        return False
    except BaseException as e:
        msg = str(e)
        if "Sealed" in msg or "changed at runtime" in msg:
            return True
        return None  # failed for an unrelated reason: undetermined


try:
{probe}
except BaseException as e:  # noqa: BLE001 - the point is to characterise failures
    rec["raised"] = "{{}}: {{}}".format(type(e).__name__, str(e).splitlines()[0][:160])

rec["ctx"] = _ctx_opened()      # before _sealed(): that call itself seals
rec["sealed"] = _sealed()
print(json.dumps(rec), file=_real_stdout)
'''


class Probe:
    """Outcome of running one probe in a fresh process."""

    def __init__(self, rec, stdout, stderr):
        self.result = rec["result"]
        self.raised = rec["raised"]
        self.ctx = rec["ctx"]
        self.sealed = rec["sealed"]
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        return f"Probe(result={self.result!r}, raised={self.raised!r}, ctx={self.ctx!r}, sealed={self.sealed!r})"

    @property
    def console(self):
        """Everything the probe wrote to a console.

        ``RBLN_CHECK`` logs to **stdout**, not stderr, so a check that only inspects
        stderr silently passes. Both streams are inspected here.
        """
        return self.stdout + self.stderr

    @property
    def has_cpp_traceback(self):
        """A caught exception must not leave a C++ stack trace in a co-tenant's log."""
        return "frame #0:" in self.console


def run_probe(body: str, env: dict) -> Probe:
    """Run ``body`` in a fresh interpreter under ``env`` and report what it did."""
    env_src = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in env.items())
    script = _RUNNER.format(
        root=_PROJECT_ROOT,
        env=env_src,
        probe=textwrap.indent(textwrap.dedent(body).strip("\n"), "    "),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The C++ logger shares stdout with the result channel, so scan back for the
    # last line that is actually the JSON record.
    for line in reversed(proc.stdout.splitlines()):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "sealed" in rec:
            return Probe(rec, proc.stdout, proc.stderr)
    raise AssertionError(
        "probe harness produced no result (the child died before reporting)\n"
        f"--- rc={proc.returncode} stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def xfail_until(owner: str, reason: str):
    """Strict xfail for a clause that is not satisfied yet.

    ``owner`` names the work that closes it. Strict, so an unexpected pass fails the suite
    and signals the marker should be removed.
    """
    return pytest.mark.xfail(strict=True, reason=f"[{owner}] {reason}")


@pytest.mark.test_set_ci
@requires_physical_devices(1)
class TestProbeHarness(TestCase):
    """Positive controls: prove the harness can actually observe a violation.

    Every clause below is expressed as "the probe did NOT do X". Such a test passes both
    when the contract holds and when the harness has gone blind, so each of the three
    measurements needs a case that deliberately triggers it. This is not hypothetical --
    while this file was being written, ``has_cpp_traceback`` inspected only stderr while
    ``RBLN_CHECK`` logs to stdout, and every "no traceback" assertion passed for free.
    """

    def test_detects_a_raise(self):
        """A probe that raises must be reported in ``raised``."""
        p = run_probe('raise RuntimeError("deliberate")', HEALTHY)
        self.assertIsNotNone(p.raised, f"harness missed a raise -- {p}")
        self.assertIn("deliberate", p.raised, f"harness lost the message -- {p}")

    def test_detects_a_device_context(self):
        """A probe that allocates must be reported in ``ctx``."""
        p = run_probe(
            'rec["result"] = float(torch.ones(4, dtype=torch.float16, device="rbln:0").sum().cpu())',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"allocation failed, cannot control for ctx -- {p}")
        if p.ctx is None:
            self.skipTest("rbln-stat unavailable; cannot observe device contexts")
        self.assertTrue(p.ctx, f"harness did not see the NPU context an allocation opens -- {p}")

    def test_detects_a_seal(self):
        """A probe that resolves a device through the runtime must be reported in ``sealed``."""
        p = run_probe(
            """
            from rebel._C import get_npu_name

            rec["result"] = get_npu_name(0)
            """,
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised, cannot control for seal -- {p}")
        self.assertTrue(p.sealed, f"harness did not see RBLN_DEVICES get sealed -- {p}")

    def test_detects_a_cpp_traceback(self):
        """A failure from a still-loud ``RBLN_CHECK`` must be seen by ``has_cpp_traceback``.

        Selecting an unassigned device index goes through ``check_device_index()``, which
        kept plain ``RBLN_CHECK`` -- it is a use error, not something a probe walks into --
        so it logs ``c10::Error::what()`` and the stack trace it embeds.

        Note what this control does *not* use: a malformed-config failure. Those are
        rethrown from the plan via ``RBLN_CHECK_QUIET``, so they now carry the detailed
        message without a logged trace even at the point of use. That is intended -- the
        exception reaches Python and gets printed there, and
        ``get_device_count_nothrow()`` warns once for callers that swallow it -- but it
        means a config error is the wrong thing to control with.
        """
        p = run_probe(
            """
            try:
                torch.ones(4, dtype=torch.float16, device="rbln:99")
                rec["result"] = "NO-RAISE"
            except Exception:
                rec["result"] = "raised"
            """,
            HEALTHY,
        )
        self.assertEqual(p.result, "raised", f"rbln:99 should not be usable -- {p}")
        self.assertTrue(p.has_cpp_traceback, f"harness cannot see a C++ traceback -- {p.console}")

    def test_scenarios_are_live(self):
        """Each scenario must actually change what the backend reports.

        Without this, a regression that ignored ``RBLN_DEVICE_MAP`` would make every
        "never throws" clause pass vacuously -- nothing would be misconfigured any more.
        """
        expected = {
            "healthy": 1,  # RBLN_DEVICES="0" -> exactly one logical device
            "no_device": 0,  # filter matches nothing
            "bad_map": 0,  # malformed map -> quiet 0, detail at point of use
            "dummy": 1,  # one host-backed logical device
            "dummy_bad_map": 0,  # invalid group size -> quiet 0
        }
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch.rbln.device_count()', env)
                self.assertIsNone(p.raised, f"{name}: probe raised -- {p}")
                self.assertEqual(p.result, expected[name], f"{name}: scenario is not live -- {p}")


@pytest.mark.test_set_ci
@requires_physical_devices(1)
class TestUpstreamClauses(TestCase):
    """One test per stated upstream requirement, cited in the docstring."""

    # -- availability must never throw -------------------------------------

    def test_is_available_never_throws(self):
        """``torch.rbln.is_available()`` must never raise.

        torch/xpu/__init__.py::is_available -- "This function never throws."
        torch/cuda/__init__.py::is_available -- "The default availability inspection
        never throws and returns 0 if the driver is missing or can't be initialized."

        torch calls this from import-time code (``TEST_PRIVATEUSE1`` in
        torch/testing/_internal/common_utils.py) and from CPU-only paths
        (``DataLoader(pin_memory=True)``), so a raise breaks unrelated callers.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch.rbln.is_available()', env)
                self.assertIsNone(p.raised, f"{name}: is_available() raised -- {p}")

    def test_device_count_never_throws(self):
        """``torch.rbln.device_count()`` must never raise.

        c10/cuda/CUDAFunctions.h:19-27 -- "people basically ~never want this function
        to fail; it should just return zero if things are not working. Oblige them."
        ``device_count() noexcept`` with a separately named throwing variant
        ``device_count_ensure_non_zero()``.
        ATen/DeviceAccelerator.h:50 -- deviceCount() "is *REQUIRED* to not raise any
        exception."
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch.rbln.device_count()', env)
                self.assertIsNone(p.raised, f"{name}: device_count() raised -- {p}")

    def test_is_initialized_never_throws(self):
        """``torch.rbln.is_initialized()`` must never raise.

        ``torch.distributed``'s ``init_device_mesh`` consults it to decide whether to
        auto-select a per-rank device; a raise there aborts mesh construction.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch.rbln.is_initialized()', env)
                self.assertIsNone(p.raised, f"{name}: is_initialized() raised -- {p}")

    # -- availability must not touch the device ----------------------------

    def test_is_available_opens_no_device_context(self):
        """``is_available()`` must not initialize a context on any device.

        ATen/detail/AcceleratorHooksInterface.h:33-38 -- isAvailable() "should NOT
        initialize the context on any device (result of hasPrimaryContext below
        should not change)."

        ``rbln_register_device_id()`` is documented as "Initializes devices to be used
        for NPU executions", so calling it from an availability query claims hardware
        a co-tenant may need.
        """
        p = run_probe('rec["result"] = torch.rbln.is_available()', HEALTHY)
        if p.ctx is None:
            self.skipTest("rbln-stat unavailable; cannot observe device contexts")
        self.assertFalse(p.ctx, f"is_available() opened an NPU context -- {p}")

    def test_device_count_opens_no_device_context(self):
        """``device_count()`` must not initialize a context on any device.

        Same clause as :meth:`test_is_available_opens_no_device_context`; enumeration
        and availability share the code path that claims devices.
        """
        p = run_probe('rec["result"] = torch.rbln.device_count()', HEALTHY)
        if p.ctx is None:
            self.skipTest("rbln-stat unavailable; cannot observe device contexts")
        self.assertFalse(p.ctx, f"device_count() opened an NPU context -- {p}")

    # -- the accelerator hooks must agree with the module ------------------

    def test_hooks_device_count_agrees_with_module(self):
        """The C++ accelerator hooks must report the same device count as the module.

        ATen/detail/AcceleratorHooksInterface.h -- ``deviceCount()`` defaults to 0;
        a backend that does not override it reports "no devices" to every C++ consumer
        while ``torch.rbln.device_count()`` says otherwise.
        """
        p = run_probe(
            'rec["result"] = [torch._C._accelerator_hooks_device_count(), torch.rbln.device_count()]',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        hooks_count, module_count = p.result
        self.assertEqual(hooks_count, module_count, f"hooks disagree with module -- {p}")

    def test_hooks_get_current_device_is_implemented(self):
        """The C++ accelerator hooks must answer ``getCurrentDevice()``.

        ATen/detail/AcceleratorHooksInterface.h -- the default implementation is
        ``TORCH_CHECK(false, "Backend doesn't support getCurrentDevice()")``.
        RBLNGuardImpl already implements the equivalent, so this is pure delegation.
        """
        p = run_probe('rec["result"] = torch._C._accelerator_hooks_get_current_device()', HEALTHY)
        self.assertIsNone(p.raised, f"hooks getCurrentDevice() raised -- {p}")

    def test_python_and_cpp_availability_agree(self):
        """``torch.rbln.is_available()`` and the C++ predicate must never disagree.

        ``RBLNHooksInterface::hasRBLN()`` answers from ``c10::rbln::runtime_available()``
        while the python entry point computes its own answer. Two sources of truth for
        one question means torch's C++ paths and its python paths can diverge.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe(
                    """
                    py = None
                    try:
                        py = torch.rbln.is_available()
                    except BaseException:
                        py = "RAISED"
                    rec["result"] = [py, torch_rbln._C.runtime_available()]
                    """,
                    env,
                )
                self.assertIsNone(p.raised, f"{name}: probe raised -- {p}")
                py_answer, cpp_answer = p.result
                self.assertEqual(py_answer, cpp_answer, f"{name}: split brain -- {p}")

    # -- torch's own entry points ------------------------------------------

    def test_torch_accelerator_apis_never_throw(self):
        """``torch.accelerator.is_available()`` / ``device_count()`` must never raise.

        torch/accelerator/__init__.py delegates to the device module. DataLoader calls
        ``torch.accelerator.is_available()`` for ``pin_memory`` (torch/utils/data/
        dataloader.py:672,681), so a raise breaks a CPU-only DataLoader.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe(
                    'rec["result"] = [torch.accelerator.is_available(), torch.accelerator.device_count()]',
                    env,
                )
                self.assertIsNone(p.raised, f"{name}: torch.accelerator raised -- {p}")

    def test_get_available_device_type_never_throws(self):
        """``torch._utils._get_available_device_type()`` must never raise.

        torch/_utils.py:799 calls ``custom_device_mod.is_available()``. It backs
        ``_get_device_attr`` / ``_get_all_device_indices``, used well outside RBLN code.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch._utils._get_available_device_type()', env)
                self.assertIsNone(p.raised, f"{name}: raised -- {p}")

    def test_common_utils_is_importable(self):
        """``import torch.testing._internal.common_utils`` must not raise.

        torch/testing/_internal/common_utils.py:1522 evaluates
        ``TEST_PRIVATEUSE1 = is_privateuse1_backend_available()`` at module scope,
        which calls ``torch.rbln.is_available()``. A raise makes torch's own test
        utilities unimportable.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe(
                    """
                    from torch.testing._internal.common_utils import TEST_PRIVATEUSE1
                    rec["result"] = TEST_PRIVATEUSE1
                    """,
                    env,
                )
                self.assertIsNone(p.raised, f"{name}: import raised -- {p}")

    # -- backend-module surface torch looks for ------------------------------

    @xfail_until("phase 4", "torch.rbln has no _is_in_bad_fork / manual_seed_all")
    def test_manual_seed_reaches_the_backend(self):
        """``torch.manual_seed()`` must seed the RBLN generators.

        torch/random.py::_seed_custom_device requires ``_is_in_bad_fork`` **and**
        ``manual_seed_all`` on the device module; without both it warns and silently does
        nothing, so RBLN results are not reproducible from ``torch.manual_seed()``.
        Listed as a required backend API in torch/utils/backend_registration.py:44-63.
        """
        p = run_probe(
            """
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                torch.manual_seed(0)
            rec["result"] = {
                # torch/random.py needs BOTH of these, and silently no-ops without either.
                "missing": [n for n in ("_is_in_bad_fork", "manual_seed_all") if not hasattr(torch.rbln, n)],
                "warnings": [str(w.message) for w in caught],
            }
            """,
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        # Asserted structurally as well as on the warning text: torch could reword the
        # message at any release, and a text-only check would then pass for free.
        self.assertEqual(p.result["missing"], [], f"missing from torch.rbln: {p.result['missing']}")
        offenders = [message for message in p.result["warnings"] if "does not take effect" in message]
        self.assertEqual(offenders, [], f"torch.manual_seed() does not reach RBLN -- {offenders}")

    @xfail_until("phase 4", "torch.rbln has no _utils._get_device_index")
    def test_serialization_device_index_helper_exists(self):
        """``torch.rbln._utils._get_device_index`` must exist for ``torch.load``.

        torch/serialization.py:606-628 documents it as required of a privateuse1 backend
        ("Implement the following methods in device_module like cuda:
        device_module._utils._get_device_index(location, True), device_module.device_count()")
        and uses it to resolve ``map_location="rbln:N"``. Without it torch falls back to a
        looser path that cannot honour a backend-specific index normalisation.
        """
        p = run_probe(
            'rec["result"] = hasattr(getattr(torch.rbln, "_utils", None), "_get_device_index")',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        self.assertTrue(p.result, "torch.rbln._utils._get_device_index is missing")

    @xfail_until("phase 4", "torch.rbln has no get_device_name/properties/capability")
    def test_device_name_apis_exist(self):
        """``torch.rbln.get_device_name()`` and friends must exist.

        ``torch.cuda`` / ``torch.xpu`` both expose ``get_device_name``,
        ``get_device_properties`` and ``get_device_capability``, and frameworks reach for
        them through the device module: vLLM's XPU platform is
        ``return torch.xpu.get_device_name(device_id)`` (vllm/platforms/xpu.py:137).
        With no RBLN equivalent, vllm-rbln calls ``rebel.get_npu_name()`` directly --
        bypassing torch and sealing RBLN_DEVICES in whatever process asks.
        ``RBLNGuardImpl::getDeviceCapability()`` already exists on the C++ side.
        """
        p = run_probe(
            'rec["result"] = [n for n in ("get_device_name", "get_device_properties",'
            ' "get_device_capability") if not hasattr(torch.rbln, n)]',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        self.assertEqual(p.result, [], f"missing from torch.rbln: {p.result}")

    # -- the counterpart clause: point of use must stay loud ----------------

    def test_point_of_use_still_raises_with_detail(self):
        """Quiet availability must not cost the diagnostic; device *use* must be loud.

        c10/cuda/CUDAFunctions.h keeps a throwing ``device_count_ensure_non_zero()``
        alongside the ``noexcept`` query. Making availability quiet is only correct if
        the detailed config error still surfaces where the user actually needs a device.
        """
        p = run_probe(
            """
            try:
                torch.ones(4, dtype=torch.float16, device="rbln:0")
                rec["result"] = "NO-RAISE"
            except Exception as e:
                rec["result"] = str(e).splitlines()[0]
            """,
            BAD_MAP,
        )
        self.assertIsNone(p.raised, f"probe harness error -- {p}")
        self.assertNotEqual(p.result, "NO-RAISE", f"device use must fail -- {p}")
        self.assertIn("out of range", p.result.lower(), f"error lost its detail -- {p}")

    def test_availability_emits_no_cpp_traceback(self):
        """A swallowed availability failure must not spam a co-tenant's console.

        ``RBLN_CHECK`` (c10/rbln/RBLNLogging.h:181-191) logs ``c10::Error::what()`` --
        which embeds the C++ stack trace -- before throwing. Measured: it writes 16
        lines to **stdout** (not stderr) and ``RBLN_LOG_LEVEL=off`` does not suppress
        it, so catching the exception in python leaves the noise in place.
        """
        p = run_probe(
            """
            try:
                rec["result"] = torch.rbln.is_available()
            except Exception:
                rec["result"] = False
            """,
            BAD_MAP,
        )
        self.assertFalse(p.has_cpp_traceback, f"C++ traceback leaked to the console:\n{p.console}")


@pytest.mark.test_set_ci
@requires_physical_devices(1)
class TestExternalConsumers(TestCase):
    """Scenarios reproducing how vLLM and LMCache actually drive this backend.

    These are not hypotheticals: each corresponds to a reported failure.
    """

    @xfail_until("fsw-inference#475", "the runtime seals RBLN_DEVICES on its first device-resolving call")
    def test_cotenant_availability_probe_does_not_seal(self):
        """A co-tenant availability probe must leave ``RBLN_DEVICES`` remappable.

        LMCache runs device detection while importing ``lmcache.v1.platform``, on every
        start, in whatever process imports it -- including a vLLM parent that has not
        yet forked its workers. Sealing there is inherited by every worker.
        """
        p = run_probe('rec["result"] = torch.rbln.is_available()', HEALTHY)
        if p.sealed is None:
            self.skipTest("seal state undetermined in this environment")
        self.assertFalse(p.sealed, f"availability probe sealed RBLN_DEVICES -- {p}")

    @xfail_until("fsw-inference#475", "the runtime seals RBLN_DEVICES on its first device-resolving call")
    def test_fork_then_worker_remap_succeeds(self):
        """A forked worker must still be able to remap ``RBLN_DEVICES``.

        ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to ``fork`` (vllm/envs.py:742) and
        ``RBLNWorker._init_device_env()`` assigns ``os.environ[RBLN_DEVICES]`` inside the
        forked worker. The seal is inherited across fork, so a probe in the parent
        breaks every worker deterministically. torch-rbln #151 was this bug via a
        different entry point.
        """
        p = run_probe(
            """
            import select
            torch.rbln.is_available()          # co-tenant probe in the parent
            r, w = os.pipe()
            if os.fork() == 0:
                os.close(r)
                os.environ["RBLN_DEVICES"] = "1"       # worker remap, after fork
                try:
                    msg = "OK:%d" % torch_rbln._C.physical_device_count()
                except BaseException as e:
                    msg = "FAIL:" + str(e).splitlines()[0][:80]
                os.write(w, msg.encode()[:200]); os.close(w); os._exit(0)
            os.close(w)
            ready, _, _ = select.select([r], [], [], 120)
            rec["result"] = os.read(r, 200).decode() if ready else "TIMEOUT"
            os.waitpid(-1, 0)
            """,
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        self.assertNotIn("Sealed", p.result, f"forked worker could not remap -- {p}")
        self.assertNotIn("FAIL", p.result, f"forked worker could not remap -- {p}")

    @xfail_until("fsw-inference#475", "the runtime seals RBLN_DEVICES on its first device-resolving call")
    def test_cpu_dataloader_does_not_touch_the_npu(self):
        """``DataLoader(pin_memory=True)`` must not seal or claim an NPU.

        torch/utils/data/dataloader.py:672,681 gate pinning on
        ``torch.accelerator.is_available()``. A pure-CPU DataLoader in a vLLM parent
        must not consume the process-wide seal or hold device contexts.
        """
        p = run_probe(
            """
            from torch.utils.data import DataLoader, TensorDataset
            dl = DataLoader(TensorDataset(torch.arange(4).float()), batch_size=2, pin_memory=True)
            rec["result"] = str(next(iter(dl))[0].shape)
            """,
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"DataLoader raised -- {p}")
        # skipTest, not a silent `if ... is not None`: an unavailable measurement must not
        # quietly turn this into a test that checks nothing.
        if p.ctx is None or p.sealed is None:
            self.skipTest(f"ctx/seal state undetermined in this environment -- {p}")
        self.assertFalse(p.ctx, f"DataLoader opened an NPU context -- {p}")
        self.assertFalse(p.sealed, f"DataLoader sealed RBLN_DEVICES -- {p}")

    def test_torch_load_reports_its_own_error(self):
        """``torch.load(map_location="rbln:0")`` must fail with torch's message.

        torch/serialization.py:606-648 ``_validate_device`` asks the device module for
        ``is_available()`` and ``device_count()`` so it can raise "Attempting to
        deserialize object on a RBLN device but torch.rbln.is_available() is False".
        A backend that raises its own config error instead replaces an actionable
        message with an unrelated one.
        """
        p = run_probe(
            """
            from torch.serialization import _validate_device
            try:
                rec["result"] = str(_validate_device("rbln:0", "rbln"))
            except Exception as e:
                rec["result"] = "RAISED: " + str(e).splitlines()[0]
            """,
            BAD_MAP,
        )
        self.assertIsNone(p.raised, f"probe harness error -- {p}")
        self.assertNotIn("out of range", p.result.lower(), f"backend error replaced torch's -- {p}")


if __name__ == "__main__":
    run_tests()
