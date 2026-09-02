# Owner(s): ["module: PrivateUse1"]

"""PrivateUse1 backend contract conformance.

torch does not merely *offer* the ``torch.rbln`` module and the RBLN accelerator hooks --
it *calls into them*, from paths that have nothing to do with wanting an NPU:
``DataLoader(pin_memory=True)``, ``torch.load(map_location=...)``, importing
``torch.testing._internal.common_utils``, ``torch._utils._get_available_device_type()``.

Every test pins exactly ONE upstream clause and cites its source, so a torch upgrade or a
new call site fails on the clause rather than on a downstream symptom.

Each probe runs in a fresh subprocess (the state involved is
process-global and one-shot):

``raised``  the probe propagated an exception
``ctx``     the probe opened an NPU context (``rbln-stat`` reports this pid)
``remap``   what a later ``RBLN_DEVICES`` remap does: ``applied`` (mapping still live),
            ``frozen`` (silently ignored), ``None`` (undetermined -- fewer than two usable
            NPUs, where a live mapping and a frozen one report the same count)

``remap`` is load-bearing for vLLM: ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to ``fork``
and ``RBLNWorker._init_device_env()`` remaps ``RBLN_DEVICES`` *inside* the forked worker, so
a mapping frozen in the parent breaks every worker.

A clause not satisfied yet is a ``strict=True`` xfail naming the work that closes it; an
unexpected pass means the marker should go. One group remains: ``phase 4``, the RNG /
serialization / device-name surface of ``torch.rbln``.
"""

import functools
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

# The RBLN_* variables a scenario owns. Stripped from a probe's environment so the runner's
# own selection cannot decide the outcome.
_SELECTION_VARS = frozenset(
    {"RBLN_DEVICES", "RBLN_VISIBLE_DEVICES", "RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE", "RBLN_DUMMY_DEVICE"}
)

ALL_SCENARIOS = {
    "healthy": HEALTHY,
    "no_device": NO_DEVICE,
    "bad_map": BAD_MAP,
    "dummy": DUMMY,
    "dummy_bad_map": DUMMY_BAD_MAP,
}

# Probes must assign ``rec["result"]``. They run after ``import torch, torch_rbln``.
_RUNNER = '''\
import json, os, select, shutil, subprocess, sys

sys.path.insert(0, {root!r})
{env}
# ERROR, not a quieter value: RBLN_CHECK logs at ERROR and has_cpp_traceback has to see
# it. TORCH_RBLN_LOG_LEVEL accepts only DEBUG/INFO/WARNING/ERROR and raises otherwise.
os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

# The probe may print; only this file descriptor carries the JSON result.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

rec = {{"result": None, "raised": None, "ctx": None, "remap_counts": None}}

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


def _remap_counts():
    """Visible device count with RBLN_DEVICES set to one device, then unset.

    A live mapping answers ``[1, every visible NPU]``; a frozen one answers the same number
    twice. Measured by the count rather than by matching an error message: a frozen mapping is
    ignored silently and only rejected at the next acquisition, so there is no message to match
    at this point.

    Returns the two numbers rather than a verdict. Reading them as "frozen" needs the physical
    pool, which only a process that has not latched anything can report, so
    :attr:`Probe.remap` draws the conclusion parent-side.

    Goes through ``rebel._C``, not a torch_rbln API, so a torch_rbln entry point that stopped
    freezing cannot make these checks pass for the wrong reason.

    Measured in a forked child, because reading the counts means rewriting RBLN_DEVICES and
    the probe body may have left threads running (``DataLoader(pin_memory=True)`` keeps a
    pin-memory thread alive). setenv/unsetenv beside another thread's getenv is a data race,
    and c10/rbln/DeviceMappingManager.h rules out mutating RBLN_* alongside a query at all.
    The seal this measures is inherited across fork, so the child reads the parent's latch
    state while its own edits stay private to it -- which also drops the "must run last"
    constraint the in-process version had.
    """
    from rebel._C import device_count

    def count(value):
        # The alias selects the same pool, so it has to be cleared either way.
        os.environ.pop("RBLN_VISIBLE_DEVICES", None)
        if value is None:
            os.environ.pop("RBLN_DEVICES", None)
        else:
            os.environ["RBLN_DEVICES"] = value
        return device_count()

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        # os._exit throughout: the child must not run the parent's atexit hooks, flush its
        # buffers, or unwind into pytest. A failed measurement is reported as none at all.
        try:
            os.close(read_fd)
            payload = json.dumps([count("0"), count(None)])
        except BaseException:
            payload = "null"
        try:
            os.write(write_fd, payload.encode())
        except BaseException:
            pass
        os._exit(0)
    os.close(write_fd)
    try:
        # A hang here would cost the whole record: run_probe would time out with no JSON line
        # to read, so the wait is bounded and a silent child means "undetermined".
        ready, _, _ = select.select([read_fd], [], [], 120)
        raw = os.read(read_fd, 200).decode() if ready else ""
    except BaseException:
        raw = ""
    finally:
        os.close(read_fd)
        try:
            os.waitpid(child, 0)
        except BaseException:
            pass
    try:
        counts = json.loads(raw)
    except ValueError:
        return None
    return counts if isinstance(counts, list) else None


try:
{probe}
except BaseException as e:  # noqa: BLE001 - the point is to characterise failures
    rec["raised"] = "{{}}: {{}}".format(type(e).__name__, str(e).splitlines()[0][:160])

rec["ctx"] = _ctx_opened()
rec["remap_counts"] = _remap_counts()
print(json.dumps(rec), file=_real_stdout)
'''


# Physical NPU ids, scanned in a process of its own. Sparse ids are possible (a container can
# be granted 5,6,7), so this scans instead of taking range(device_count()); 127 is the ceiling
# the mapping layer enforces on a logical index.
_POOL_SCAN = """\
import json
from rebel._C import npu_is_available

print(json.dumps({"ids": [i for i in range(127) if npu_is_available(i)]}))
"""


@functools.lru_cache(maxsize=1)
def _host_pool() -> tuple:
    """Every physical NPU id on this host, scanned in a process of its own.

    It has to be another process. The runtime's RBLN_DEVICES seal is process-local, so once
    anything here has used a device an in-process query answers with the sealed value --
    ``physical_device_count()`` included, since it is the same ``rbln_get_device_count()`` the
    probe measures -- and the state under test would be deciding its own gate. Stripping the
    selection variables is what makes the answer the host rather than a selection of it.

    Empty when the runtime, the driver or the NPUs are absent, and when the scan itself fails:
    callers then treat every remap verdict as undetermined.
    """
    env = {k: v for k, v in os.environ.items() if k not in _SELECTION_VARS}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _POOL_SCAN],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception:
        return ()
    # The C++ logger shares stdout, so scan back for the line that is the record.
    for line in reversed(proc.stdout.splitlines()):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "ids" in rec:
            return tuple(rec["ids"])
    return ()


@functools.lru_cache(maxsize=1)
def _visible_pool() -> tuple:
    """Physical NPU ids this job may use: the host pool, narrowed to the runner's selection.

    The reference every remap verdict is judged against, and what the scenarios select from.
    A CI step allocated one NPU out of several is the case that matters: judged against the
    host it would read a legitimate 1-and-1 as a freeze, and a scenario picking an id off the
    host would claim an NPU that belongs to a co-tenant job.

    The runner's selection is a bound on which ids are ours; it never decides what a scenario
    *sets*, which is the whole point of stripping it from the probe's environment.
    """
    host = _host_pool()
    selected = os.environ.get("RBLN_DEVICES") or os.environ.get("RBLN_VISIBLE_DEVICES")
    if not selected:
        return host
    try:
        ids = tuple(int(part) for part in selected.split(",") if part.strip())
    except ValueError:
        return host  # a malformed selection is the runtime's to reject, not ours
    return tuple(i for i in ids if i in host)


class Probe:
    """Outcome of running one probe in a fresh process."""

    def __init__(self, rec, stdout, stderr):
        self.result = rec["result"]
        self.raised = rec["raised"]
        self.ctx = rec["ctx"]
        self.remap_counts = rec["remap_counts"]
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        return (
            f"Probe(result={self.result!r}, raised={self.raised!r}, ctx={self.ctx!r}, "
            f"remap={self.remap!r}, counts={self.remap_counts!r}, "
            f"pool={len(_visible_pool())}/{len(_host_pool())})"
        )

    @property
    def remap(self):
        """What a later ``RBLN_DEVICES`` remap did: ``applied`` / ``frozen`` / ``None``.

        ``None`` means undetermined, not benign: with fewer than two usable NPUs a live mapping
        and a frozen one report the same count, so nothing can be concluded either way.

        The reference is :func:`_visible_pool`, not ``/dev/rbln*``. A container can expose more
        device nodes than it grants -- a Buildkite step asking for ``npu: count: 1`` on a
        multi-NPU host -- and counting nodes there reads a legitimate 1-and-1 as a freeze.
        """
        if self.remap_counts is None or len(_visible_pool()) < 2:
            return None
        one, every = self.remap_counts
        if one == every:
            return "frozen"  # neither value reached the runtime
        if one == 1:
            return "applied"
        return None

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
    # The scenario is the whole RBLN_* configuration: inheriting the parent's would let a
    # device selection in the test runner's environment decide what the probe sees, which
    # shows up as a silent skip rather than a failure.
    child_env = {k: v for k, v in os.environ.items() if k not in _SELECTION_VARS}
    child_env.update({k: v for k, v in env.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        env=child_env,
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
        if isinstance(rec, dict) and "remap_counts" in rec:
            # The record is not enough: the runner catches every exception, so a probe that
            # reported and then died in teardown (SIGABRT out of the runtime, say) exits
            # non-zero with a complete JSON line. Reading only the record passes that.
            if proc.returncode != 0:
                raise AssertionError(
                    f"probe reported but exited abnormally (rc={proc.returncode})\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
                )
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

    Every clause below is expressed as "the probe did NOT do X", which passes both when the
    contract holds and when the harness has gone blind. Each of the three measurements
    therefore needs a case that deliberately triggers it.
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

    def test_detects_a_frozen_mapping(self):
        """A probe that acquires a device must be reported as freezing the mapping.

        The freeze is only observable through an *acquisition*: a query leaves the mapping
        live, so a query-based control would report "not frozen" forever and every "left it
        remappable" clause below would pass for free.
        """
        p = run_probe(
            'rec["result"] = float(torch.ones(4, dtype=torch.float16, device="rbln:0").sum().cpu())',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"allocation failed, cannot control for the freeze -- {p}")
        if p.remap is None:
            self.skipTest(f"remap state undetermined in this environment -- {p}")
        self.assertEqual(p.remap, "frozen", f"harness did not see the mapping freeze -- {p}")

    def test_detects_a_cpp_traceback(self):
        """A failure from a still-loud ``RBLN_CHECK`` must be seen by ``has_cpp_traceback``.

        An unassigned device index goes through ``check_device_index()``, which kept plain
        ``RBLN_CHECK`` -- a use error, not something a probe walks into -- so it logs
        ``c10::Error::what()`` and the trace it embeds.

        Deliberately not a malformed-config failure: those rethrow the stored plan error via
        ``RBLN_CHECK_QUIET`` and so carry the message without a logged trace, which makes
        them useless as a control here.
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

        c10/cuda/CUDAFunctions.h -- "people basically ~never want this function
        to fail; it should just return zero if things are not working. Oblige them."
        ``device_count() noexcept`` with a separately named throwing variant
        ``device_count_ensure_non_zero()``.
        ATen/DeviceAccelerator.h -- deviceCount() "is *REQUIRED* to not raise any
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

        ATen/detail/AcceleratorHooksInterface.h -- isAvailable() "should NOT
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
        # The count is pinned as well as the agreement: this scenario has exactly one device,
        # and equality alone would also hold if both sides regressed to 0 -- which is the
        # half of the original bug that the hooks side had.
        self.assertEqual(module_count, 1, f"scenario is not live -- {p}")
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
                # And pin the answer: agreement alone would also hold if both sides said
                # False everywhere, which is a backend that reports itself unusable.
                self.assertEqual(py_answer, name in ("healthy", "dummy"), f"{name}: wrong answer -- {p}")

    # -- torch's own entry points ------------------------------------------

    def test_torch_accelerator_apis_never_throw(self):
        """``torch.accelerator.is_available()`` / ``device_count()`` must never raise.

        torch/accelerator/__init__.py delegates to the device module. DataLoader calls
        ``torch.accelerator.is_available()`` for ``pin_memory`` (torch/utils/data/
        dataloader.py), so a raise breaks a CPU-only DataLoader.
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

        torch/_utils.py calls ``custom_device_mod.is_available()``. It backs
        ``_get_device_attr`` / ``_get_all_device_indices``, used well outside RBLN code.
        """
        for name, env in ALL_SCENARIOS.items():
            with self.subTest(scenario=name):
                p = run_probe('rec["result"] = torch._utils._get_available_device_type()', env)
                self.assertIsNone(p.raised, f"{name}: raised -- {p}")

    def test_common_utils_is_importable(self):
        """``import torch.testing._internal.common_utils`` must not raise.

        torch/testing/_internal/common_utils.py evaluates
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

        The obvious implementation walks ``device_count()`` to build one generator per device,
        which is only viable because enumeration claims nothing.

        torch/random.py::_seed_custom_device requires ``_is_in_bad_fork`` **and**
        ``manual_seed_all`` on the device module; without both it warns and silently does
        nothing, so RBLN results are not reproducible from ``torch.manual_seed()``.
        Listed as a required backend API in torch/utils/backend_registration.py.
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

        torch/serialization.py documents it as required of a privateuse1 backend
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
        ``get_device_properties`` and ``get_device_capability``, and frameworks reach for them
        through the device module (vLLM's XPU platform is
        ``torch.xpu.get_device_name(device_id)``, vllm/platforms/xpu.py). With no RBLN
        equivalent, vllm-rbln calls ``rebel.get_npu_name()`` directly, bypassing torch, so a
        torch-level policy has nothing to apply to. ``RBLNGuardImpl::getDeviceCapability()``
        already exists on the C++ side.
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

        ``RBLN_CHECK`` (c10/rbln/RBLNLogging.h) logs ``c10::Error::what()`` -- the
        C++ stack trace included -- to **stdout** before throwing, at ERROR level, so no
        ``TORCH_RBLN_LOG_LEVEL`` setting suppresses it and catching the exception does not
        either.
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

    # -- the counterpart clause: using a device must freeze the mapping -----

    def test_device_use_freezes_the_mapping(self):
        """Using a device must freeze the mapping, even though querying does not.

        The clauses above all say a probe left the mapping remappable. On their own they
        are also satisfied by a backend that never freezes it at all, which would let a
        launcher renumber devices out from under live allocations. ``torch.cuda`` draws the
        same line: torch/cuda/__init__.py refuses to cache the device count "prior to CUDA
        initialization" and caches it from ``_lazy_init`` onwards.

        Both layers freeze at this same moment -- ``DeviceMappingManager::commit()`` calls
        ``rbln_register_device_id()``, which reaches ``Context::Create`` and the runtime's
        latch -- so this also pins that they cannot drift apart.
        """
        p = run_probe(
            'rec["result"] = float(torch.ones(4, dtype=torch.float16, device="rbln:0").sum().cpu())',
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"allocation failed -- {p}")
        if p.remap is None:
            self.skipTest(f"remap state undetermined in this environment -- {p}")
        self.assertEqual(p.remap, "frozen", f"device use left the mapping remappable -- {p}")

    def test_device_selection_after_import_is_honoured(self):
        """Selecting devices *after* import must change what this layer reports.

        This is the point of planning separately from committing: a launcher assigns the
        visible devices once it knows the rank, necessarily after ``import torch``.
        torch/cuda/__init__.py states the same rule -- do not cache the device count "prior
        to CUDA initialization, because the number of devices can change due to changes to
        ``CUDA_VISIBLE_DEVICES``".

        Both names, because the runtime ships ``RBLN_VISIBLE_DEVICES`` as an alias of the
        same flag and it renumbers the pool identically. Both answers, because checking one
        cannot tell "the assignment was honoured" from "the two disagree" -- leaving the
        alias out of the plan signature made ``device_count()`` answer from a stale plan
        while ``physical_device_count()``, which bypasses it, answered from the new pool.
        """
        for name in ("RBLN_DEVICES", "RBLN_VISIBLE_DEVICES"):
            with self.subTest(variable=name):
                p = run_probe(
                    f"""
                    before = [torch.rbln.device_count(), torch.rbln.physical_device_count()]
                    os.environ[{name!r}] = "0"
                    rec["result"] = [before, [torch.rbln.device_count(), torch.rbln.physical_device_count()]]
                    """,
                    {},  # neither name set: the primary wins when both are, so both must be absent
                )
                self.assertIsNone(p.raised, f"{name}: probe raised -- {p}")
                before, after = p.result
                if before[0] < 2:
                    self.skipTest(f"needs at least two visible NPUs to tell the counts apart -- {p}")
                self.assertEqual(after, [1, 1], f"{name} did not reach both answers -- {p}")

    def test_a_zero_device_plan_is_recoverable(self):
        """A failed device use with no devices must not freeze the mapping.

        Commit freezes the plan so a later ``RBLN_*`` change cannot renumber devices out from
        under live allocations. With nothing planned nothing is registered, so the freeze has
        nothing to protect and only makes the 0-device state permanent -- a launcher that
        probed with a bad value could never recover in that process.
        """
        p = run_probe(
            """
            before = torch.rbln.device_count()
            try:
                torch.empty(1, dtype=torch.float16, device="rbln:0")
                raised = None
            except RuntimeError as e:
                raised = str(e).splitlines()[0]
            os.environ["RBLN_DEVICES"] = "0"
            rec["result"] = [before, raised is not None, torch.rbln.device_count()]
            """,
            NO_DEVICE,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        before, use_failed, after = p.result
        self.assertEqual(before, 0, f"scenario is not live -- {p}")
        self.assertTrue(use_failed, f"device use must fail with no devices -- {p}")
        self.assertEqual(after, 1, f"the 0-device plan was frozen and could not recover -- {p}")

    def test_current_device_stays_inside_the_plan(self):
        """``current_device()`` must never name a device outside the current plan.

        The selection is ``thread_local`` while the plan is process-wide, so shrinking
        ``RBLN_DEVICES`` after ``set_device()`` -- both legal before the mapping commits --
        can leave the selection pointing past the end of it.
        """
        p = run_probe(
            """
            before = torch.rbln.device_count()
            if before == 2:
                torch.rbln.set_device(1)
                os.environ["RBLN_DEVICES"] = "0"
            rec["result"] = [before, torch.rbln.device_count(), torch.rbln.current_device()]
            """,
            {"RBLN_DEVICES": "0,1"},
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        before, count, current = p.result
        if before != 2:
            self.skipTest(f"needs two visible NPUs to select a device the replan drops -- {p}")
        self.assertEqual(count, 1, f"the replan did not take effect -- {p}")
        self.assertLess(current, count, f"current_device() is outside the plan -- {p}")

    def test_a_frozen_mapping_survives_unsetting_the_variable(self):
        """Clearing ``RBLN_DEVICES`` after a device is in use must not un-freeze the mapping.

        The runtime checks the freeze before the environment, so an unset cannot fall back to
        "auto-discover all devices" and quietly widen the pool under live allocations. Pinned
        because it is the escape hatch a remap-rejection check alone would miss: unsetting is
        not a changed value, it is no value.
        """
        p = run_probe(
            """
            torch.ones(4, dtype=torch.float16, device="rbln:0")
            os.environ.pop("RBLN_DEVICES", None)
            rec["result"] = [torch.rbln.device_count(), torch.rbln.physical_device_count()]
            """,
            HEALTHY,  # RBLN_DEVICES="0": one device, so widening would be visible
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        self.assertEqual(p.result, [1, 1], f"unsetting RBLN_DEVICES widened the pool -- {p}")


@pytest.mark.test_set_ci
@requires_physical_devices(1)
class TestExternalConsumers(TestCase):
    """Scenarios reproducing how vLLM and LMCache drive this backend; each is a reported
    failure.
    """

    def test_cotenant_availability_probe_leaves_the_mapping_remappable(self):
        """A co-tenant availability probe must leave ``RBLN_DEVICES`` remappable.

        LMCache runs device detection while importing ``lmcache.v1.platform``, on every
        start, in whatever process imports it -- including a vLLM parent that has not yet
        forked its workers. A mapping frozen there is inherited by every worker.
        """
        p = run_probe('rec["result"] = torch.rbln.is_available()', HEALTHY)
        self.assertIsNone(p.raised, f"is_available() raised -- {p}")
        if p.remap is None:
            self.skipTest(f"remap state undetermined in this environment -- {p}")
        self.assertEqual(p.remap, "applied", f"availability probe froze the mapping -- {p}")

    def test_fork_then_worker_remap_succeeds(self):
        """A forked worker must still be able to remap ``RBLN_DEVICES``.

        ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to ``fork`` (vllm/envs.py) and
        ``RBLNWorker._init_device_env()`` assigns ``os.environ[RBLN_DEVICES]`` inside the
        forked worker. A frozen mapping is inherited across fork, so a probe in the parent
        breaks every worker deterministically.

        The parent gets *two* devices and the child keeps *one*, so the counts differ: an
        ignored remap leaves the child reporting the parent's 2. A same-size remap cannot tell
        those apart, and neither can "the child did not fail" -- an ignored remap raises
        nothing. Both counts are read because ``physical_device_count()`` asks the runtime
        directly, so it alone would pass with this layer still serving the parent's plan.
        """
        p = run_probe(
            """
            import select
            torch.rbln.is_available()          # co-tenant probe in the parent
            parent = torch.rbln.device_count()
            r, w = os.pipe()
            if os.fork() == 0:
                os.close(r)
                os.environ["RBLN_DEVICES"] = "1"       # worker remap, after fork
                try:
                    msg = "OK:%d,%d" % (torch_rbln._C.physical_device_count(), torch.rbln.device_count())
                except BaseException as e:
                    msg = "FAIL:" + str(e).splitlines()[0][:80]
                os.write(w, msg.encode()[:200]); os.close(w); os._exit(0)
            os.close(w)
            ready, _, _ = select.select([r], [], [], 120)
            child = os.read(r, 200).decode() if ready else "TIMEOUT"
            os.waitpid(-1, 0)
            rec["result"] = [parent, child]
            """,
            {"RBLN_DEVICES": "0,1"},
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        parent, child = p.result
        if parent != 2:
            self.skipTest(f"needs two visible NPUs for the child's count to differ -- {p}")
        self.assertNotIn("FAIL", child, f"forked worker could not remap -- {p}")
        self.assertEqual(child, "OK:1,1", f"the child kept the parent's mapping -- {p}")

    def test_cpu_dataloader_does_not_touch_the_npu(self):
        """``DataLoader(pin_memory=True)`` must not freeze the mapping or claim an NPU.

        torch/utils/data/dataloader.py gate pinning on
        ``torch.accelerator.is_available()``. A pure-CPU DataLoader in a vLLM parent must
        not freeze the process-wide mapping or hold device contexts.
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
        if p.ctx is None or p.remap is None:
            self.skipTest(f"ctx/remap state undetermined in this environment -- {p}")
        self.assertFalse(p.ctx, f"DataLoader opened an NPU context -- {p}")
        self.assertEqual(p.remap, "applied", f"DataLoader froze the mapping -- {p}")

    def test_import_does_not_resolve_a_device(self):
        """Importing ``torch_rbln`` must not resolve a device to read its architecture.

        Resolving one opens and closes a device node on every import.
        ``get_device_arch`` is an ``lru_cache``: an empty cache means nothing asked.
        """
        p = run_probe(
            """
            from torch_rbln._internal.device_arch_utils import get_device_arch

            rec["result"] = get_device_arch.cache_info().currsize
            """,
            HEALTHY,
        )
        self.assertIsNone(p.raised, f"probe raised -- {p}")
        self.assertEqual(p.result, 0, f"import resolved a device to read its arch -- {p}")

    def test_torch_load_reports_its_own_error(self):
        """``torch.load(map_location="rbln:0")`` must fail with torch's message.

        torch/serialization.py ``_validate_device`` asks the device module for
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
        # Both halves: torch's message has to be the one that surfaces, and ours must not.
        # Asserting only the absence would also hold if _validate_device stopped raising.
        self.assertTrue(p.result.startswith("RAISED"), f"_validate_device must reject rbln:0 -- {p}")
        self.assertIn("torch.rbln.is_available() is False", p.result, f"not torch's message -- {p}")
        self.assertNotIn("out of range", p.result.lower(), f"backend error replaced torch's -- {p}")


if __name__ == "__main__":
    run_tests()
