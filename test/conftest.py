import json
import os
import sys
from pathlib import Path

import pytest
import torch
from _pytest.skipping import evaluate_skip_marks, evaluate_xfail_marks

import torch_rbln
from test.utils import CpuFallbackRecorder, set_cpu_fallback_recorder, set_deterministic_seeds, xfail_rebel
from torch_rbln._internal.log_utils import rbln_log_debug


# Set (per worker process) when an RBLN device drain fails in reset_caching_allocator:
# the device is faulted, so the rest of this worker's tests are skipped rather than run
# against it.
_rbln_device_poisoned = False

_CI_NPU_COUNTS = (0, 1, 2, 4, 8)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip once this worker's device has faulted, before fixture setup frees runtime
    buffers the device may still be using. Fail a serial test when xdist runs more than
    one worker. Skip a test that requires more NPUs than are available, and a test that
    runs only without an NPU when one is.
    """
    if _rbln_device_poisoned:
        pytest.skip("RBLN device drain failed earlier on this worker; skipping to avoid the faulted device")

    if item.get_closest_marker("serial") and hasattr(item.config, "workerinput"):
        worker_count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
        if worker_count > 1:
            pytest.fail(f"serial test needs one worker, not {worker_count}", pytrace=False)

    marker = item.get_closest_marker("npu")
    if marker is None:
        return
    count = marker.kwargs["count"]
    if count == 0:
        if torch.rbln.is_available():
            pytest.skip(f"NPUs required: {count}, available: {torch.rbln.physical_device_count()}")
        return
    physical_npu_count = torch.rbln.physical_device_count()
    if physical_npu_count < count:
        pytest.skip(f"NPUs required: {count}, available: {physical_npu_count}")


# =============================================================================
# Deterministic seed fixture (autouse)
# =============================================================================
@pytest.fixture(autouse=True)
def set_seeds():
    """Set deterministic seeds for reproducibility."""
    rbln_log_debug("Setting deterministic seeds")
    set_deterministic_seeds(0)


# =============================================================================
# TorchDynamo reset fixture (autouse)
# =============================================================================
@pytest.fixture(autouse=True)
def reset_dynamo(request):
    """Reset TorchDynamo before each test unless explicitly opted out."""
    if request.node.get_closest_marker("no_dynamo_reset"):
        rbln_log_debug("Skipping TorchDynamo reset")
        return

    rbln_log_debug("Resetting TorchDynamo")
    torch._dynamo.reset()


# =============================================================================
# RBLN caching-allocator reset fixture (autouse)
# =============================================================================
@pytest.fixture(autouse=True)
def reset_caching_allocator():
    """Drain this test's in-flight device work, then release the RBLN caching
    allocator's cached blocks -- both in teardown.

    Draining (synchronize) in this test's teardown attributes an async error to the test
    that caused it, instead of misreporting it against an unrelated later test.
    Releasing cached blocks then keeps the pool unfragmented for the next test. This
    always runs (no opt-out marker): skipping it would leave a test's blocks/in-flight
    work for the next, unrelated test to inherit -- the cross-test leak this fixture
    exists to prevent.
    """
    yield

    try:
        num_devices = torch_rbln.device.device_count()
    except Exception as exc:
        rbln_log_debug(f"device_count() failed; skipping caching allocator reset: {exc}")
        return

    # Drain first so an async fault is attributed to this test, not a later one. Collect
    # every device's failure so one device can't mask another's.
    sync_errors = []
    for idx in range(num_devices):
        try:
            torch.rbln.synchronize(idx)
        except Exception as exc:  # noqa: PERF203
            sync_errors.append(f"synchronize(rbln:{idx}): {exc!r}")

    # Drain failed: the device is faulted and may still have in-flight DMA, so don't
    # free its blocks/WarmCache (re-opens the async-buffer race). Poison the worker so
    # its remaining tests skip before any fixture runs -- else the next reset_dynamo
    # would flush the WarmCache and free those buffers anyway (see pytest_runtest_setup)
    # -- then surface the error.
    if sync_errors:
        global _rbln_device_poisoned
        _rbln_device_poisoned = True
        raise RuntimeError("RBLN device drain failed:\n  " + "\n  ".join(sync_errors))

    # Clean drain. Drop the SDPA attn-weights cache before flushing -- production
    # empty_cache() keeps it (pending backward), but there is none across a test
    # boundary; see memory.py. Lazy import avoids a load-time cycle.
    from torch_rbln._internal.kernels.sdpa import _clear_attn_weights_cache  # noqa: PLC0415

    _clear_attn_weights_cache()

    # Release cached blocks so the next test starts unfragmented. Aggregate flush
    # failures too.
    flush_errors = []
    for idx in range(num_devices):
        device = torch.device("rbln", idx)
        rbln_log_debug(f"Releasing RBLN caching allocator blocks on {device}")
        try:
            torch_rbln.memory.empty_cache(device)
        except Exception as exc:
            flush_errors.append(f"empty_cache(rbln:{idx}): {exc!r}")
    if flush_errors:
        raise RuntimeError("RBLN caching-allocator flush failed:\n  " + "\n  ".join(flush_errors))


# =============================================================================
# Global-state leak guards (autouse): keep one test's process-scoped mutations
# from leaking into later tests on the same xdist worker.
# =============================================================================
@pytest.fixture(autouse=True)
def restore_current_device():
    """Restore the RBLN device selection and the lazy-init flag after each test, so a
    test that called set_device() cannot leak either into later tests on the same
    worker.

    Only re-issue set_device when the index actually changed -- calling it
    unconditionally would itself flip ``_initialized`` False->True and leak that.
    Restore index and flag in lockstep; if the index restore fails, let it propagate (a
    half-restored device would silently mislead later tests) rather than rolling back
    only the flag.
    """
    # The `torch_rbln.device` package re-exports a `device` class that shadows the
    # `device` submodule, so reach the module (which owns the `_initialized` global that
    # set_device mutates) via sys.modules rather than a `from ... import device`.
    _dev = sys.modules[torch_rbln.device.set_device.__module__]

    # Fail loud on a snapshot failure too: device_count() returns 0 (never raises) when
    # there is no device, so an exception here is a real fault (e.g. malformed RBLN_*
    # config) that should surface, not be swallowed into a silent no-op teardown.
    saved = _dev.current_device() if _dev.device_count() > 0 else None
    saved_initialized = _dev._initialized
    yield
    if saved is None:
        _dev._initialized = saved_initialized
        return
    # Restore the flag only after the index restore succeeds, so the two never diverge.
    if _dev.current_device() != saved:
        _dev.set_device(saved)
    _dev._initialized = saved_initialized


# =============================================================================
# Current stream
# =============================================================================
@pytest.fixture(autouse=True)
def restore_current_stream(restore_current_device):  # noqa: ARG001
    """Restore every device's current stream after each test, so a test that selected a
    non-default stream cannot leak it into later tests on the same xdist worker.

    Reading a stream claims no device: ``RBLNGuardImpl::getStream`` answers with the
    default stream for a device whose context is not initialized, so the snapshot
    reaches the runtime (and with it ``to_device_id``, which commits the device mapping)
    only for devices already in use -- and initializing a context commits the mapping
    anyway. Restore only the devices whose stream moved: ``set_stream`` also selects the
    stream's device. Depends on ``restore_current_device`` so this teardown runs first
    and the device index (and the lazy-init flag) are put back after it.
    """
    count = torch.rbln.device_count()
    if count == 0:
        yield
        return
    saved = [torch.rbln.current_stream(index) for index in range(count)]
    yield
    for stream in saved:
        if torch.rbln.current_stream(stream.device_index) != stream:
            torch.rbln.set_stream(stream)


# =============================================================================
# Environment variable isolation fixtures
# =============================================================================
@pytest.fixture(autouse=True)
def enable_nan_inf_checks(monkeypatch):
    """Keep compatibility tests' NaN/Inf CPU fallback without changing runtime defaults.

    Function scope restores the caller's environment and lets an individual test
    explicitly exercise the default-off path. Subprocesses inherit the test setting.
    """
    monkeypatch.setenv("TORCH_RBLN_CHECK_NAN_INF", "ON")


@pytest.fixture(autouse=True)
def disable_compile_error_fallback(monkeypatch):
    """Disable 'compile_error' fallback by appending it to the existing
    TORCH_RBLN_DISABLE_FALLBACK list.
    """
    original_env = os.getenv("TORCH_RBLN_DISABLE_FALLBACK", "")
    fallback_categories = {c.strip() for c in original_env.split(",") if c.strip()} | {"compile_error"}
    new_env = ",".join(sorted(fallback_categories))
    rbln_log_debug(f"Setting TORCH_RBLN_DISABLE_FALLBACK='{new_env}' (was '{original_env}')")
    monkeypatch.setenv("TORCH_RBLN_DISABLE_FALLBACK", new_env)


@pytest.fixture
def enable_deploy_mode(monkeypatch):
    """Enable TORCH_RBLN_DEPLOY mode for eager execution tests."""
    original_env = os.getenv("TORCH_RBLN_DEPLOY", "")
    rbln_log_debug(f"Setting TORCH_RBLN_DEPLOY=ON (was '{original_env}')")
    monkeypatch.setenv("TORCH_RBLN_DEPLOY", "ON")


@pytest.fixture
def enable_eager_malloc(monkeypatch):
    """Enable TORCH_RBLN_EAGER_MALLOC for memory tests."""
    original_env = os.getenv("TORCH_RBLN_EAGER_MALLOC", "")
    rbln_log_debug(f"Setting TORCH_RBLN_EAGER_MALLOC=1 (was '{original_env}')")
    monkeypatch.setenv("TORCH_RBLN_EAGER_MALLOC", "1")


# REBEL-failing tests keyed by fully expanded name -> (test file, reason). Keying on
# the name pins one parametrization across separate ``@parametrize`` axes, which a
# per-parameter mark cannot.
#
# The float16 unaligned all-gather at size 67109568 (64 MiB) used to be xfailed
# here; it is now fixed (AllGather is chunked to a CS-safe size, see
# RCCL_ALLGATHER_MAX_OUTPUT_BYTES in ProcessGroupRBLN.cpp / fsw-inference#324),
# so its strict-xfail entry was removed.
_REBEL_XFAILS: dict[str, tuple[str, str]] = {}


# The dtype rebel_compiler's CI keeps of a test that also runs in other device
# dtypes, and the dtypes it then leaves out. Integer and bool variants stay: they
# take other paths.
_REPRESENTATIVE_DTYPE = torch.float16
_DTYPE_VARIANTS = (torch.bfloat16, torch.float32)


def _dtype_and_siblings_key(item):
    """Return ``(dtype, key)`` for an item parametrized over ``dtype``, else ``None``.

    Items sharing ``key`` differ only in ``dtype``. Covers
    ``instantiate_device_type_tests`` (``@dtypes`` and torch's ``@parametrize``,
    whose values the instantiated test keeps as its ``param_kwargs`` default,
    under the wrapper torch's ``TestCase`` adds) and ``pytest.mark.parametrize``.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is not None and "dtype" in callspec.params:
        params, name = callspec.params, item.originalname
    else:
        func, params = getattr(item.obj, "__func__", item.obj), {}
        name = getattr(func, "__name__", item.name)
        while func is not None:
            defaults = getattr(func, "__defaults__", None) or ()
            if len(defaults) == 1 and isinstance(defaults[0], dict):
                params = defaults[0]
                break
            func = getattr(func, "__wrapped__", None)
    dtype = params.get("dtype")
    if not isinstance(dtype, torch.dtype):
        return None
    rest = tuple(sorted((k, repr(v)) for k, v in params.items() if k not in ("dtype", "device")))
    return dtype, (item.parent.nodeid, name, rest)


def _runs_and_must_pass(item):
    """Whether, as far as collection can tell, ``item`` runs and a failure fails it.

    Reads the skip and xfail marks the way pytest does when it runs them, so a
    ``skipif`` whose condition is false still counts. A skip decided in the test
    body is not visible here.
    """
    if getattr(item.obj, "__unittest_skip__", False):  # @unittest.skip / skipIf
        return False
    return evaluate_skip_marks(item) is None and evaluate_xfail_marks(item) is None


def _mark_dtype_variants(items):
    """Mark ``dtype_variant`` on the bf16/fp32 items whose fp16 sibling runs.

    An fp16 item that is skipped or xfailed does not stand in for its siblings, so
    they stay unmarked and still run in every lane.
    """
    keyed = [(item, _dtype_and_siblings_key(item)) for item in items]
    representative = {
        k[1] for item, k in keyed if k is not None and k[0] == _REPRESENTATIVE_DTYPE and _runs_and_must_pass(item)
    }
    for item, k in keyed:
        if k is not None and k[0] in _DTYPE_VARIANTS and k[1] in representative:
            item.add_marker(pytest.mark.dtype_variant)


def pytest_collection_modifyitems(items):
    matched = set()
    for item in items:
        entry = _REBEL_XFAILS.get(item.name)
        if entry is not None:
            _, reason = entry
            item.add_marker(xfail_rebel(reason))
            matched.add(item.name)

    # Key whose file was collected but whose name no longer matches = stale (renamed or
    # re-parametrized); fail loudly instead of silently dropping the xfail.
    collected_files = {item.nodeid.split("::", 1)[0] for item in items}
    stale = [name for name, (path, _) in _REBEL_XFAILS.items() if name not in matched and path in collected_files]
    if stale:
        msg = f"Stale _REBEL_XFAILS keys (file collected, name unmatched): {stale}"
        raise pytest.UsageError(msg)

    # After the xfails above, so an fp16 item they mark is not taken as representative.
    _mark_dtype_variants(items)

    # The check covers every item, including those that -m deselects afterward.
    for item in items:
        # instantiate_device_type_tests copies and inherits class markers, so pytest
        # yields each twice.
        markers = list({id(m): m for m in item.iter_markers("npu")}.values())
        if not markers:
            continue
        if len(markers) > 1:
            msg = f"{item.nodeid}: expected at most 1 npu marker, got {len(markers)}: {[m.kwargs for m in markers]}"
            raise pytest.UsageError(msg)
        marker = markers[0]
        count = marker.kwargs.get("count")
        if marker.args or set(marker.kwargs) != {"count"} or count not in _CI_NPU_COUNTS:
            msg = (
                f"{item.nodeid}: expected npu(count=N) with N in {_CI_NPU_COUNTS}, "
                f"got args={marker.args} kwargs={marker.kwargs}"
            )
            raise pytest.UsageError(msg)


# =============================================================================
# CPU fallback report (--fallback-report)
# =============================================================================
def pytest_addoption(parser):
    parser.addoption(
        "--fallback-report",
        type=Path,
        metavar="PATH",
        help="Write each test's call outcome and the CPU fallbacks of the op calls it marks with "
        "measure_cpu_fallbacks(), per op and reason, to PATH as JSON, leaving out skipped tests and tests "
        "that mark nothing",
    )
    parser.addoption(
        "--shard",
        metavar="K/N",
        help="Run only the K-th of N shards of the selected tests, so N runs with K from 1 to N run each "
        "selected test once",
    )


def pytest_configure(config):
    path = config.getoption("--fallback-report")
    if path is not None:
        config.pluginmanager.register(CpuFallbackReport(config, path), "fallback-report")
    shard = config.getoption("--shard")
    if shard is not None:
        config.pluginmanager.register(Shard(shard), "shard")


_CPU_FALLBACK = pytest.StashKey[dict]()


class CpuFallbackReport:
    """CPU fallbacks over the op calls a test marks with ``measure_cpu_fallbacks()``
    (test/utils.py), so a test that passed on a CPU fallback -- comparing CPU against
    CPU -- shows up. ``test_compare_cpu`` marks its device call; a test that marks
    nothing has no record. Fixtures, the test's own input generation and comparison,
    and child processes are not counted.

    The counts are the per-op CPU fallback counts that every fallback path records, the
    ones ``torch.rbln.explain()`` reads. A test that resets them inside a window is
    marked, unless the reset came before an op's first fallback in its process, which
    leaves no trace.
    """

    def __init__(self, config, path):
        self.path = path
        self.worker = hasattr(config, "workerinput")
        self.tests = {}

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        recorder = CpuFallbackRecorder()
        set_cpu_fallback_recorder(recorder)
        try:
            return (yield)
        finally:
            set_cpu_fallback_recorder(None)
            if recorder.windows:
                item.stash[_CPU_FALLBACK] = recorder.record()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        # Subtest reports are made during the call, before its counts exist. User
        # properties carry the counts from an xdist worker to the controller.
        if call.when == "call" and _CPU_FALLBACK in item.stash:
            report.user_properties.append(("rbln_cpu_fallback", item.stash[_CPU_FALLBACK]))
        return report

    def pytest_runtest_logreport(self, report):
        # Subtest reports carry a `context`; the parent's call report covers its call.
        # Skipped tests, xfails among them, are left out.
        if self.worker or report.when != "call" or report.skipped or hasattr(report, "context"):
            return
        # Missing when reading the counts raised, which failed the call with that error.
        fallback = dict(report.user_properties).get("rbln_cpu_fallback")
        if fallback is not None:
            # The call phase's outcome: a test recorded as passed can still fail in
            # fixture teardown, such as the device synchronization after it.
            self.tests[report.nodeid] = {"call_outcome": report.outcome, **fallback}

    def pytest_sessionfinish(self):
        if not self.worker:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.tests, indent=2))


# =============================================================================
# Sharding: --shard K/N
# =============================================================================
class Shard:
    """Keep every N-th selected test in collection order, starting at the K-th.

    Taking neighbors in turn spreads a run of similar tests, such as one test's dtype
    parametrizations, over the shards.
    """

    def __init__(self, spec: str):
        k, sep, n = spec.partition("/")
        if not (sep and k.isdigit() and n.isdigit() and 1 <= int(k) <= int(n)):
            msg = f"--shard must be K/N with 1 <= K <= N, not {spec!r}"
            raise pytest.UsageError(msg)
        self.index = int(k) - 1
        self.count = int(n)
        os.environ.setdefault("MASTER_PORT", str(29604 + self.index))

    # Last, so the shards split what -m and -k leave selected.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        config.hook.pytest_deselected(items=[item for i, item in enumerate(items) if i % self.count != self.index])
        items[:] = items[self.index :: self.count]
