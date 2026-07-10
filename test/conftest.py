import os
import sys

import pytest
import torch

import torch_rbln
from test.utils import set_deterministic_seeds, xfail_rebel
from torch_rbln._internal.log_utils import rbln_log_debug


# =============================================================================
# Deterministic seed fixture (autouse)
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def set_seeds():
    """Set deterministic seeds for reproducibility."""
    rbln_log_debug("Setting deterministic seeds")
    set_deterministic_seeds(0)


# =============================================================================
# TorchDynamo reset fixture (autouse)
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
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
@pytest.fixture(scope="function", autouse=True)
def reset_caching_allocator(request):
    """Drain this test's in-flight device work, then release the RBLN caching allocator's
    cached blocks -- both in teardown.

    Draining (synchronize) runs in this test's teardown, not the next test's setup, so an
    async error surfaces here (attributed to the test that caused it) and fails loudly
    rather than being swallowed and misreported as an empty_cache failure in an unrelated
    later test. Releasing cached blocks then keeps freed small blocks from fragmenting the
    pool for the next test. Opt out per-test with the ``no_caching_allocator_reset`` marker.
    """
    yield
    if request.node.get_closest_marker("no_caching_allocator_reset"):
        rbln_log_debug("Skipping RBLN caching allocator reset")
        return

    try:
        num_devices = torch_rbln.device.device_count()
    except Exception as exc:
        rbln_log_debug(f"device_count() failed; skipping caching allocator reset: {exc}")
        return

    # Collect every device's cleanup failure and raise them together at the end, so one
    # device's fault can't mask another's and nothing is silently swallowed.
    errors: list[str] = []
    # Drain first so an async fault is attributed to this test (not a later one).
    for idx in range(num_devices):
        try:
            torch.rbln.synchronize(idx)
        except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
            errors.append(f"synchronize(rbln:{idx}): {exc!r}")
    # Release cached blocks regardless of the drain outcome, so the next test starts with
    # an unfragmented pool even when the drain failed.
    for idx in range(num_devices):
        device = torch.device("rbln", idx)
        rbln_log_debug(f"Releasing RBLN caching allocator blocks on {device}")
        try:
            torch_rbln.memory.empty_cache(device)
        except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
            errors.append(f"empty_cache(rbln:{idx}): {exc!r}")
    if errors:
        raise RuntimeError("RBLN caching-allocator teardown failed:\n  " + "\n  ".join(errors))


# =============================================================================
# Global-state leak guards (autouse): keep one test's process-scoped mutations
# from leaking into later tests on the same xdist worker.
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def restore_current_device():
    """Restore the RBLN device selection AND the lazy-init flag after each test, so a test
    that calls set_device() cannot leak either into later tests on the same worker.

    Snapshot both, then only issue the low-level set_device when the index actually changed:
    calling it unconditionally flips ``_initialized`` False->True (set_device sets it) on
    tests that never touched the device, which is itself a leak. Restore the index and the
    flag in lockstep and fail loud if the index restore fails -- a half-restored device
    (index moved, flag rolled back to the old value) would silently mislead every later
    test on this worker, so surface it as a teardown error instead of swallowing it."""
    # The `torch_rbln.device` package re-exports a `device` class that shadows the
    # `device` submodule, so reach the module (which owns the `_initialized` global that
    # set_device mutates) via sys.modules rather than a `from ... import device`.
    _dev = sys.modules[torch_rbln.device.set_device.__module__]

    try:
        saved = _dev.current_device() if _dev.device_count() > 0 else None
    except Exception:
        saved = None
    saved_initialized = _dev._initialized
    yield
    if saved is None:
        _dev._initialized = saved_initialized
        return
    # Do NOT swallow a restore failure: if current_device()/set_device() raise, let it
    # propagate as a teardown error rather than rolling back only the flag. The flag is
    # restored only after the index restore succeeds, so the two never diverge.
    if _dev.current_device() != saved:
        _dev.set_device(saved)
    _dev._initialized = saved_initialized


@pytest.fixture(scope="function", autouse=True)
def keep_torch_compile_patches():
    """Re-apply torch_rbln's import-time torch.compile / torch._dynamo.reset patches if a
    test removed or swapped them, so later tests (and the autouse reset_dynamo above) keep
    the RBLN wrappers rather than the bare originals."""
    yield
    import torch_rbln._internal.monkey_patches as mp

    # patches_active() checks callable identity (not just the bookkeeping flags), so a test
    # that rebinds torch.compile to something else is caught and the RBLN wrappers restored.
    if not mp.patches_active():
        rbln_log_debug("Re-applying RBLN torch.compile patches leaked-off by a prior test")
        # remove first: patch_torch_compile() is a no-op while the flags still read
        # "patched", so a silent rebind would otherwise survive apply_all_patches().
        mp.remove_all_patches()
        mp.apply_all_patches()


# =============================================================================
# Environment variable isolation fixtures
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def disable_compile_error_fallback(monkeypatch):
    """Disable 'compile_error' fallback by appending it to the existing TORCH_RBLN_DISABLE_FALLBACK list."""
    original_env = os.getenv("TORCH_RBLN_DISABLE_FALLBACK", "")
    fallback_categories = {c.strip() for c in original_env.split(",") if c.strip()} | {"compile_error"}
    new_env = ",".join(sorted(fallback_categories))
    rbln_log_debug(f"Setting TORCH_RBLN_DISABLE_FALLBACK='{new_env}' (was '{original_env}')")
    monkeypatch.setenv("TORCH_RBLN_DISABLE_FALLBACK", new_env)


@pytest.fixture(scope="function")
def enable_deploy_mode(monkeypatch):
    """Enable TORCH_RBLN_DEPLOY mode for eager execution tests."""
    original_env = os.getenv("TORCH_RBLN_DEPLOY", "")
    rbln_log_debug(f"Setting TORCH_RBLN_DEPLOY=ON (was '{original_env}')")
    monkeypatch.setenv("TORCH_RBLN_DEPLOY", "ON")


@pytest.fixture(scope="function")
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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # Defense-in-depth for single_worker: those tests must run in a dedicated serial pass
    # (test/run_tests.py splits them out). A raw parallel `pytest -n<N>` that collects them
    # without `-m "not single_worker"` lands them on parallel workers and corrupts shared
    # process/runtime state -- fail at collection instead of flaking later. trylast so this
    # runs after pytest's own -m/-k deselection (the serial pass filters them out first).
    #
    # Detect a genuine parallel run by WORKER COUNT, not xdist's dist mode. Under a real
    # `pytest -nN` the controller never collects (dsession blocks it); collection runs only
    # inside each worker, where xdist resets numprocesses=None / dist="no" -- so a
    # numprocesses/dist check would always read "serial" there and never fire. The worker
    # instead carries workerinput["workercount"] (the true parallelism). Fall back to
    # numprocesses only off-worker (controller / no xdist / --collect-only, which skips the
    # distributed session). `-n1` -> workercount 1 -> serial-safe (the single_worker pass).
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        parallel = int(workerinput.get("workercount", 1)) > 1
    else:
        numprocesses = config.getoption("numprocesses", 0) or 0
        parallel = numprocesses > 1 if isinstance(numprocesses, int) else True
    if parallel:
        offenders = [it.nodeid for it in items if it.get_closest_marker("single_worker")]
        if offenders:
            shown = ", ".join(offenders[:5]) + (f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else "")
            raise pytest.UsageError(
                "single_worker test(s) selected under a parallel xdist run; isolate them with "
                f"-m 'not single_worker' (or run via test/run_tests.py): {shown}"
            )

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
        raise pytest.UsageError(f"Stale _REBEL_XFAILS keys (file collected, name unmatched): {stale}")
