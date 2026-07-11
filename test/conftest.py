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

    Draining (synchronize) in this test's teardown attributes an async error to the test
    that caused it, instead of misreporting it against an unrelated later test. Releasing
    cached blocks then keeps the pool unfragmented for the next test. Opt out with the
    ``no_caching_allocator_reset`` marker -- which then leaves this test's blocks/in-flight
    work for the next test to inherit (cleaned only at that next test's teardown)."""
    yield
    if request.node.get_closest_marker("no_caching_allocator_reset"):
        rbln_log_debug("Skipping RBLN caching allocator reset")
        return

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
        except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
            sync_errors.append(f"synchronize(rbln:{idx}): {exc!r}")

    # If any drain failed, do NOT flush: a faulted/undrained device may still have in-flight
    # DMA referencing its blocks, so freeing them re-opens the async-buffer/free race this
    # drain guards against -- and empty_cache() additionally clears the *process-global*
    # WarmCache. Surface the drain errors and stop.
    if sync_errors:
        raise RuntimeError("RBLN device drain failed:\n  " + "\n  ".join(sync_errors))

    # All drains succeeded -> safe to release cached blocks so the next test starts with an
    # unfragmented pool. Aggregate flush failures too (nothing hidden).
    flush_errors = []
    for idx in range(num_devices):
        device = torch.device("rbln", idx)
        rbln_log_debug(f"Releasing RBLN caching allocator blocks on {device}")
        try:
            torch_rbln.memory.empty_cache(device)
        except Exception as exc:  # noqa: BLE001 - aggregated and re-raised below
            flush_errors.append(f"empty_cache(rbln:{idx}): {exc!r}")
    if flush_errors:
        raise RuntimeError("RBLN caching-allocator flush failed:\n  " + "\n  ".join(flush_errors))


# =============================================================================
# Global-state leak guards (autouse): keep one test's process-scoped mutations
# from leaking into later tests on the same xdist worker.
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def restore_current_device():
    """Restore the RBLN device selection and the lazy-init flag after each test, so a test
    that called set_device() cannot leak either into later tests on the same worker.

    Only re-issue set_device when the index actually changed -- calling it unconditionally
    would itself flip ``_initialized`` False->True and leak that. Restore index and flag in
    lockstep; if the index restore fails, let it propagate (a half-restored device would
    silently mislead later tests) rather than rolling back only the flag."""
    # The `torch_rbln.device` package re-exports a `device` class that shadows the
    # `device` submodule, so reach the module (which owns the `_initialized` global that
    # set_device mutates) via sys.modules rather than a `from ... import device`.
    _dev = sys.modules[torch_rbln.device.set_device.__module__]

    # Fail loud on a snapshot failure too: device_count() returns 0 (never raises) when there
    # is no device, so an exception here is a real fault (e.g. malformed RBLN_* config) that
    # should surface, not be swallowed into a silent no-op teardown.
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
        raise pytest.UsageError(f"Stale _REBEL_XFAILS keys (file collected, name unmatched): {stale}")
