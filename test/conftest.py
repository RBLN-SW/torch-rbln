import os

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
    """Release the RBLN caching allocator's cached blocks before each test.

    Without this, freed small blocks from prior tests accumulate in the
    caching allocator and fragment the pool, so later tests can fail to
    allocate large contiguous blocks. Mirrors the TorchDynamo reset above;
    opt out per-test with the ``no_caching_allocator_reset`` marker.
    """
    if request.node.get_closest_marker("no_caching_allocator_reset"):
        rbln_log_debug("Skipping RBLN caching allocator reset")
        return

    try:
        num_devices = torch_rbln.device.device_count()
    except Exception as exc:
        rbln_log_debug(f"device_count() failed; skipping caching allocator reset: {exc}")
        return

    for idx in range(num_devices):
        device = torch.device("rbln", idx)
        rbln_log_debug(f"Releasing RBLN caching allocator blocks on {device}")
        try:
            # Drain in-flight device work first: empty_cache() without a prior sync can race
            # async ops left by a previous test and surface later as a misattributed error.
            torch.rbln.synchronize(idx)
            torch_rbln.memory.empty_cache(device)
        except Exception as exc:
            rbln_log_debug(f"empty_cache({device}) failed: {exc}")


# =============================================================================
# Global-state leak guards (autouse): keep one test's process-scoped mutations
# from leaking into later tests on the same xdist worker.
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def restore_current_device():
    """Restore the selected RBLN device after each test, so a test that calls set_device()
    cannot leak the selection into later tests on the same worker."""
    try:
        saved = torch.rbln.current_device() if torch_rbln.device.device_count() > 0 else None
    except Exception:
        saved = None
    yield
    if saved is not None:
        try:
            torch.rbln.set_device(saved)
        except Exception as exc:
            rbln_log_debug(f"restore current device -> {saved} failed: {exc}")


@pytest.fixture(scope="function", autouse=True)
def keep_torch_compile_patches():
    """Re-apply torch_rbln's import-time torch.compile / torch._dynamo.reset patches if a
    test removed or swapped them, so later tests (and the autouse reset_dynamo above) keep
    the RBLN wrappers rather than the bare originals."""
    yield
    import torch_rbln._internal.monkey_patches as mp

    if not (mp._torch_compile_patched and mp._torch_dynamo_reset_patched):
        rbln_log_debug("Re-applying RBLN torch.compile patches leaked-off by a prior test")
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
    if config.getoption("dist", "no") != "no":
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
