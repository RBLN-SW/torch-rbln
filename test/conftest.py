import os

import pytest
import torch

import torch_rbln
from test.utils import set_deterministic_seeds
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
            torch_rbln.memory.empty_cache(device)
        except Exception as exc:
            rbln_log_debug(f"empty_cache({device}) failed: {exc}")


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
