import os

import pytest
import torch

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
