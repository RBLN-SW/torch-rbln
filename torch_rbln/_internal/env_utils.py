"""torch_rbln utils for global & local environment

This module provides utility functions to check the status of various RBLN environment variables.
The functions determine whether certain modes or features are enabled based on the values of specific environment variables.
"""

import os
import sys


def _running_as_diagnose_module() -> bool:
    """True if the interpreter was invoked as python -m torch_rbln.diagnose."""
    if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "torch_rbln.diagnose":
        return True
    # runpy may set argv[0] to the module path in some versions
    if sys.argv and (str(sys.argv[0]).endswith("diagnose.py") or "torch_rbln.diagnose" in str(sys.argv)):
        return True
    # When argv is truncated (e.g. ['-m'] in debuggers/launchers), check runpy caller
    import inspect

    for frame_info in inspect.getouterframes(inspect.currentframe()):
        try:
            if "runpy" not in (frame_info.filename or ""):
                continue
            if frame_info.frame.f_locals.get("mod_name") == "torch_rbln.diagnose":
                return True
        except Exception:
            pass
    return False


def is_fallback_disabled(category: str) -> bool:
    """
    Check if a specific fallback category is disabled via `TORCH_RBLN_DISABLE_FALLBACK` environment variable.

    Args:
        category: One of 'all', 'compile_error', 'non_blocking_copy', 'unsupported_op'.

    Returns:
        True if the given category is disabled, either by name or via 'all'.
    """
    import torch_rbln._C

    return torch_rbln._C._is_fallback_disabled(category)


def is_rbln_deploy() -> bool:
    """
    Check if the RBLN deployment mode is enabled.

    Returns:
        bool: True if the deployment mode is enabled (environment variable TORCH_RBLN_DEPLOY is "ON"), False otherwise.
    """
    return os.getenv("TORCH_RBLN_DEPLOY") == "ON"


def use_device_group_tensor_parallel_size() -> bool:
    """
    Check if eager mode ops should use device group tensor parallel size instead of tp_size=1.

    By default, eager mode ops use tp_size=1. When this returns True, eager mode ops will
    follow the logical device size (RBLN_NPUS_PER_DEVICE) like torch.compile operations do.

    Returns:
        bool: True if eager mode ops should use logical device tensor parallel size
              (environment variable TORCH_RBLN_USE_DEVICE_TP is "ON"), False otherwise.
    """
    return os.getenv("TORCH_RBLN_USE_DEVICE_TP") == "ON"


def use_tp_failover() -> bool:
    """
    Check if tensor parallel failover is enabled.

    When enabled, if a RuntimeError occurs during execution with tensor parallel size > 1,
    the system will automatically retry with tp_size=1.

    Returns:
        bool: True if failover is enabled (environment variable TORCH_RBLN_USE_TP_FAILOVER is "ON"),
              False otherwise (default).
    """
    return os.getenv("TORCH_RBLN_USE_TP_FAILOVER") == "ON"


def is_diagnose_mode() -> bool:
    """
    Check if torch-rbln is running in diagnose-only mode (skip backend init).

    When True, the package skips loading torch and native libs so that
    ``python -m torch_rbln.diagnose`` can run even when they are missing or broken.

    Returns True if either:
    - TORCH_RBLN_DIAGNOSE is set to "1", or
    - the process was started as ``python -m torch_rbln.diagnose``.

    Returns:
        bool: True if in diagnose-only mode, False otherwise.
    """
    return os.environ.get("TORCH_RBLN_DIAGNOSE", "") == "1" or _running_as_diagnose_module()
