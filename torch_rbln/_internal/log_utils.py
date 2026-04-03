import logging
import os
import sys
import warnings


__all__ = [
    "rbln_log_debug",
    "rbln_is_debug_enabled",
    "rbln_log_info",
    "rbln_log_warn",
    "rbln_log_error",
    "rbln_log_cpu_fallback",
]


def _get_log_level() -> int:
    """
    Retrieves the current log level from `TORCH_RBLN_LOG_LEVEL` environment variable.

    Returns:
        int: The current log level, defaulting to logging.WARNING if `TORCH_RBLN_LOG_LEVEL` is not set.
    """
    env_level = os.getenv("TORCH_RBLN_LOG_LEVEL")
    env_legacy = os.getenv("TORCH_RBLN_LOG")

    # `TORCH_RBLN_LOG_LEVEL` takes precedence over deprecated `TORCH_RBLN_LOG`.
    level = "WARNING"
    if env_level is not None:
        level = env_level
    elif env_legacy is not None:
        warnings.warn(
            "The environment variable `TORCH_RBLN_LOG` is deprecated, please use `TORCH_RBLN_LOG_LEVEL` instead"
        )
        level = env_legacy

    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(level)
    if log_level is None:
        raise ValueError(f"Invalid TORCH_RBLN_LOG_LEVEL `{level}`, expected one of: DEBUG, INFO, WARNING, ERROR")
    return log_level


_logger = logging.getLogger("torch-rbln")
_logger.propagate = False

if not _logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("[TORCH-RBLN][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

_logger.setLevel(_get_log_level())


def rbln_is_debug_enabled() -> bool:
    """True if ``TORCH_RBLN_LOG_LEVEL`` (or legacy ``TORCH_RBLN_LOG``) enables DEBUG on the ``torch-rbln`` logger."""
    return _logger.isEnabledFor(logging.DEBUG)


def rbln_log_debug(msg: str):
    """
    Logs a message at the DEBUG level.

    Args:
        msg (str): The debug message to log.
    """
    _logger.debug(msg)


def rbln_log_info(msg: str):
    """
    Logs a message at the INFO level.

    Args:
        msg (str): The informational message to log.
    """
    _logger.info(msg)


def rbln_log_warn(msg: str):
    """
    Logs a message at the WARNING level.

    Args:
        msg (str): The warning message to log.
    """
    _logger.warning(msg)


def rbln_log_error(msg: str):
    """
    Logs a message at the ERROR level.

    Args:
        msg (str): The error message to log.
    """
    _logger.error(msg)


def rbln_log_cpu_fallback(op_name: str):
    """
    Logs a message indicating that a specified operation ran on CPU instead of RBLN.

    This function logs an info-level message that includes the name of the operation that ran on CPU.
    It also generates a UserWarning that includes the file location where the warning is issued. The
    warning is formatted to indicate that a fallback to CPU execution is being used for the specified
    operation. The message is logged when `TORCH_RBLN_LOG_LEVEL` is set to `INFO` or lower.

    Example output:
        [2026-01-01 00:00:00.000][I] `aten::mul` op ran on CPU instead of RBLN
        /llama/modeling_llama.py:73: UserWarning: TRACE
          result, result_shape = mul_rbln(*args, **kwargs)

    Parameters:
        op_name (str): The name of the operation that ran on CPU.
    """
    import torch_rbln._C

    torch_rbln._C._log_cpu_fallback(op_name)
