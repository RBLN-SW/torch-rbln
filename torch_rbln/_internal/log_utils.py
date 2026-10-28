import functools
import logging
import os
import sys
import warnings


__all__ = [
    "rbln_is_debug_enabled",
    "rbln_log_debug",
    "rbln_log_error",
    "rbln_log_info",
    "rbln_log_warn",
    "rbln_log_warn_once",
    "rbln_record_cpu_fallback",
]


def _get_log_level() -> int:
    """Retrieves the current log level from `TORCH_RBLN_LOG_LEVEL` environment variable.

    Returns:
        int: The current log level, defaulting to logging.WARNING if
            `TORCH_RBLN_LOG_LEVEL` is not set.
    """
    env_level = os.getenv("TORCH_RBLN_LOG_LEVEL")
    env_legacy = os.getenv("TORCH_RBLN_LOG")

    # `TORCH_RBLN_LOG_LEVEL` takes precedence over deprecated `TORCH_RBLN_LOG`.
    level = "WARNING"
    if env_level is not None:
        level = env_level
    elif env_legacy is not None:
        warnings.warn(
            "The environment variable `TORCH_RBLN_LOG` is deprecated, please use `TORCH_RBLN_LOG_LEVEL` instead",
            stacklevel=1,
        )
        level = env_legacy

    log_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(level)
    if log_level is None:
        msg = f"Invalid TORCH_RBLN_LOG_LEVEL `{level}`, expected one of: DEBUG, INFO, WARNING, ERROR"
        raise ValueError(msg)
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
    """True if ``TORCH_RBLN_LOG_LEVEL`` (or legacy ``TORCH_RBLN_LOG``) enables DEBUG on
    the ``torch-rbln`` logger.
    """
    return _logger.isEnabledFor(logging.DEBUG)


def rbln_log_debug(msg: str):
    """Logs a message at the DEBUG level.

    Args:
        msg (str): The debug message to log.
    """
    _logger.debug(msg)


def rbln_log_info(msg: str):
    """Logs a message at the INFO level.

    Args:
        msg (str): The informational message to log.
    """
    _logger.info(msg)


def rbln_log_warn(msg: str):
    """Logs a message at the WARNING level.

    Args:
        msg (str): The warning message to log.
    """
    _logger.warning(msg)


@functools.cache
def rbln_log_warn_once(msg: str):
    """Logs a message at the WARNING level, once per distinct message.

    Args:
        msg (str): The warning message to log.
    """
    _logger.warning(msg)


def rbln_log_error(msg: str):
    """Logs a message at the ERROR level.

    Args:
        msg (str): The error message to log.
    """
    _logger.error(msg)


def rbln_record_cpu_fallback(op_name: str, reason: str):
    """Context manager around one CPU fallback: ``with rbln_record_cpu_fallback(name,
    reason):`` then run the op on CPU inside the block.

    Entering counts the fallback per op, which ``torch.rbln.explain()`` reports, and
    when `TORCH_RBLN_LOG_LEVEL` is `INFO` or lower logs it with a UserWarning that shows
    the caller's location. Leaving adds the block's wall time to the same total as the
    C++ fallback paths (c10::rbln::CpuFallbackScope), once for nested fallbacks.

    Example output (at INFO):
        [2026-01-01 00:00:00.000][I] `aten::mul` op ran on CPU instead of RBLN
        /llama/modeling_llama.py:73: UserWarning: TRACE
          result, result_shape = mul_rbln(*args, **kwargs)

    Parameters:
        op_name (str): The name of the operation that ran on CPU.
        reason (str): Why it ran on CPU, such as the case ``is_cpu_fallback_cases``
            returned.
    """
    import torch_rbln._C  # noqa: PLC0415

    return torch_rbln._C._CpuFallbackScope(op_name, reason)
