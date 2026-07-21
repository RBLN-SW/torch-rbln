"""Device architecture detection via RBLN NPU name."""

import functools


__all__ = ["get_device_arch", "is_atom_device", "is_rebel_device"]


def _arch_from_npu_name(name: str) -> str:
    """Map an RBLN NPU name to a device family: ``RBLN-CA*`` -> ``"atom"``,
    ``RBLN-CR*`` -> ``"rebel"``, anything else -> ``"unknown"``.
    """
    name = name.upper()
    if name.startswith("RBLN-CA"):
        return "atom"
    if name.startswith("RBLN-CR"):
        return "rebel"
    return "unknown"


@functools.lru_cache(maxsize=1)
def get_device_arch() -> str:
    """Identify the current NPU family (``"atom"``/``"rebel"``/``"unknown"``) via
    ``get_npu_name`` from ``rebel-compiler`` (cached). Returns ``"unknown"`` if the
    NPU name can't be queried.
    """
    try:
        from rebel.device_info import get_npu_name

        return _arch_from_npu_name(get_npu_name(0) or "")
    except Exception:
        return "unknown"


def is_atom_device() -> bool:
    """True on the ATOM lineup (``RBLN-CA*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "atom"


def is_rebel_device() -> bool:
    """True on the REBEL lineup (``RBLN-CR*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "rebel"
