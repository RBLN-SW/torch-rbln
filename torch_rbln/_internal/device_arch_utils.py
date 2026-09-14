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
    ``get_npu_name`` from ``rebel-compiler`` (cached).

    ``"unknown"`` is reserved for a host with no NPU: the runtime's query API
    answers ``None`` for an index no device claims and never raises for it, and
    ``None`` maps to ``"unknown"`` without any catch here.

    Anything else -- the import failing, the lookup raising -- propagates.
    Every caller of this is an architecture gate (``xfail_atom``,
    ``xfail_rebel``, the per-lineup branches in the model tests), and a
    swallowed failure would turn all of them off at once while the suite
    goes on passing.
    """
    from rebel.device_info import get_npu_name

    return _arch_from_npu_name(get_npu_name(0) or "")


def is_atom_device() -> bool:
    """True on the ATOM lineup (``RBLN-CA*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "atom"


def is_rebel_device() -> bool:
    """True on the REBEL lineup (``RBLN-CR*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "rebel"
