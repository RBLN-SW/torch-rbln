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

    ``"unknown"`` is what a host with no NPU gets on its own: ``get_npu_name``
    answers ``None`` for an index no device claims, which maps to ``"unknown"``
    without raising. So nothing here has to catch that case.

    A ``get_npu_name`` that moved is the opposite, and must not arrive as the
    same answer. Every caller of this is an architecture gate -- ``xfail_atom``,
    ``xfail_rebel``, the per-lineup branches in the model tests -- so one
    ``"unknown"`` turns all of them off at once, and the suite goes on
    asserting something other than what it says it does. Let it raise.
    """
    from rebel.device_info import get_npu_name

    return _arch_from_npu_name(get_npu_name(0) or "")


def is_atom_device() -> bool:
    """True on the ATOM lineup (``RBLN-CA*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "atom"


def is_rebel_device() -> bool:
    """True on the REBEL lineup (``RBLN-CR*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "rebel"
