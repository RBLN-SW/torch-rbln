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
    """Identify the NPU family (``"atom"``/``"rebel"``/``"unknown"``) of RBLN
    device 0 (cached).

    ``"unknown"`` is reserved for a host with no NPU to ask: no device, or the
    host-backed ``RBLN_DUMMY_DEVICE``.

    Anything else -- the lookup raising -- propagates. Every caller of this is
    an architecture gate (``xfail_atom``, ``xfail_rebel``, the per-lineup
    branches in the model tests), and a swallowed failure would turn all of
    them off at once while the suite goes on passing.
    """
    import torch_rbln._C as _C

    if _C.is_dummy_device() or _C.device_count() == 0:
        return "unknown"
    return _arch_from_npu_name(_C.get_device_properties(0).name)


def is_atom_device() -> bool:
    """True on the ATOM lineup (``RBLN-CA*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "atom"


def is_rebel_device() -> bool:
    """True on the REBEL lineup (``RBLN-CR*``); thin wrapper over :func:`get_device_arch`."""
    return get_device_arch() == "rebel"
