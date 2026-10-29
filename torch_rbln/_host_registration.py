"""Temporary admission control for the old driver's finite page-table pool.

Public Python registrations share one ledger. Native _C entry points are the
unchecked implementation, not an alternative public registration API.
"""

import contextlib
import logging
import os
import threading
import weakref
from bisect import bisect_right
from dataclasses import dataclass
from operator import attrgetter

import torch_rbln._C
from torch_rbln._internal.env_utils import host_registration_profile


_PAGE = 4096
_HUGE = 2 << 20
_WINDOW = 1 << 30
_TABLE_LIMIT = (16 << 30) // _HUGE
_VA_LIMIT = 256 << 30
_lock = threading.RLock()
_logger = logging.getLogger(__name__)
_region_start = attrgetter("ptr")


@dataclass
class _Budget:
    tables: int = 0
    va_bytes: int = 0

    def reserve(self, tables: int, va_bytes: int) -> None:
        if self.tables + tables > _TABLE_LIMIT or self.va_bytes + va_bytes > _VA_LIMIT:
            msg = "Host registration admission changed during preparation"
            raise RuntimeError(msg)
        self.tables += tables
        self.va_bytes += va_bytes


@dataclass
class _Region:
    ptr: int
    size: int
    tables: int
    registered: bool = False

    @property
    def va_bytes(self) -> int:
        return ((self.ptr % _PAGE + self.size + _HUGE - 1) // _HUGE) * _HUGE


_budgets: dict[int, _Budget] = {}
_groups: dict[tuple[int, int], "_RegistrationState"] = {}
_forked_with_registrations = False


def _after_fork() -> None:
    global _lock, _forked_with_registrations
    # A lock held by a vanished parent thread must never strand the child.
    # Device registrations themselves cannot be inherited by a new process.
    _lock = threading.RLock()
    _forked_with_registrations = bool(_groups)


os.register_at_fork(after_in_child=_after_fork)


def _plan(ptr: int, size: int, huge: list[tuple[int, int]], tables: int, va_bytes: int) -> list[_Region]:
    """Select whole device-table windows; only proven PMD mappings are free.

    An unaligned head gets its own registration: merging it into an aligned body
    would destroy host/DVA alignment and invalidate the body's huge-page credit.
    """
    end = ptr + size
    selected: list[_Region] = []
    huge_index = 0
    cursor = ptr
    while cursor < end:
        stop = min(end, (cursor // _HUGE + 1) * _HUGE)
        while huge_index < len(huge) and huge[huge_index][0] + huge[huge_index][1] <= cursor:
            huge_index += 1
        is_huge = (
            cursor % _HUGE == 0
            and stop - cursor == _HUGE
            and huge_index < len(huge)
            and huge[huge_index][0] <= cursor
            and stop <= huge[huge_index][0] + huge[huge_index][1]
        )
        cost = 0 if is_huge else 1
        if cost <= tables and va_bytes >= _HUGE:
            if selected and selected[-1].ptr % _HUGE == 0 and selected[-1].ptr + selected[-1].size == cursor:
                prev = selected[-1]
                selected[-1] = _Region(prev.ptr, stop - prev.ptr, prev.tables + cost)
            else:
                selected.append(_Region(cursor, stop - cursor, cost))
            tables -= cost
            va_bytes -= _HUGE
        cursor = stop
    return selected


class _RegistrationState:
    def __init__(self, ptr: int, size: int, device: int, owner: object | None, *, managed: bool) -> None:
        self.ptr = ptr
        self.size = size
        self.device = device
        self.owner = owner
        self.managed = managed
        self.pid = os.getpid()
        self.regions: list[_Region] = []
        self.closed = False

    def covers(self, ptr: int, size: int) -> bool:
        if self.pid != os.getpid() or size <= 0:
            return False
        index = bisect_right(self.regions, ptr, key=_region_start) - 1
        return (
            index >= 0
            and self.regions[index].registered
            and ptr + size <= self.regions[index].ptr + self.regions[index].size
        )

    def close(self) -> None:
        with _lock:
            if self.closed:
                return
            if self.pid != os.getpid():
                msg = "A host registration cannot be released in a forked child"
                raise RuntimeError(msg)
            error = None
            remaining = []
            processed = 0
            try:
                for region in self.regions:
                    region.registered = False
                    try:
                        torch_rbln._C._host_unregister(self.device, region.ptr)
                    except RuntimeError as exc:
                        # The runtime can erase its entry before a failed unpin.
                        # Retain the owner and charge until release is confirmed.
                        remaining.append(region)
                        if error is None:
                            error = exc
                    else:
                        budget = _budgets[self.device]
                        budget.tables -= region.tables
                        budget.va_bytes -= region.va_bytes
                    processed += 1
            finally:
                # Also retain unattempted entries on interruption. Avoid repeated
                # list removal when a fragmented pool has many registrations.
                self.regions = remaining + self.regions[processed:]
            if error is not None:
                raise error
            if _groups.get((self.device, self.ptr)) is self:
                del _groups[self.device, self.ptr]
            self.closed = True
            self.owner = None


def _finalize(state: _RegistrationState) -> None:
    try:
        state.close()
    except RuntimeError:
        _logger.exception("Host unregister failed; retaining the backing memory and registration budget")


class HostMemoryRegistration:
    """Owned result of a best-effort registration, including partial coverage.

    ``registered_ranges`` contains (address, size) pairs. ``covers`` tests whether
    one such range contains a transfer; adjacent registrations need not have
    adjacent device addresses. ``close`` drains transfers and unregisters every
    range before releasing the retained CPU tensor. Do not resize or remap that
    tensor while the registration is alive.
    """

    def __init__(self, state: _RegistrationState) -> None:
        self._state = state
        self._finalizer = weakref.finalize(self, _finalize, state)

    @property
    def registered_ranges(self) -> tuple[tuple[int, int], ...]:
        with _lock:
            if self._state.pid != os.getpid():
                return ()
            return tuple((r.ptr, r.size) for r in self._state.regions if r.registered)

    @property
    def registered_bytes(self) -> int:
        with _lock:
            if self._state.pid != os.getpid():
                return 0
            return sum(r.size for r in self._state.regions if r.registered)

    @property
    def fully_registered(self) -> bool:
        return not self._state.closed and self.registered_bytes == self._state.size

    @property
    def closed(self) -> bool:
        return self._state.closed

    def covers(self, ptr: int, size: int) -> bool:
        with _lock:
            return self._state.covers(ptr, size)

    def close(self) -> None:
        self._state.close()
        self._finalizer.detach()

    def __bool__(self) -> bool:
        msg = "Use fully_registered or registered_bytes for a host registration result"
        raise TypeError(msg)

    def __enter__(self) -> "HostMemoryRegistration":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _register(
    ptr: int,
    size: int,
    device: int,
    *,
    owner: object | None = None,
    partial: bool = False,
    available: bool = True,
) -> _RegistrationState:
    state = _RegistrationState(ptr, size, device, owner, managed=partial)
    if not available:
        return state
    use_huge = host_registration_profile() == "thp"
    with _lock:
        if _forked_with_registrations:
            msg = "Host registration was inherited across fork; start the worker with spawn"
            raise RuntimeError(msg)
        for (registered_device, start), other in _groups.items():
            if registered_device == device and ptr < start + other.size and start < ptr + size:
                msg = "Host range overlaps an existing or pending registration"
                raise RuntimeError(msg)
        budget = _budgets.setdefault(device, _Budget())
        _groups[device, ptr] = state
        try:
            cursor, end = ptr, ptr + size
            warned = False
            while cursor < end:
                if budget.va_bytes >= _VA_LIMIT or (not use_huge and budget.tables >= _TABLE_LIMIT):
                    if not partial:
                        state.close()
                    break
                stop = min(end, (cursor // _WINDOW + 1) * _WINDOW)
                snapshot = torch_rbln._C._HostPageSnapshot(cursor, stop - cursor, promote=partial) if use_huge else None
                with contextlib.closing(snapshot) if snapshot is not None else contextlib.nullcontext():
                    if snapshot is not None and snapshot.error and not warned:
                        _logger.warning(
                            "Cannot verify pinned huge pages (errno=%d); charging unverified memory as 4 KiB pages",
                            snapshot.error,
                        )
                        warned = True
                    huge = snapshot.huge_ranges if snapshot is not None else []
                    plan = _plan(cursor, stop - cursor, huge, _TABLE_LIMIT - budget.tables, _VA_LIMIT - budget.va_bytes)
                    if not partial and sum(r.size for r in plan) != stop - cursor:
                        state.close()
                        return state
                    for region in plan:
                        budget.reserve(region.tables, region.va_bytes)
                        state.regions.append(region)
                        # Record the attempt before entering the driver: an exception
                        # does not prove that a partially failed mapping was removed.
                        if not torch_rbln._C._host_register(device, region.ptr, region.size):
                            state.regions.remove(region)
                            budget.tables -= region.tables
                            budget.va_bytes -= region.va_bytes
                            state.close()
                            return state
                        region.registered = True
                cursor = stop
            if not state.regions:
                state.close()
        except BaseException:
            try:
                state.close()
            except RuntimeError:
                _logger.exception("Host registration cleanup failed; backing memory and budget remain reserved")
            raise
        else:
            return state


def register_strict(ptr: int, size: int, device: int) -> bool:
    state = _register(ptr, size, device)
    return not state.closed and sum(r.size for r in state.regions) == size


def unregister_strict(ptr: int, device: int) -> None:
    with _lock:
        state = _groups.get((device, ptr))
        if state is None:
            msg = "No host registration starts at this address on the device"
            raise RuntimeError(msg)
        if state.managed:
            msg = "Release a managed host registration with its close() method"
            raise RuntimeError(msg)
        state.close()


def register_managed(ptr: int, size: int, device: int, owner: object, *, available: bool) -> HostMemoryRegistration:
    state = _register(ptr, size, device, owner=owner, partial=True, available=available)
    try:
        result = HostMemoryRegistration(state)
    except BaseException:
        _finalize(state)
        raise
    else:
        return result


def is_registered(ptr: int, size: int, device: int) -> bool:
    with _lock:
        return any(key[0] == device and state.covers(ptr, size) for key, state in _groups.items())
