"""``torch.rbln`` stream & event API (``torch.cuda`` parity).

Streams and events route through the RBLN ``PrivateUse1`` device-guard
implementation, so the generic ``torch.Stream`` / ``torch.Event`` already work with
``device="rbln"``; this module adds the ``torch.cuda``-style surface on top.

Not supported: event timing (:meth:`Event.elapsed_time` raises), stream priorities
(accepted but ignored), :meth:`torch.Stream.native_handle`, and native cross-device
event waits (they degrade to a host-side synchronize).
"""

from typing import Any, Optional  # noqa: UP035

import torch

import torch_rbln._C
from torch_rbln.device.device import _get_device_index, current_device, device


__all__ = [
    "Stream",
    "Event",
    "current_stream",
    "default_stream",
    "set_stream",
    "stream",
    "StreamContext",
]


def _as_rbln_device(dev: Any) -> torch.device:
    """Resolve ``dev`` (int / str / torch.device) to a concrete ``rbln:<idx>``."""
    return torch.device("rbln", _get_device_index(dev))


class Stream(torch.Stream):
    r"""An RBLN stream: an in-order sequence of device work.

    Work on the same stream runs in order; work on different streams may run
    concurrently. Mirrors :class:`torch.cuda.Stream`. ``priority`` is accepted for
    API parity but ignored (RBLN has no stream priorities).

    Streams come from a fixed per-device pool; past the pool size, new instances
    reuse earlier streams (as in :mod:`torch.cuda`).
    """

    def __new__(cls, device: Any = None, priority: int = 0, **kwargs: Any) -> "Stream":  # noqa: PYI034
        # With ids given (current_stream/default_stream), the base wraps them instead
        # of creating a stream.
        if "stream_id" in kwargs and "device_index" in kwargs and "device_type" in kwargs:
            return super().__new__(cls, priority=priority, **kwargs)
        dev = torch.device("rbln", current_device()) if device is None else _as_rbln_device(device)
        return super().__new__(cls, dev, priority=priority)

    def record_event(self, event: Optional["Event"] = None) -> "Event":
        """Record ``event`` on this stream, creating an :class:`Event` if ``None``.

        Overridden like :meth:`torch.cuda.Stream.record_event` so the event is an
        rbln one; the inherited method takes only a base :class:`torch.Event`.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    @property
    def priority(self) -> int:
        return 0

    @staticmethod
    def priority_range() -> tuple:
        """``(least_priority, greatest_priority)``; ``(0, 0)`` — no priority levels."""
        return (0, 0)

    def __repr__(self) -> str:
        return f"torch.rbln.Stream(device={self.device}, stream_id={self.stream_id})"


class Event(torch.Event):
    r"""An RBLN event: a recordable marker in a stream for cross-stream ordering and
    completion queries. Mirrors :class:`torch.cuda.Event`. ``enable_timing`` is
    accepted for API parity, but :meth:`elapsed_time` is unsupported.
    """

    def __new__(  # noqa: PYI034
        cls,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> "Event":
        return super().__new__(
            cls,
            device="rbln",
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    def elapsed_time(self, other: "Event") -> float:
        raise RuntimeError("RBLN does not support Event.elapsed_time (event timing is not supported).")

    def __repr__(self) -> str:
        return f"torch.rbln.Event(device={self.device})"


def current_stream(device: Any = None) -> Stream:
    """Return the currently selected :class:`Stream` (default: current device)."""
    idx = current_device() if device is None else _get_device_index(device)
    stream_id, device_index, device_type = torch_rbln._C._get_current_stream(idx)
    return Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)


def default_stream(device: Any = None) -> Stream:
    """Return the default :class:`Stream` (default: current device)."""
    idx = current_device() if device is None else _get_device_index(device)
    stream_id, device_index, device_type = torch_rbln._C._get_default_stream(idx)
    return Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)


def set_stream(stream: Optional[Stream]) -> None:
    """Set the current stream (no-op if ``None``). Prefer the :func:`stream` manager.

    Also selects the stream's device, matching :func:`torch.cuda.set_stream`.
    """
    if stream is None:
        return
    torch_rbln._C._exchange_stream(stream.stream_id, stream.device_index, stream.device_type)


class StreamContext:
    r"""Context-manager that makes a stream current, restoring the previous on exit.

    Mirrors :class:`torch.cuda.StreamContext`: the stream's device is selected for the
    duration. No-op if the stream is ``None``.
    """

    def __init__(self, stream: Optional[Stream]):
        self.stream = stream
        self.src_prev_stream: Optional[Stream] = None
        self.dst_prev_stream: Optional[Stream] = None

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return
        self.src_prev_stream = current_stream()
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = current_stream(cur_stream.device)
        set_stream(cur_stream)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        cur_stream = self.stream
        if cur_stream is None:
            return False
        if self.src_prev_stream is not None and self.src_prev_stream.device != cur_stream.device:
            set_stream(self.dst_prev_stream)
        set_stream(self.src_prev_stream)
        return False


def stream(stream: Optional[Stream]) -> StreamContext:
    """Wrap :class:`StreamContext` to select a given stream (no-op if ``None``)."""
    return StreamContext(stream)
