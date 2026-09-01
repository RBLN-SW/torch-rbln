"""Permutations run as compiled device programs (see RBLNCompiledPermute.h)."""

import threading

import torch

__all__ = ["compiled_permute"]

_lock = threading.Lock()
_entries: dict = {}
_verified: set = set()


def _entry(shape: tuple, dtype: torch.dtype, device: torch.device, dims: tuple, slot: int):
    """The compiled program for this permutation and the holder its runtime lands in."""
    key = (shape, dtype, device.index, dims, slot)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            # torch.compile caches on the code object, so each (dims, slot) is compiled from
            # its own source with the dims baked in as constants.
            body = "    return x.permute(" + repr(list(dims)) + ").contiguous()"
            source = "def compiled_permute(x):\n" + body + "\n"
            namespace: dict = {}
            exec(compile(source, f"<compiled_permute {dims} slot {slot}>", "exec"), {}, namespace)
            holder: list = []
            program = torch.compile(
                namespace["compiled_permute"],
                backend="rbln",
                dynamic=False,
                options={"use_static_output": True, "_runtime_holder": holder},
            )
            entry = _entries[key] = (program, holder)
        return entry


def compiled_permute(
    src: torch.Tensor, dims: list[int], slot: int = 0, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``src.permute(dims).contiguous()`` as a compiled device program, bit-exact for any dtype.

    With ``out`` the result lands in the caller's tensor; without it the program's own buffer is
    returned, borrowed until the next call with the same shape, dtype, ``dims`` and ``slot``.

    A program binds its operands on first use and rebinding a large buffer costs milliseconds,
    so a caller that alternates buffers (a double-buffered pipeline) gives each its own
    ``slot``; ``copy_``'s fast path does that by source address.

    Each program is checked against a host reference on its first use: the compiler miscomputes
    some permutations (``[2, 0, 1, 3]`` among them) without reporting anything, and one that
    fails the check raises here rather than returning wrong data.
    """
    if src.device.type != "rbln":
        raise ValueError("compiled_permute: an RBLN tensor is required")
    if not src.is_contiguous():
        raise ValueError("compiled_permute: a contiguous tensor is required")
    if sorted(dims) != list(range(src.dim())):
        raise ValueError(f"compiled_permute: {dims} is not a permutation of {src.dim()} dims")
    if dims[-1] != src.dim() - 1:
        raise ValueError("compiled_permute: the last dimension must stay last")
    shape = [src.shape[d] for d in dims]
    if out is not None and (list(out.shape) != shape or out.dtype != src.dtype):
        raise ValueError(f"compiled_permute: out must be {shape} of {src.dtype}")

    program, holder = _entry(tuple(src.shape), src.dtype, src.device, tuple(dims), slot)
    if out is None:
        result = program(src)
    else:
        if not holder:
            program(src)  # compiles, and publishes the runtime this call needs
        holder[0].run(src, out=[out])
        result = out

    key = (tuple(src.shape), src.dtype, tuple(dims))
    if key not in _verified:
        if not torch.equal(result.cpu(), src.cpu().permute(dims)):
            raise RuntimeError(
                f"compiled_permute: the compiler miscomputes permute({dims}) for shape "
                f"{tuple(src.shape)}; use the eager path for this permutation"
            )
        with _lock:
            _verified.add(key)
    return result
