"""RBLN_DUMMY_DEVICE execution policy.

A dummy device (``RBLN_DUMMY_DEVICE``) is a host-backed stand-in for a real NPU: it
lets a no-NPU host construct device tensors, run host/device copies, and build
``.rbln`` artifacts with compile-only ``torch.compile``. It has **no NPU**, so it
cannot *execute* compute — neither an eager op nor a compiled graph.

This module is the single place that owns that policy. Every execution primitive
(the compiled-graph call in ``CompiledFunctionWrapper`` and the eager CPU-fallback
path) routes its "may I run here?" question through :func:`raise_if_dummy_execution`,
so the rule and its error message live in exactly one place.
"""


def is_dummy_device() -> bool:
    """True when host-backed ``RBLN_DUMMY_DEVICE`` mode is active (no NPU present).

    Runtime-free: it reads a flag set at device init and never opens a device, so it
    is safe to call on any dispatch path. Returns ``False`` if torch_rbln's native
    extension is unavailable.
    """
    try:
        import torch_rbln

        return bool(torch_rbln._C.is_dummy_device())
    except Exception:
        return False


def raise_if_dummy_execution(what: str, *, compile_only: bool = False) -> None:
    """Reject compute execution on ``RBLN_DUMMY_DEVICE`` — the single policy gate.

    Dummy mode is build-only: with no NPU it cannot run ``what`` (an eager op or a
    compiled graph). Rather than silently running on CPU or returning zeros, raise a
    clear error. ``compile_only`` builds are exempt — they only write the artifact and
    never consume its output. No-op when dummy mode is off.
    """
    if compile_only or not is_dummy_device():
        return
    raise RuntimeError(
        "RBLN_DUMMY_DEVICE has no NPU and cannot execute " + what + ". Dummy mode supports "
        "tensor construction, host/device copies, and building artifacts with "
        "torch.compile(options={'mode': ['compile_only']}); run on a host with a real NPU to execute."
    )
