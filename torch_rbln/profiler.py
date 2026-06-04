"""``torch.rbln.profile`` — surface HIDDEN torch-rbln overhead, with CAUSE.

A normal PyTorch op (``a.copy_(b)``, ``torch.cat``, an indexing write, ...) can
silently round-trip the host (NPU -> CPU -> NPU) or fall back to a CPU kernel
for reasons the user never asked for and cannot see. This profiler counts those
hidden events AND attributes WHY, so a developer can both tell "am I using
torch-rbln well" and know what to fix.

Two complementary, non-overlapping witnesses (so nothing is hidden, double
counted, or missed):
  * **torch-side** counters (this extension) — host bounces torch-rbln's OWN
    code path chose (a non-direct ``copy_`` taking the host path, a strided op
    falling back to CPU, a batched v2v dropping to per-entry), plus dispatch
    counters (cpu_fallback, recompile/miss).
  * **runtime-side** counter (rebel-compiler) — a v2v the caller issued as a
    *device* copy but the runtime silently serviced via its host-sync slow path.
    torch-side counters are structurally blind to this. Read via ``ctypes`` if
    the loaded ``librbln`` exposes it; otherwise reported as pending (never a
    false GREEN). Tagged BY CAUSE:
      - ``src_not_on_device``           : src wasn't device-resident yet.
      - ``src_on_device_but_unplannable``: src on device but the fast d2d plan
        was rejected (dtype conversion / misalignment / >30 physical chunks,
        e.g. a sharded KV layout).

Design invariants: ON == OFF in latency (every counter lives on an already-slow
branch; reads are lazy), and only user-actionable signals are in the verdict.

Usage::

    import torch

    with torch.rbln.profile() as p:
        model(x)
    print(p.report())  # verdict + cause
    p.dump()  # raw dict for CI gates
    p.verdict()  # {'status': ..., 'reasons': [...]}
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional  # noqa: UP035


__all__ = ["profile", "RBLNProfile"]


# Must match the BounceSite enum order in c10/rbln/RBLNProfiler.h.
_BOUNCE_SITES: tuple[tuple[str, str], ...] = (
    ("copy_d2d_host_bounce", "copy_: non-direct device->device round-tripped host (v2h+h2v)"),
    ("copy_h2d_staging", "copy_: cpu src needed a staging alloc + CPU copy before h2v"),
    ("copy_h2d_noncontig_dst", "copy_: non-contiguous rbln dst pulled to host then written back"),
    ("strided_v2v_cpu_fallback", "cat/index_select/index_copy/copy_: strided v2v fell back to a CPU op"),
    ("v2v_batch_to_per_entry", "batched v2v rejected by runtime -> per-entry loop (may host-bounce)"),
)
_HOST_BOUNCE_SITES = (0, 1, 2, 3)  # torch-side sites that are an actual host round-trip

# Must match the reason order in rebel-compiler vmemory_manager.cc (kNumV2VHiddenReasons).
# (name, actionable fix hint)
_RUNTIME_REASONS: tuple[tuple[str, str], ...] = (
    ("src_not_on_device", "src tensor was not on device yet -> establish device residency before the copy"),
    (
        "src_device_only_real_d2h",
        "src was device-only; a REAL device->host transfer occurred (layout unplannable: dtype/align/>30 chunks, e.g. sharded KV) -> align dtype or native strided v2v",
    ),
    (
        "src_synced_host_served",
        "src already on host+device; host memcpy with NO transfer (dst becomes host-latest) -> usually benign",
    ),
)


def _read_bounces() -> list[tuple[int, int]]:
    import torch_rbln._C as _C

    return list(_C._profiler_dump_bounces())


def _read_dispatch() -> tuple[int, int, int, int, int, int]:
    import torch_rbln._C as _C

    return tuple(_C._dispatch_shim_diag_dump())


# ---------------------------------------------------------------------------
# Runtime-side (rebel-compiler) hidden-d2h counter, read via ctypes from the
# already-loaded librbln. Resolved once and cached. Absent on older runtimes
# (then the residency signal degrades to "pending", never a false clean).
# ---------------------------------------------------------------------------
_runtime_fn = None
_runtime_resolved = False


def _runtime_hidden_d2h_fn():
    global _runtime_fn, _runtime_resolved
    if _runtime_resolved:
        return _runtime_fn
    _runtime_resolved = True
    for loader in (lambda: ctypes.CDLL(None), lambda: ctypes.CDLL("librbln.so")):
        try:
            lib = loader()
            fn = lib.rbln_prof_get_v2v_hidden_d2h
        except (OSError, AttributeError):
            continue
        fn.restype = None
        fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        _runtime_fn = fn
        return fn
    _runtime_fn = None
    return None


def _read_runtime_hidden_d2h() -> Optional[list[tuple[int, int]]]:
    """Per-reason [(count, bytes), ...] in _RUNTIME_REASONS order, or None if the
    loaded runtime does not expose the counter."""
    fn = _runtime_hidden_d2h_fn()
    if fn is None:
        return None
    n = len(_RUNTIME_REASONS)
    counts = (ctypes.c_uint64 * n)()
    nbytes = (ctypes.c_uint64 * n)()
    fn(counts, nbytes, n)
    return [(int(counts[i]), int(nbytes[i])) for i in range(n)]


class RBLNProfile:
    """A profiling region. Use via :func:`profile` as a context manager, or
    drive manually with :meth:`start` / :meth:`stop`. All numbers are *deltas*."""

    def __init__(self) -> None:
        self._b0: list[tuple[int, int]] | None = None
        self._d0: tuple[int, ...] | None = None
        self._rt0: Optional[list[tuple[int, int]]] = None
        self._bounces: list[tuple[int, int]] | None = None
        self._dispatch: tuple[int, ...] | None = None
        self._runtime: Optional[list[tuple[int, int]]] = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> RBLNProfile:
        self._b0 = _read_bounces()
        self._d0 = _read_dispatch()
        self._rt0 = _read_runtime_hidden_d2h()
        return self

    def stop(self) -> RBLNProfile:
        assert self._b0 is not None and self._d0 is not None, "start() before stop()"
        b1 = _read_bounces()
        d1 = _read_dispatch()
        self._bounces = [(c1 - c0, by1 - by0) for (c0, by0), (c1, by1) in zip(self._b0, b1)]
        self._dispatch = tuple(x1 - x0 for x0, x1 in zip(self._d0, d1))
        rt1 = _read_runtime_hidden_d2h()
        if self._rt0 is not None and rt1 is not None:
            self._runtime = [(c1 - c0, b1_ - b0) for (c0, b0), (c1, b1_) in zip(self._rt0, rt1)]
        else:
            self._runtime = None
        return self

    def __enter__(self) -> RBLNProfile:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- readout -------------------------------------------------------------
    def dump(self) -> dict[str, Any]:
        """Raw deltas as a dict (stable keys for CI gates)."""
        assert self._bounces is not None and self._dispatch is not None, "region not stopped"
        per_site = {name: {"count": c, "bytes": by} for (name, _desc), (c, by) in zip(_BOUNCE_SITES, self._bounces)}
        total_count = sum(self._bounces[i][0] for i in _HOST_BOUNCE_SITES)
        total_bytes = sum(self._bounces[i][1] for i in _HOST_BOUNCE_SITES)
        n_total, n_fallback, n_warm_hit, n_miss, _ns_warm_hit, _ns_miss = self._dispatch

        if self._runtime is not None:
            by_reason = {}
            rt_count = rt_bytes = 0
            for (name, _fix), (c, b) in zip(_RUNTIME_REASONS, self._runtime):
                by_reason[name] = {"count": c, "bytes": b}
                rt_count += c
                rt_bytes += b
            runtime_residency = {
                "available": True,
                "total_count": rt_count,
                "total_bytes": rt_bytes,
                "by_reason": by_reason,
            }
        else:
            runtime_residency = {"available": False}

        pending = ["device-idle bubbles (TIME)", "memory peak / waste (GAUGE)"]
        if self._runtime is None:
            pending.insert(0, "hidden_d2h / device-residency (STATE) — needs a recent rebel-compiler")

        return {
            "hidden_host_bounce": {
                "by_site": per_site,
                "total_count": total_count,
                "total_bytes": total_bytes,
            },
            "dispatch": {
                "total": n_total,
                "cpu_fallback": n_fallback,
                "warm_hit": n_warm_hit,
                "recompile_miss": n_miss,
            },
            "runtime_residency": runtime_residency,
            "pending_runtime_signals": pending,
        }

    def verdict(self) -> dict[str, Any]:
        """One top-line status + the evidence (with CAUSE) behind it."""
        d = self.dump()
        hb = d["hidden_host_bounce"]
        disp = d["dispatch"]
        rr = d["runtime_residency"]
        reasons: list[str] = []
        status = "GREEN"

        if hb["total_count"] > 0:
            status = "RED"
            offenders = sorted(
                ((n, v["count"], v["bytes"]) for n, v in hb["by_site"].items() if v["count"]),
                key=lambda t: t[2],
                reverse=True,
            )
            for name, c, by in offenders:
                reasons.append(f"{name}: {c}x, {by / 1e6:.1f} MB host round-trip")

        runtime_hidden = rr["total_count"] if rr["available"] else 0
        if runtime_hidden > 0:
            status = "RED"
            fix = dict(_RUNTIME_REASONS)
            for name, vv in rr["by_reason"].items():
                if vv["count"]:
                    reasons.append(
                        f"runtime hidden d2h [{name}]: {vv['count']}x, {vv['bytes'] / 1e6:.1f} MB — cause: {fix[name]}"
                    )

        if disp["recompile_miss"] > 0:
            status = "RED" if status == "RED" else "AMBER"
            reasons.append(f"recompile/cold-compile: {disp['recompile_miss']}x (bucket shapes if steady-state)")
        if disp["cpu_fallback"] > 0:
            status = "RED" if status == "RED" else "AMBER"
            reasons.append(f"cpu_fallback: {disp['cpu_fallback']}x (op ran on CPU, not NPU)")
        if not reasons:
            reasons.append("no hidden host bounce, no CPU fallback, no recompile")

        note = (
            "torch-side + runtime residency (with cause) reconciled"
            if rr["available"]
            else "torch-side signals only; runtime residency (hidden d2h) not exposed by the loaded "
            "librbln — install a recent rebel-compiler to light it up. Idle/memory remain pending."
        )
        return {
            "status": status,
            "hidden_host_bounces": hb["total_count"],
            "hidden_host_bounce_bytes": hb["total_bytes"],
            "runtime_hidden_d2h": runtime_hidden if rr["available"] else None,
            "runtime_hidden_by_reason": rr.get("by_reason") if rr["available"] else None,
            "cpu_fallbacks": disp["cpu_fallback"],
            "recompiles": disp["recompile_miss"],
            "reasons": reasons,
            "note": note,
        }

    def report(self) -> str:
        v = self.verdict()
        d = self.dump()
        rr = d["runtime_residency"]
        rt = f"{v['runtime_hidden_d2h']}" if rr["available"] else "n/a"
        lines = [
            f"RBLN PROFILE  |  VERDICT: {v['status']}  |  hidden host bounces: "
            f"{v['hidden_host_bounces']} ({v['hidden_host_bounce_bytes'] / 1e6:.1f} MB)  |  "
            f"runtime hidden d2h: {rt}  |  cpu_fallback: {v['cpu_fallbacks']}  recompile: {v['recompiles']}",
        ]
        for r in v["reasons"]:
            lines.append(f"  - {r}")
        hb = d["hidden_host_bounce"]["by_site"]
        nonzero = {n: vv for n, vv in hb.items() if vv["count"]}
        if nonzero:
            lines.append("  hidden-bounce sites (torch-side):")
            for name, vv in sorted(nonzero.items(), key=lambda t: t[1]["bytes"], reverse=True):
                lines.append(f"    {name:<28} count={vv['count']:<8} bytes={vv['bytes'] / 1e6:.2f} MB")
        if rr["available"]:
            rt_nonzero = {n: vv for n, vv in rr["by_reason"].items() if vv["count"]}
            if rt_nonzero:
                lines.append("  runtime hidden d2h by cause:")
                for name, vv in sorted(rt_nonzero.items(), key=lambda t: t[1]["bytes"], reverse=True):
                    lines.append(f"    {name:<32} count={vv['count']:<8} bytes={vv['bytes'] / 1e6:.2f} MB")
        lines.append(f"  note: {v['note']}")
        return "\n".join(lines)


def profile() -> RBLNProfile:
    """Return a profiling region usable as a context manager."""
    return RBLNProfile()
