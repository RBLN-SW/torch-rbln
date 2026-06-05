"""``torch.rbln.profile`` — surface HIDDEN torch-rbln overhead, with CAUSE.

A normal PyTorch op (``a.copy_(b)``, ``torch.cat``, an indexing write, a forward
pass) can silently round-trip the host (NPU->CPU->NPU), fall back to a CPU
kernel, recompile, or leave the NPU idle — for reasons the user never asked for
and cannot see. This profiler counts those hidden events, attributes the cause,
and renders a torch.profiler-style report.

Signals (every counter sits on an already-slow point — a host DMA or a fallback
branch — never the fast device path; reads are lazy, so wrapping a run with
profile() does not change its latency):

  host bounce (torch-side, aten intent) : non-direct copy_ / strided fallback / etc.
  runtime hidden-d2h cause (v2v slow)   : why a device v2v fell to host (src state)
  dispatch                              : cpu_fallback, recompile/miss, warm-hit

Plus one RESOURCE gauge (NOT a hidden-overhead signal — kept because it is
accurate and complete: every device allocation routes through BufferAllocator):

  device memory : current live + peak high-water

Scope: this profiler surfaces ONLY hidden host overhead a user issued as a normal
op and cannot see. Generic dispatch/utilization counters (command-stream count,
device-idle, total host-traffic bytes) are a different profiler's job
(torch.profiler / nsys), are out of scope, and on the executed path live in the
TVM runtime where they read ~0 here — surfacing them would mislead. The one
hidden signal still missing — a side-effect host-sync on the TVM graph-exec path
— is a tracked follow-up (it, not the excluded counters, is what TVM-runtime
instrumentation is for).

Usage::

    import torch

    with torch.rbln.profile() as p:
        model(x)
    print(p.report())  # torch.profiler-style table, verdict first
    p.dump()  # dict for CI gates
    p.verdict()  # {'status': 'GREEN'|'AMBER'|'RED', ...}
"""

from __future__ import annotations

import ctypes
import time
from typing import Any, Optional  # noqa: UP035


__all__ = ["profile", "RBLNProfile"]


# Must match the BounceSite enum order in c10/rbln/RBLNProfiler.h.
_BOUNCE_SITES: tuple[tuple[str, str], ...] = (
    ("copy_d2d_host_bounce", "copy_: non-direct device->device round-tripped host"),
    ("copy_h2d_staging", "copy_: cpu src staged before h2v"),
    ("copy_h2d_noncontig_dst", "copy_: non-contiguous rbln dst pulled to host"),
    ("strided_v2v_cpu_fallback", "cat/index/copy_: strided v2v fell back to CPU"),
    ("v2v_batch_to_per_entry", "batched v2v rejected -> per-entry"),
)
_HOST_BOUNCE_SITES = (0, 1, 2, 3)

# Must match the reason order in rebel-compiler vmemory_manager.cc.
_RUNTIME_REASONS: tuple[tuple[str, str], ...] = (
    ("src_not_on_device", "src not on device yet -> establish device residency before the copy"),
    ("src_device_only_real_d2h", "src device-only; real d2h (layout unplannable: dtype/align/>30 chunks)"),
    ("src_synced_host_served", "src already on host+device; host memcpy, no transfer (usually benign)"),
)

# Static cause -> one-line REMEDY ("what to change"). Read only at report()/
# verdict() time, so it adds zero runtime cost. The runtime v2v-slow reasons
# carry their own fix text in _RUNTIME_REASONS and are not duplicated here.
_REMEDY: dict[str, str] = {
    "copy_d2d_host_bounce": (
        "non-contiguous device->device copy went via host; make it contiguous "
        "(e.g. KV layout) or route it through the on-device v2v engine"
    ),
    "copy_h2d_staging": "cpu source staged before h2v; keep the source on-device, or stage it once and reuse",
    "copy_h2d_noncontig_dst": (
        "host->device write into a non-contiguous device dst; write into a contiguous buffer first, then h2d"
    ),
    "strided_v2v_cpu_fallback": (
        "strided v2v fell back to CPU; lower outer_count / use a fatter contiguous inner block "
        "so the device engine qualifies"
    ),
    "v2v_batch_to_per_entry": (
        "batched v2v rejected to per-entry; check the per-dst limit (kMaxV2VMultiCopies) and batch geometry"
    ),
    "cpu_fallback": (
        "op ran on CPU; prefer graph mode (torch.compile backend='rbln'), add a native rbln kernel, or fix "
        "an unsupported dtype -- host-only ops (argmax/sampling) are expected"
    ),
    "recompile": (
        "graph (re)compiled; stabilize shapes (pad/bucket) for warm-cache reuse, or use graph mode -- a cold "
        "first compile is expected, repeated recompiles in the steady loop are not"
    ),
}


def _collect_remedies(d: dict[str, Any]) -> list[dict[str, str]]:
    """Map the signals that actually fired in this region to their one-line fix.

    Returned in priority order (host bounce -> runtime cause -> dispatch) so the
    most impactful change is listed first. Pure lookup over the already-computed
    dump, so it costs nothing at runtime."""
    fixes: list[dict[str, str]] = []
    for name, vv in d["hidden_host_bounce"]["by_site"].items():
        if vv["count"] and name in _REMEDY:
            fixes.append({"signal": f"host_bounce/{name}", "fix": _REMEDY[name]})
    rr = d["runtime_residency"]
    if rr.get("available"):
        rfix = dict(_RUNTIME_REASONS)
        for n, vv in rr["by_reason"].items():
            if vv["count"]:
                fixes.append({"signal": f"runtime/v2v_slow:{n}", "fix": rfix[n]})
    disp = d["dispatch"]
    if disp["cpu_fallback"]:
        fixes.append({"signal": "dispatch/cpu_fallback", "fix": _REMEDY["cpu_fallback"]})
    if disp["recompile_miss"]:
        fixes.append({"signal": "dispatch/recompile", "fix": _REMEDY["recompile"]})
    return fixes


def _read_bounces() -> list[tuple[int, int]]:
    import torch_rbln._C as _C

    return list(_C._profiler_dump_bounces())


def _read_dispatch() -> tuple[int, int, int, int, int, int]:
    import torch_rbln._C as _C

    return tuple(_C._dispatch_shim_diag_dump())


def _read_fallback_by_op() -> dict[str, int]:
    """Per-op CPU-fallback counts from the dispatch shim (op_name -> count), or
    {} if the loaded torch_rbln._C predates the binding (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_dispatch_fallback_by_op", None)
    if fn is None:
        return {}
    return {str(op): int(c) for op, c in fn()}


# --- runtime (rebel-compiler) counters via ctypes from the loaded librbln -----
_rt_fns: Optional[dict] = None
_rt_resolved = False


def _runtime_fns():
    global _rt_fns, _rt_resolved
    if _rt_resolved:
        return _rt_fns
    _rt_resolved = True
    u64p = ctypes.POINTER(ctypes.c_uint64)
    for loader in (lambda: ctypes.CDLL(None), lambda: ctypes.CDLL("librbln.so")):
        try:
            lib = loader()
            hidden = lib.rbln_prof_get_v2v_hidden_d2h  # core hidden-d2h signal; required
        except (OSError, AttributeError):
            continue
        hidden.restype = None
        hidden.argtypes = [u64p, u64p, ctypes.c_uint32]
        fns = {"hidden": hidden}
        try:
            mem = lib.rbln_prof_get_memory  # device-memory gauge; optional (newer runtime)
            mem.restype = None
            mem.argtypes = [u64p, u64p]
            fns["memory"] = mem
        except (OSError, AttributeError):
            pass  # older runtime without the gauge -> memory reported pending, not zero
        _rt_fns = fns
        return _rt_fns
    _rt_fns = None
    return None


def _read_runtime() -> Optional[dict]:
    """Snapshot of runtime counters, or None if the loaded librbln lacks them.

    ``hidden_count`` (the hidden-d2h cause breakdown) is the core signal. The
    ``mem_cur``/``mem_peak`` device-memory gauge is present only on a runtime
    that exposes ``rbln_prof_get_memory``; on an older one it is omitted and
    reported as pending — never a false zero."""
    fns = _runtime_fns()
    if fns is None:
        return None
    n = len(_RUNTIME_REASONS)
    rc = (ctypes.c_uint64 * n)()
    rb = (ctypes.c_uint64 * n)()
    fns["hidden"](rc, rb, n)
    out: dict[str, Any] = {"hidden_count": [int(rc[i]) for i in range(n)]}
    if "memory" in fns:
        cur, peak = ctypes.c_uint64(), ctypes.c_uint64()
        fns["memory"](ctypes.byref(cur), ctypes.byref(peak))
        out["mem_cur"] = int(cur.value)
        out["mem_peak"] = int(peak.value)
    return out


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024 or unit == "GB":
            return f"{b:.2f} {unit}" if unit != "B" else f"{int(b)} B"
        b /= 1024
    return f"{b:.2f} GB"


def _fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:.2f} ms"


def _table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    """A torch.profiler-style bordered, column-aligned table (list of lines)."""
    cols = len(headers)
    widths = [len(headers[c]) for c in range(cols)]
    for r in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(r[c]))
    sep = "  ".join("-" * widths[c] for c in range(cols))

    def fmt_row(cells: list[str]) -> str:
        out = []
        for c in range(cols):
            out.append(cells[c].rjust(widths[c]) if aligns[c] == "r" else cells[c].ljust(widths[c]))
        return "  ".join(out)

    lines = [sep, fmt_row(headers), sep]
    lines += [fmt_row(r) for r in rows]
    lines.append(sep)
    return lines


class RBLNProfile:
    """A profiling region. Use via :func:`profile` as a context manager. All flow
    numbers are deltas over the region; memory is read as a level/high-water mark."""

    def __init__(self) -> None:
        self._b0 = self._d0 = self._rt0 = self._wall0 = self._f0 = None
        self._bounces = self._dispatch = self._rt = self._fallback_by_op = None
        self._wall_ns = 0

    def start(self) -> RBLNProfile:
        self._b0 = _read_bounces()
        self._d0 = _read_dispatch()
        self._f0 = _read_fallback_by_op()
        self._rt0 = _read_runtime()
        self._wall0 = time.perf_counter_ns()
        return self

    def stop(self) -> RBLNProfile:
        self._wall_ns = time.perf_counter_ns() - self._wall0
        b1, d1, rt1 = _read_bounces(), _read_dispatch(), _read_runtime()
        self._bounces = [(c1 - c0, by1 - by0) for (c0, by0), (c1, by1) in zip(self._b0, b1)]
        self._dispatch = tuple(x1 - x0 for x0, x1 in zip(self._d0, d1))
        f1 = _read_fallback_by_op()
        self._fallback_by_op = {op: f1[op] - self._f0.get(op, 0) for op in f1 if f1[op] - self._f0.get(op, 0) > 0}
        if self._rt0 is not None and rt1 is not None:
            hc = [a - b for a, b in zip(rt1["hidden_count"], self._rt0["hidden_count"])]
            self._rt = {"hidden_count": hc}
            if "mem_cur" in rt1:
                # memory is a process-level high-water gauge (a level, not a delta)
                self._rt["mem_cur"] = rt1["mem_cur"]
                self._rt["mem_peak"] = rt1["mem_peak"]
        else:
            self._rt = None
        return self

    def __enter__(self) -> RBLNProfile:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- readout -------------------------------------------------------------
    def dump(self) -> dict[str, Any]:
        assert self._bounces is not None, "region not stopped"
        per_site = {n: {"count": c, "bytes": by} for (n, _d), (c, by) in zip(_BOUNCE_SITES, self._bounces)}
        hb_count = sum(self._bounces[i][0] for i in _HOST_BOUNCE_SITES)
        hb_bytes = sum(self._bounces[i][1] for i in _HOST_BOUNCE_SITES)
        n_total, n_fallback, n_warm_hit, n_miss, _a, _b = self._dispatch

        out: dict[str, Any] = {
            "hidden_host_bounce": {"by_site": per_site, "total_count": hb_count, "total_bytes": hb_bytes},
            "dispatch": {
                "total": n_total,
                "cpu_fallback": n_fallback,
                "warm_hit": n_warm_hit,
                "recompile_miss": n_miss,
            },
            "wall_ns": self._wall_ns,
        }
        out["cpu_fallback_by_op"] = dict(sorted((self._fallback_by_op or {}).items(), key=lambda kv: -kv[1]))
        if self._rt is not None:
            out["runtime_residency"] = {
                "available": True,
                "total_count": sum(self._rt["hidden_count"]),
                "by_reason": {n: {"count": c} for (n, _f), c in zip(_RUNTIME_REASONS, self._rt["hidden_count"])},
            }
            pending = [
                "finer hidden-sync cause (dtype/align/chunks)",
                "hidden host-sync on the TVM graph-exec path (side-effect h2v)",
            ]
            if "mem_cur" in self._rt:
                out["device_memory"] = {"current_bytes": self._rt["mem_cur"], "peak_bytes": self._rt["mem_peak"]}
            else:
                pending.append("device memory gauge (older runtime)")
            out["pending_runtime_signals"] = pending
        else:
            out["runtime_residency"] = {"available": False}
            out["pending_runtime_signals"] = [
                "hidden_d2h / residency (STATE)",
                "device memory gauge",
            ]
        out["remedies"] = _collect_remedies(out)
        return out

    def verdict(self) -> dict[str, Any]:
        d = self.dump()
        hb, disp, rr = d["hidden_host_bounce"], d["dispatch"], d["runtime_residency"]
        reasons: list[str] = []
        status = "GREEN"
        if hb["total_count"] > 0:
            status = "RED"
            reasons.append(f"hidden host bounce (torch-side): {hb['total_count']}x, {_fmt_bytes(hb['total_bytes'])}")
        runtime_hidden = rr["total_count"] if rr["available"] else 0
        if runtime_hidden > 0:
            status = "RED"
            fix = dict(_RUNTIME_REASONS)
            for n, vv in rr["by_reason"].items():
                if vv["count"]:
                    reasons.append(f"runtime hidden d2h [{n}]: {vv['count']}x — {fix[n]}")
        if disp["recompile_miss"] > 0:
            status = "RED" if status == "RED" else "AMBER"
            reasons.append(f"recompile/cold-compile: {disp['recompile_miss']}x")
        if disp["cpu_fallback"] > 0:
            status = "RED" if status == "RED" else "AMBER"
            reasons.append(f"cpu_fallback: {disp['cpu_fallback']}x (ran on CPU)")
        if not reasons:
            reasons.append("no hidden host bounce, no CPU fallback, no recompile")
        return {
            "status": status,
            "hidden_host_bounces": hb["total_count"],
            "hidden_host_bounce_bytes": hb["total_bytes"],
            "runtime_hidden_d2h": runtime_hidden if rr["available"] else None,
            "runtime_hidden_by_reason": rr.get("by_reason") if rr["available"] else None,
            "cpu_fallbacks": disp["cpu_fallback"],
            "recompiles": disp["recompile_miss"],
            "reasons": reasons,
            "remedies": d.get("remedies", []),
        }

    def report(self) -> str:
        """torch.profiler-style report, verdict first."""
        d, v = self.dump(), self.verdict()
        mark = {"GREEN": "[ OK ]", "AMBER": "[WARN]", "RED": "[BAD ]"}[v["status"]]
        lines = [f"{mark}  RBLN PROFILE — VERDICT: {v['status']}   (wall {_fmt_ms(d['wall_ns'])})"]
        for r in v["reasons"]:
            lines.append(f"   • {r}")
        lines.append("")

        rows: list[list[str]] = []
        # torch-side host bounces (aten intent tier)
        for name, vv in d["hidden_host_bounce"]["by_site"].items():
            if vv["count"]:
                rows.append([f"host_bounce/{name}", str(vv["count"]), _fmt_bytes(vv["bytes"]), "torch aten intent"])
        # runtime hidden-d2h cause (v2v slow)
        rr = d["runtime_residency"]
        if rr["available"]:
            for n, vv in rr["by_reason"].items():
                if vv["count"]:
                    rows.append([f"runtime/v2v_slow:{n}", str(vv["count"]), "-", "cause of hidden d2h"])
        # dispatch
        disp = d["dispatch"]
        if disp["cpu_fallback"]:
            rows.append(["dispatch/cpu_fallback", str(disp["cpu_fallback"]), "-", "op ran on CPU"])
        if disp["recompile_miss"]:
            rows.append(["dispatch/recompile", str(disp["recompile_miss"]), "-", "graph (re)compiled"])
        if not rows:
            rows.append(["(clean)", "0", "-", "no hidden overhead"])

        lines += _table(["Signal", "Count", "Bytes", "Note"], rows, ["l", "r", "r", "l"])

        # WHERE: attribute the aggregate cpu_fallback count to the specific ops.
        fbo = d.get("cpu_fallback_by_op") or {}
        if fbo:
            items = list(fbo.items())
            shown = ", ".join(f"{op} {c}" for op, c in items[:8])
            extra = "" if len(items) <= 8 else f", +{len(items) - 8} more"
            lines.append(f"  cpu_fallback by op (top): {shown}{extra}")

        # RESOURCE gauge (not a hidden-overhead signal): device memory high-water.
        if "device_memory" in d:
            m = d["device_memory"]
            lines.append(
                f"  device memory: current {_fmt_bytes(m['current_bytes'])} | peak {_fmt_bytes(m['peak_bytes'])}"
            )
        if not rr["available"]:
            lines.append("  note: runtime signals not exposed by the loaded librbln (install a recent rebel-compiler)")
        fixes = d.get("remedies", [])
        if fixes:
            lines.append("")
            lines.append("  Suggested fixes (what to change), most impactful first:")
            for f in fixes:
                lines.append(f"   -> {f['signal']}: {f['fix']}")
        return "\n".join(lines)


def profile() -> RBLNProfile:
    """Return a profiling region usable as a context manager."""
    return RBLNProfile()
