"""``torch.rbln.explain`` — surface HIDDEN torch-rbln overhead, with CAUSE + FIX.

A normal PyTorch op (``a.copy_(b)``, ``torch.cat``, an indexing write, a forward
pass) can silently round-trip the host (NPU->CPU->NPU), fall back to a CPU
kernel, or recompile — for reasons the user never asked for and cannot see.
``explain()`` counts those hidden events, attributes the cause + a remedy, and
renders a torch.profiler-style report. It is a hidden-overhead explainer (think
``torch._dynamo.explain`` / JAX ``transfer_guard``), NOT a timing profiler.

Signals (every counter sits on an already-slow point — a host DMA or a fallback
branch — never the fast device path; reads are lazy, so wrapping a run with
explain() does not change its latency):

  host bounce (torch-side, aten intent) : non-direct copy_ / strided fallback / etc.
  runtime hidden-d2h cause (v2v slow)   : why a device v2v fell to host (src state)
  dispatch                              : cpu_fallback, recompile/miss, warm-hit

Plus one RESOURCE gauge (NOT a hidden-overhead signal — kept because it is
accurate and complete: every device allocation routes through BufferAllocator):

  device memory : current live + peak high-water

Scope: explain() surfaces ONLY hidden host overhead a user issued as a normal
op and cannot see. Generic dispatch/utilization counters (command-stream count,
device-idle, total host-traffic bytes) are a different profiler's job
(torch.profiler / nsys), are out of scope, and on the executed path live in the
TVM runtime where they read ~0 here — surfacing them would mislead. The one
hidden signal still missing — a side-effect host-sync on the TVM graph-exec path
— is a tracked follow-up (it, not the excluded counters, is what TVM-runtime
instrumentation is for).

What it does NOT know: explain observes a bounded region; it has no idea where
that region sits in your run (first step or thousandth), how many times you will
run, or whether each run does the same work. So it never labels a signal
"one-time" or "cold/warm" on its own — that would be a guess, and a wrong guess
("ignore this, it's one-time") is worse than none. To tell a one-time cost from
a recurring one, place two regions YOURSELF (an early one and a later one) and
compare them with ``a.diff(b)``; only you know which is which.

Usage::

    import torch

    with torch.rbln.explain() as p:
        model(x)
    print(p.report())  # torch.profiler-style table, verdict first
    p.dump()  # dict for CI gates
    p.verdict()  # {'status': 'GREEN'|'AMBER'|'RED', ...}

    with torch.rbln.explain(trace=True) as p:  # opt-in: WHERE each fallback/recompile originates
        model(x)
    print(p.report())  # adds an "at <file:line(func)>" line per offending op

    with torch.rbln.explain() as early:  # YOU place both points; only you know
        model(x)  # which is "early" and which is "later"
    for _ in range(5):
        model(x)
    with torch.rbln.explain() as later:
        model(x)
    print(early.diff(later).report())  # only what changed between YOUR two points
"""

from __future__ import annotations

import ctypes
import time
from typing import Any, Callable, Optional  # noqa: UP035


__all__ = ["explain", "explain_steady", "RBLNExplain", "RBLNDiff", "profile", "RBLNProfile"]


# Must match the BounceSite enum order in c10/rbln/RBLNProfiler.h.
_BOUNCE_SITES: tuple[tuple[str, str], ...] = (
    ("copy_d2d_host_bounce", "copy_: non-direct device->device round-tripped host"),
    ("copy_h2d_staging", "copy_: cpu src staged before h2v"),
    ("copy_h2d_noncontig_dst", "copy_: non-contiguous rbln dst pulled to host"),
    ("strided_v2v_cpu_fallback", "cat/index/copy_: strided v2v fell back to CPU"),
    ("v2v_batch_to_per_entry", "batched v2v rejected -> per-entry"),
)
_HOST_BOUNCE_SITES = (0, 1, 2, 3)

# cpu_fallback reason names; index matches the runtime histogram order (the WHY
# behind dispatch.cpu_fallback). See quick_fallback_check in DispatchShim.cpp.
_FALLBACK_REASON_NAMES = ("dtype-not-fp16", "nan/inf input", "all-scalar inputs")

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


# Terse inline "what to change" shown in the report's Fix column. The full prose
# lives in _REMEDY, surfaced on demand via dump()["remedies"] / RBLNExplain.help().
_FIX_SHORT: dict[str, str] = {
    "copy_d2d_host_bounce": "make contiguous, or v2v engine",
    "copy_h2d_staging": "keep src on-device / stage once",
    "copy_h2d_noncontig_dst": "contiguous staging, then h2d",
    "strided_v2v_cpu_fallback": "lower outer_count / fatter inner",
    "v2v_batch_to_per_entry": "check kMaxV2VMultiCopies / batch geom",
    "cpu_fallback": "graph mode / native kernel / dtype",
    "recompile": "stabilize shapes, or graph mode",
    "v2v_slow": "establish device residency first",
}

_RT_UNAVAIL = "  note: runtime signals not exposed by the loaded librbln (install a recent rebel-compiler)"


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


def _read_recompile_by_op() -> dict[str, int]:
    """Per-op warm-cache miss (recompile) counts (op_name -> count), or {} if the
    loaded torch_rbln._C predates the binding (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_dispatch_recompile_by_op", None)
    if fn is None:
        return {}
    return {str(op): int(c) for op, c in fn()}


def _read_fallback_reasons() -> list[int]:
    """cpu_fallback reason counts [dtype-not-fp16, nan/inf input, all-scalar], or
    [] if the loaded torch_rbln._C predates the binding (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_dispatch_fallback_reasons", None)
    if fn is None:
        return []
    return [int(c) for c in fn()]


# --- (A) WHERE: opt-in Python call-site capture (off by default) --------------
def _read_trace_by_op() -> dict[str, str]:
    """Per-op captured call-site (op_name -> 'file:line(func) <- ...'), or {} if
    trace was never enabled / the binding is absent (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_explain_trace_by_op", None)
    if fn is None:
        return {}
    return {str(op): str(site) for op, site in fn()}


def _set_trace(on: bool) -> None:
    """Flip the C++ capture gate. No-op on a _C that predates the binding (so a
    trace=True request on an old build degrades to no call-sites, not an error)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_explain_set_trace", None)
    if fn is not None:
        fn(on)


def _reset_trace() -> None:
    import torch_rbln._C as _C

    fn = getattr(_C, "_explain_trace_by_op_reset", None)
    if fn is not None:
        fn()


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
        try:
            # REAL device->host transfers (count, bytes) emitted by the v-mem manager
            # at every real d2h primitive; the authoritative real-vs-host-served signal
            # (a host-served copy never reaches it). Optional (newer runtime).
            hs = lib.rbln_prof_get_host_sync_d2h
            hs.restype = None
            hs.argtypes = [u64p, u64p]
            fns["host_sync"] = hs
        except (OSError, AttributeError):
            pass
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
    if "host_sync" in fns:
        hc, hb = ctypes.c_uint64(), ctypes.c_uint64()
        fns["host_sync"](ctypes.byref(hc), ctypes.byref(hb))
        out["host_sync_count"] = int(hc.value)
        out["host_sync_bytes"] = int(hb.value)
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


class RBLNExplain:
    """A hidden-overhead explain region. Use via :func:`explain` as a context manager. All flow
    numbers are deltas over the region; memory is read as a level/high-water mark."""

    def __init__(self, trace: bool = False) -> None:
        self._b0 = self._d0 = self._rt0 = self._wall0 = self._f0 = self._r0 = self._fr0 = None
        self._bounces = self._dispatch = self._rt = self._fallback_by_op = self._recompile_by_op = None
        self._fallback_reasons = None
        self._trace = trace
        self._trace_by_op: Optional[dict] = None
        self._wall_ns = 0

    def start(self) -> RBLNExplain:
        if self._trace:
            _set_trace(True)
            _reset_trace()  # region-local captures (the C++ map dedups per op)
        self._b0 = _read_bounces()
        self._d0 = _read_dispatch()
        self._f0 = _read_fallback_by_op()
        self._r0 = _read_recompile_by_op()
        self._fr0 = _read_fallback_reasons()
        self._rt0 = _read_runtime()
        self._wall0 = time.perf_counter_ns()
        return self

    def stop(self) -> RBLNExplain:
        self._wall_ns = time.perf_counter_ns() - self._wall0
        b1, d1, rt1 = _read_bounces(), _read_dispatch(), _read_runtime()
        self._bounces = [(c1 - c0, by1 - by0) for (c0, by0), (c1, by1) in zip(self._b0, b1)]
        self._dispatch = tuple(x1 - x0 for x0, x1 in zip(self._d0, d1))
        f1 = _read_fallback_by_op()
        self._fallback_by_op = {op: f1[op] - self._f0.get(op, 0) for op in f1 if f1[op] - self._f0.get(op, 0) > 0}
        r1 = _read_recompile_by_op()
        self._recompile_by_op = {op: r1[op] - self._r0.get(op, 0) for op in r1 if r1[op] - self._r0.get(op, 0) > 0}
        fr1 = _read_fallback_reasons()
        self._fallback_reasons = (
            [b - a for a, b in zip(self._fr0, fr1)] if self._fr0 and fr1 and len(fr1) == len(self._fr0) else []
        )
        if self._rt0 is not None and rt1 is not None:
            hc = [a - b for a, b in zip(rt1["hidden_count"], self._rt0["hidden_count"])]
            self._rt = {"hidden_count": hc}
            if "mem_cur" in rt1:
                # memory is a process-level high-water gauge (a level, not a delta)
                self._rt["mem_cur"] = rt1["mem_cur"]
                self._rt["mem_peak"] = rt1["mem_peak"]
            if "host_sync_count" in rt1 and "host_sync_count" in self._rt0:
                self._rt["host_sync_count"] = rt1["host_sync_count"] - self._rt0["host_sync_count"]
                self._rt["host_sync_bytes"] = rt1["host_sync_bytes"] - self._rt0["host_sync_bytes"]
        else:
            self._rt = None
        if self._trace:
            self._trace_by_op = _read_trace_by_op()
            _set_trace(False)
        else:
            self._trace_by_op = {}
        return self

    def __enter__(self) -> RBLNExplain:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- readout -------------------------------------------------------------
    def dump(self) -> dict[str, Any]:
        assert self._bounces is not None, "region not stopped"
        per_site = {n: {"count": c, "bytes": by} for (n, _d), (c, by) in zip(_BOUNCE_SITES, self._bounces)}
        hb_count = sum(self._bounces[i][0] for i in _HOST_BOUNCE_SITES)
        hb_bytes = sum(self._bounces[i][1] for i in _HOST_BOUNCE_SITES)
        # tuple is (n_total, n_fallback, n_warm_hit, n_miss, ns_warm_hit, ns_miss[, ns_fallback]);
        # the 7th (cpu_fallback wall ns) is absent on a _C that predates it -> degrade to 0.
        d = self._dispatch
        n_total, n_fallback, n_warm_hit, n_miss = d[0], d[1], d[2], d[3]
        ns_fallback = d[6] if len(d) > 6 else 0

        out: dict[str, Any] = {
            "hidden_host_bounce": {"by_site": per_site, "total_count": hb_count, "total_bytes": hb_bytes},
            "dispatch": {
                "total": n_total,
                "cpu_fallback": n_fallback,
                "cpu_fallback_ns": ns_fallback,
                "warm_hit": n_warm_hit,
                "recompile_miss": n_miss,
            },
            "wall_ns": self._wall_ns,
        }
        out["cpu_fallback_by_op"] = dict(sorted((self._fallback_by_op or {}).items(), key=lambda kv: -kv[1]))
        out["recompile_by_op"] = dict(sorted((self._recompile_by_op or {}).items(), key=lambda kv: -kv[1]))
        out["cpu_fallback_reasons"] = {
            n: c for n, c in zip(_FALLBACK_REASON_NAMES, self._fallback_reasons or []) if c > 0
        }
        out["trace_by_op"] = dict(self._trace_by_op or {})  # (A) WHERE; {} unless trace=True
        if self._rt is not None:
            out["runtime_residency"] = {
                "available": True,
                "total_count": sum(self._rt["hidden_count"]),
                "by_reason": {n: {"count": c} for (n, _f), c in zip(_RUNTIME_REASONS, self._rt["hidden_count"])},
            }
            if "host_sync_count" in self._rt:
                # authoritative REAL device->host this region (manager-emitted). A host
                # bounce with 0 real d2h was served on host (no device crossing).
                out["runtime_residency"]["real_host_sync_d2h"] = {
                    "count": self._rt["host_sync_count"],
                    "bytes": self._rt["host_sync_bytes"],
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
        """Verdict-first report. The terse fix is inline (Fix column); the full
        remedy prose is in dump()['remedies'] / help(signal), not dumped here."""
        d, v = self.dump(), self.verdict()
        mark = {"GREEN": "[ OK ]", "AMBER": "[WARN]", "RED": "[BAD ]"}[v["status"]]
        head = f"{mark}  RBLN EXPLAIN — {v['status']}   (wall {_fmt_ms(d['wall_ns'])}"
        if "device_memory" in d:
            head += f" · mem {_fmt_bytes(d['device_memory']['peak_bytes'])} peak"
        head += ")"
        lines = [head]

        rr = d["runtime_residency"]
        rows: list[list[str]] = []
        for name, vv in d["hidden_host_bounce"]["by_site"].items():
            if vv["count"]:
                rows.append(
                    [f"host_bounce/{name}", str(vv["count"]), _fmt_bytes(vv["bytes"]), _FIX_SHORT.get(name, "")]
                )
        if rr["available"]:
            for n, vv in rr["by_reason"].items():
                if vv["count"]:
                    rows.append([f"runtime/v2v_slow:{n}", str(vv["count"]), "-", _FIX_SHORT["v2v_slow"]])
        disp = d["dispatch"]
        if disp["cpu_fallback"]:
            rows.append(["dispatch/cpu_fallback", str(disp["cpu_fallback"]), "-", _FIX_SHORT["cpu_fallback"]])
        if disp["recompile_miss"]:
            rows.append(["dispatch/recompile", str(disp["recompile_miss"]), "-", _FIX_SHORT["recompile"]])

        if not rows:
            lines.append("  (clean) no hidden overhead")
            if not rr["available"]:
                lines.append(_RT_UNAVAIL)
            return "\n".join(lines)

        lines.append("")
        lines += _table(["Signal", "Count", "Bytes", "Fix"], rows, ["l", "r", "r", "l"])

        # Qualify the torch-side host_bounce (a copy-PATH count, blind to residency)
        # with the v-mem manager's authoritative REAL device->host count: 0 real means
        # the bounce was served on the host (no device crossing) -> low cost, not a leak.
        rhs = rr.get("real_host_sync_d2h") if rr.get("available") else None
        if rhs is not None and (rhs["count"] or d["hidden_host_bounce"]["total_count"]):
            if rhs["count"]:
                lines.append(f"    real device->host (runtime): {rhs['count']} copies, {_fmt_bytes(rhs['bytes'])}")
            else:
                lines.append("    real device->host (runtime): 0 — host_bounce above was served on host")

        # attribution sub-lines (which ops, and WHY) — compact, under the table.
        def _top(m: dict) -> str:
            items = list(m.items())
            shown = ", ".join(f"{op} {c}" for op, c in items[:8])
            return shown + ("" if len(items) <= 8 else f", +{len(items) - 8} more")

        tbo = d.get("trace_by_op") or {}  # (A) WHERE: op -> call-site, only when trace=True

        def _where(by_op: dict) -> None:
            for op in list(by_op)[:3]:
                if op in tbo:
                    lines.append(f"      at {op}: {tbo[op]}")

        fbo = d.get("cpu_fallback_by_op") or {}
        if fbo:
            fb_ns = d["dispatch"].get("cpu_fallback_ns", 0)
            # the COST -- distinguishes "many cheap fallbacks (path overhead)" from
            # "few expensive ones (hidden transfer)"; only shown when the loaded _C exposes it.
            cost = f"   (Σ {fb_ns / 1000:.0f} µs wall)" if fb_ns else ""
            lines.append(f"    cpu_fallback: {_top(fbo)}{cost}")
        fr = d.get("cpu_fallback_reasons") or {}
        if fr:
            lines.append("      why: " + ", ".join(f"{n} {c}" for n, c in fr.items()))
        _where(fbo)
        rbo = d.get("recompile_by_op") or {}
        if rbo:
            lines.append(f"    recompile: {_top(rbo)}")
        _where(rbo)
        if (fbo or rbo) and not tbo:
            lines.append("      where? -> rerun with explain(trace=True)")
        if not rr["available"]:
            lines.append(_RT_UNAVAIL)
        lines.append("  (full fix detail -> p.help(signal) or dump()['remedies'])")
        return "\n".join(lines)

    def help(self, signal: Optional[str] = None) -> str:
        """Full remedy prose. No arg: the fix for every fired signal. With a
        signal (bare cause or report label like 'dispatch/cpu_fallback'): just
        that one."""
        fixes = self.dump().get("remedies", [])
        if signal is None:
            return "\n".join(f"{f['signal']}: {f['fix']}" for f in fixes) or "no hidden overhead"
        key = signal.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if key in _REMEDY:
            return _REMEDY[key]
        rfix = dict(_RUNTIME_REASONS)
        if key in rfix:
            return rfix[key]
        return next((f["fix"] for f in fixes if signal in f["signal"]), f"no remedy for '{signal}'")

    def diff(self, other: RBLNExplain) -> RBLNDiff:
        """Compare this region with another one YOU placed (self -> other), e.g.
        an early call vs a later steady call. explain cannot tell a one-time cost
        from a recurring one on its own (it does not know your run structure);
        placing the two regions is how YOU supply that. See :class:`RBLNDiff`."""
        assert self._bounces is not None, "this region is not stopped"
        assert other._bounces is not None, "the other region is not stopped"
        return RBLNDiff(self, other)


class RBLNDiff:
    """The signal-by-signal change between two regions YOU placed (``a`` -> ``b``).

    explain knows only what happened inside each region; it does NOT know your
    run structure, so on its own it cannot tell a one-time cost from a recurring
    one. By choosing where to put the two regions (e.g. an early call and a later
    steady one) YOU supply that structure. This reports ONLY what changed between
    your two points: a signal gone (0) in ``b`` did not recur across them; one
    that persists is overhead that recurs between them. It makes no claim about
    anything outside the two regions you captured."""

    def __init__(self, a: RBLNExplain, b: RBLNExplain) -> None:
        self._a = a.dump()
        self._b = b.dump()

    def _rows(self) -> list[tuple[str, int, int]]:
        da, db = self._a, self._b
        rows = [("host_bounce", da["hidden_host_bounce"]["total_count"], db["hidden_host_bounce"]["total_count"])]
        ra, rb = da["runtime_residency"], db["runtime_residency"]
        if ra.get("available") and rb.get("available"):
            rows.append(("runtime/v2v_slow", ra["total_count"], rb["total_count"]))
        rows.append(("cpu_fallback", da["dispatch"]["cpu_fallback"], db["dispatch"]["cpu_fallback"]))
        rows.append(("recompile", da["dispatch"]["recompile_miss"], db["dispatch"]["recompile_miss"]))
        return rows

    def dump(self) -> dict[str, Any]:
        """{'signals': {name: {'a','b'}}, 'persists'|'gone'|'appeared': [names],
        'persists_by_op': [{signal, op, count, at}], 'device_memory': {...}}.
        ``persists`` (present in ``b``) is the actionable, recurring set."""
        out: dict[str, Any] = {"signals": {}, "persists": [], "gone": [], "appeared": [], "persists_by_op": []}
        for name, av, bv in self._rows():
            out["signals"][name] = {"a": av, "b": bv}
            if bv > 0:
                out["persists"].append(name)
            elif av > 0:
                out["gone"].append(name)
            # av == 0 and bv == 0 -> neither; nothing to record
        for name, av, bv in self._rows():
            if av == 0 and bv > 0:
                out["appeared"].append(name)
        tbo = self._b.get("trace_by_op") or {}
        for sig_key, by_key in (("cpu_fallback", "cpu_fallback_by_op"), ("recompile", "recompile_by_op")):
            for op, c in (self._b.get(by_key) or {}).items():
                if c > 0:
                    out["persists_by_op"].append({"signal": sig_key, "op": op, "count": c, "at": tbo.get(op)})
        if "device_memory" in self._a and "device_memory" in self._b:
            out["device_memory"] = {
                "a_peak_bytes": self._a["device_memory"]["peak_bytes"],
                "b_peak_bytes": self._b["device_memory"]["peak_bytes"],
            }
        return out

    def report(self) -> str:
        d = self.dump()
        lines = [
            "RBLN EXPLAIN DIFF   (A -> B; you placed both points)",
            "  explain does not know your run structure; this compares ONLY your two regions.",
            "",
        ]
        rows = []
        for name, vv in d["signals"].items():
            av, bv = vv["a"], vv["b"]
            mark = "*** PERSISTS" if bv > 0 else ("gone in B" if av > 0 else "")
            rows.append([name, str(av), str(bv), mark])
        lines += _table(["Signal", "A", "B", ""], rows, ["l", "r", "r", "l"])
        pbo = d.get("persists_by_op") or []
        if pbo:
            lines.append("")
            lines.append("  PERSISTS across your two points -> recurring overhead, act on these:")
            for e in pbo[:8]:
                at = f"  @ {e['at']}" if e.get("at") else ""
                lines.append(f"    {e['signal']}: {e['op']} {e['count']}{at}")
        elif not d["persists"]:
            lines.append("")
            lines.append("  nothing persisted into B (no recurring overhead across your two points)")
        return "\n".join(lines)


def explain(trace: bool = False) -> RBLNExplain:
    """Return a hidden-overhead explain region, usable as a context manager.

    With ``trace=True`` (opt-in; OFF by default, so a plain region adds nothing to
    any path), the FIRST time each op falls back / recompiles its Python call-site
    is captured (deduped per op) and shown as an ``at <file:line(func)>`` line in
    the report — telling you WHERE in your model the hidden overhead originates."""
    return RBLNExplain(trace=trace)


# Backward-compatible aliases. The tool was originally exposed as ``profile``;
# renamed to ``explain`` since it is a hidden-overhead explainer (cause + fix),
# not a timing profiler. Kept so existing ``torch.rbln.profile()`` callers work.
RBLNProfile = RBLNExplain
profile = explain


def explain_steady(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    return_cold: bool = False,
    as_diff: bool = False,
    trace: bool = False,
) -> RBLNExplain | tuple[RBLNExplain, RBLNExplain] | RBLNDiff:
    """Convenience: capture two regions around ``fn`` — the FIRST call it makes
    and a later call after ``warmup`` more — so you can compare them.

    This does NOT know your lifecycle. It only AUTOMATES placing two regions; its
    "cold"/"warm" labels mean literally "the first call I made" and "a later call
    I made", nothing more. Reading the result as one-time vs recurring rests on
    two assumptions YOU must ensure — neither of which explain can verify:

      1. ``fn`` was not already run/compiled before this call. Otherwise the real
         one-time cost happened before the "cold" sample, so it is not captured.
      2. every ``fn()`` does the SAME work. Otherwise the "warm" sample's
         recompiles are just you feeding different shapes, not recurring overhead.

    When those hold, ``cold.diff(warm)`` shows what changed: signals gone in warm
    did not recur across the two calls; signals that persist are the recurring
    overhead. Pure-Python orchestration; adds nothing to the hot path.

    Returns the warm :class:`RBLNExplain`; ``return_cold=True`` -> ``(cold, warm)``;
    ``as_diff=True`` -> ``cold.diff(warm)`` (the recommended, most honest read,
    since it states only what changed between the two calls it made).

    Example::

        d = torch.rbln.explain_steady(lambda: model(x), warmup=2, as_diff=True)
        print(d.report())  # what PERSISTS is the overhead that recurs across the two calls
    """
    cold = RBLNExplain(trace=trace).start()
    try:
        fn()
    finally:
        cold.stop()
    for _ in range(max(0, warmup)):
        fn()
    warm = RBLNExplain(trace=trace).start()
    try:
        fn()
    finally:
        warm.stop()
    if as_diff:
        return cold.diff(warm)
    return (cold, warm) if return_cold else warm
