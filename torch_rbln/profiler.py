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
"""

from __future__ import annotations

import ctypes
import time
from typing import Any, Callable, Optional  # noqa: UP035


__all__ = ["explain", "explain_steady", "RBLNExplain", "profile", "RBLNProfile"]


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
            lines.append(f"    cpu_fallback: {_top(fbo)}")
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
    fn: Callable[[], Any], *, warmup: int = 2, return_cold: bool = False, trace: bool = False
) -> RBLNExplain | tuple[RBLNExplain, RBLNExplain]:
    """Profile ``fn()`` at STEADY STATE, isolating per-call overhead from one-time
    cold cost (compile / first-touch / cache fill).

    Many signals only matter if they recur every step: a single cold run conflates
    one-time compilation with per-call overhead (e.g. a decode whose first call is
    8 s of compile but whose steady call is 12 ms). This runs ``fn`` once as the
    COLD sample, ``warmup`` more times to reach steady state, then profiles one
    more call as the WARM (steady) sample.

    Pure-Python orchestration — it only snapshots the lazy counters at region
    boundaries, so it adds nothing to the hot path.

    Returns the warm :class:`RBLNExplain`; with ``return_cold=True`` returns
    ``(cold, warm)`` so you can see what was one-time vs steady.

    Example::

        warm = torch.rbln.explain_steady(lambda: model(x), warmup=2)
        print(warm.report())  # the steady-state overhead that recurs
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
    return (cold, warm) if return_cold else warm
