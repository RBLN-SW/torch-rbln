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
    print(p.report())  # torch.profiler-style table, [clean]/[overhead] marker + facts
    p.dump()  # dict for CI gates
    p.verdict()  # {'clean': bool, 'reasons': [...], ...}  (a fact, not a severity grade)

    with torch.rbln.explain(with_stack=True) as p:  # opt-in: WHERE each fallback/recompile originates
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

import threading
import time
from typing import Any, Callable, Optional  # noqa: UP035
from typing_extensions import Self


__all__ = ["explain", "explain_steady", "RBLNExplain", "RBLNDiff", "profile", "RBLNProfile"]


# explain() drives PROCESS-GLOBAL instrumentation: the (B) rebel-runtime timer gate
# (RBLNFunctions.cpp) and the trace gate/map (DispatchShim.cpp) are single globals, and
# the counters are read as deltas. A region owns that global state for its duration, so
# regions MUST NOT overlap. Rather than build per-region views over shared singletons
# (refcounts/snapshots), we keep one honest contract -- a single, non-overlapping,
# single-thread region -- enforced by a guard that raises on overlap.
_region_lock = threading.Lock()
_active = False  # a region is currently open (regions must not overlap)


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
# lives in _REMEDY, surfaced on demand via dump()["notes"] / RBLNExplain.help().
_FIX_SHORT: dict[str, str] = {
    "copy_d2d_host_bounce": "make contiguous, or v2v engine",
    "copy_h2d_staging": "keep src on-device / stage once",
    "copy_h2d_noncontig_dst": "contiguous staging, then h2d",
    "strided_v2v_cpu_fallback": "lower outer_count / fatter inner",
    "v2v_batch_to_per_entry": "check kMaxV2VMultiCopies / batch geom",
    "cpu_fallback": "graph mode, or a supported dtype",
    "recompile": "stabilize shapes, or graph mode",
    "v2v_slow": "establish device residency first",
}

_RT_UNAVAIL = "  note: runtime signals not exposed by the loaded librbln (install a recent rebel-compiler)"


def _collect_notes(d: dict[str, Any]) -> list[dict[str, str]]:
    """Map the signals that actually fired in this region to their one-line note.

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
                # by_reason keys are exactly _RUNTIME_REASONS (dump() fails fast on axis
                # drift), so rfix[n] always resolves.
                fixes.append({"signal": f"runtime/v2v_slow:{n}", "fix": rfix[n]})
        # Reject axis (so help()/dump()['notes'] resolve the v2v_reject:* signals the
        # report shows): user-actionable reasons carry their fix; the internal bucket is a
        # notification, not a to-do.
        rj = rr.get("reject") or {}
        for lbl in rj.get("user_actionable", {}):
            fixes.append({"signal": f"runtime/v2v_reject:{lbl}", "fix": _REJECT_FIX.get(lbl, "")})
        if rj.get("internal_fallback", {}).get("count"):
            fixes.append(
                {
                    "signal": "runtime/v2v_reject:internal_fallback",
                    "fix": "runtime-internal fallback; not user-fixable (a notification, not a to-do)",
                }
            )
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


# --- (A) fast-path handler coverage: which fallback ops are un-accelerated -----
def _read_fast_path_registered(op_name: str) -> Optional[bool]:
    """True/False if the loaded _C can say whether ``op_name`` has a CPU
    fast-path handler; None if the binding is absent (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_cpu_fast_path_registered", None)
    if fn is None:
        return None
    try:
        return bool(fn(op_name))
    except Exception:
        return None


# --- (E) host CPU oversubscription: an environment amplifier of host overhead --
def _host_thread_info() -> dict[str, Any]:
    """Cores this process may run on vs the host-thread parallelism. When threads
    far exceed cores the worker's tiny serial host ops get preempted by the idle
    OMP/numba pool, inflating per-op host latency several-fold -- invisible to any
    per-op counter. A resource fact (like device memory), never a per-op verdict."""
    import os

    info: dict[str, Any] = {}
    try:
        info["cores"] = len(os.sched_getaffinity(0))  # cores actually allowed (post-pin)
    except (AttributeError, OSError):
        info["cores"] = os.cpu_count() or 0
    try:
        import torch

        info["torch_threads"] = int(torch.get_num_threads())
    except Exception:
        info["torch_threads"] = 0
    try:
        info["omp_threads"] = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    except ValueError:
        info["omp_threads"] = 0
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("Threads:"):
                    info["proc_threads"] = int(ln.split()[1])
                    break
    except Exception:
        pass
    intended = max(info.get("omp_threads", 0), info["torch_threads"], info.get("proc_threads", 0))
    info["intended_threads"] = intended
    info["oversubscribed"] = info["cores"] > 0 and intended > info["cores"]
    return info


# --- (B) rebel-runtime (librbln) boundary time, gated on the explain region ----
# Order MUST match the C++ RtIdx enum in c10/rbln/RBLNFunctions.cpp.
_RT_PRIMS = ("v2v", "v2v_multi", "borrow", "acquire", "return", "v2h", "h2v")


def _rt_timing_enable(on: bool) -> None:
    import torch_rbln._C as _C

    fn = getattr(_C, "_rt_timing_enable", None)
    if fn is not None:
        fn(on)


def _rt_timing_reset() -> None:
    import torch_rbln._C as _C

    fn = getattr(_C, "_rt_timing_reset", None)
    if fn is not None:
        fn()


def _read_rt_timing() -> Optional[list[tuple[int, int]]]:
    """Per-primitive (ns, calls) spent inside librbln boundary calls, or None if
    the loaded _C predates the binding (graceful degrade)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_rt_timing_get", None)
    if fn is None:
        return None
    try:
        return [(int(ns), int(cnt)) for ns, cnt in fn()]
    except Exception:
        return None


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
    with_stack=True request on an old build degrades to no call-sites, not an error)."""
    import torch_rbln._C as _C

    fn = getattr(_C, "_explain_set_trace", None)
    if fn is not None:
        fn(on)


def _reset_trace() -> None:
    import torch_rbln._C as _C

    fn = getattr(_C, "_explain_trace_by_op_reset", None)
    if fn is not None:
        fn()


# --- runtime (rebel-compiler) counters via the linked _C extension (pybind) -----
# Read through torch_rbln._C (which links librbln) rather than dlsym'ing librbln by
# ctypes: one typed channel, consistent with every other runtime signal here. Each
# getter returns Python-native values (lists of (count, bytes) / tuples), no marshal.
# Per-reason axes are POSITIONAL; their meaning is mapped on THIS (torch) side — no
# internal classification name crosses over from the runtime.

# Reject-axis presentation: mirrors the runtime V2VRejectReason axis POSITIONALLY
# (must match its order). Only user-actionable reasons are named; every "internal"
# reason is collapsed into one bucket and not named, because the internal reject reason
# is a non-deterministic runtime detail (the same op can hit a different reason by
# residency/layout state) that the user cannot act on -- naming it would imply false
# actionability. The collapsed bucket is a magnitude-carrying notification, not a to-do.
# index 0 = none/unused.
_REJECT_AUDIENCE = ("none", "user", "internal", "internal", "user", "internal", "internal", "internal", "internal")
_REJECT_LABEL = {1: "src_not_on_device", 4: "dtype_mismatch"}
_REJECT_FIX = {
    "src_not_on_device": "establish device residency before the copy",
    "dtype_mismatch": "align the src/dst dtype",
}


def _rt_prof(name: str):
    """Resolve a runtime-counter binding on _C, or None if the loaded _C/librbln
    predates it (graceful degradation -> signal reports pending, never a false zero)."""
    try:
        import torch_rbln._C as _C
    except Exception:
        return None
    return getattr(_C, name, None)


def _read_runtime() -> Optional[dict]:
    """Snapshot of runtime (rebel-compiler) counters via _C, or None if unavailable.

    ``hidden_count`` (the hidden-d2h cause breakdown) is the core signal. ``reject_*``
    (WHY a fast v2v plan was rejected), the ``host_sync_*`` real-transfer counters, and
    the ``mem_*`` device-memory gauge are each present only on a runtime that exposes
    them; on an older one they are omitted (reported pending — never a false zero)."""
    hidden = _rt_prof("_rt_prof_hidden")
    if hidden is None:
        return None
    h = hidden()  # [(count, bytes), ...] positional by src-state cause
    out: dict[str, Any] = {"hidden_count": [c for c, _b in h], "hidden_bytes": [b for _c, b in h]}
    rej = _rt_prof("_rt_prof_reject")
    if rej is not None:
        r = rej()  # [(count, bytes), ...] positional by V2VRejectReason
        out["reject_count"] = [c for c, _b in r]
        out["reject_bytes"] = [b for _c, b in r]
    hs = _rt_prof("_rt_prof_host_sync")
    if hs is not None:
        (dc, db), (hc, hb) = hs()  # [0] = d2h, [1] = h2d
        out["host_sync_count"], out["host_sync_bytes"] = dc, db
        out["host_sync_h2d_count"], out["host_sync_h2d_bytes"] = hc, hb
    mem = _rt_prof("_rt_prof_memory")
    if mem is not None:
        cur, peak = mem()
        out["mem_cur"], out["mem_peak"] = cur, peak
    return out


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024 or unit == "GB":
            return f"{b:.2f} {unit}" if unit != "B" else f"{int(b)} B"
        b /= 1024
    return f"{b:.2f} GB"


def _fmt_time(ns: float) -> str:
    # Mirror torch.profiler's _format_time (adaptive us/ms/s) so explain reads like a
    # torch profiler table. ASCII "us" (not the Unicode micro sign) for log/Slack safety.
    us = ns / 1e3
    if us >= 1e6:
        return f"{us / 1e6:.3f}s"
    if us >= 1e3:
        return f"{us / 1e3:.3f}ms"
    return f"{us:.3f}us"


# Note-column prefixes: split a SUGGESTED action (try:) from a fact/notification (fyi:).
# "try:" (not "fix:") is deliberate -- the advice is a starting point, not a guaranteed fix;
# it may not fit a given workload. torch tables have no advice column, so this is
# explain-specific -- kept ASCII.
def _fix(note: str) -> str:
    return f"try: {note}" if note else ""


def _fyi(note: str) -> str:
    return f"fyi: {note}" if note else ""


def _site_label(name: str) -> str:
    # Display-only trim: the 'host_bounce/' namespace already carries "copy"/"bounce",
    # so the raw site keys read redundantly and are the widest table cell. Shorten for
    # the report; the underlying data/dump keys (by_site, _REMEDY) are unchanged.
    return {
        "copy_d2d_host_bounce": "d2d_copy",
        "copy_h2d_staging": "h2d_staging",
        "copy_h2d_noncontig_dst": "h2d_noncontig_dst",
    }.get(name, name)


def _thousands(n: int) -> str:
    return f"{n:,}"


def _short_op(op: str) -> str:
    # Drop the 'aten::' namespace for compact candidate lists (the per-op lines keep it).
    return op.removeprefix("aten::")


def _short_path(callsite: str) -> str:
    """Tail-truncate long file paths in an ``at`` call-site string so the informative
    end (``.../parent/file.py:line(fn)``) survives without a full absolute path. Applied
    only at report time; ``dump()['trace_by_op']`` keeps the raw string."""
    import re

    def _trim(m) -> str:
        path = m.group(0)
        parts = path.split("/")
        return path if len(parts) <= 2 else ".../" + "/".join(parts[-2:])

    # Match path-like runs ending in a component (kept greedy on the dir portion).
    return re.sub(r"(?:[\w.\-]+/){2,}[\w.\-]+", _trim, callsite)


def _fmt_dur(ns: float) -> str:
    # Compact 1-decimal form for the per-primitive breakdown ('7.3ms'); headline totals
    # use the adaptive _fmt_time. Sub-ms primitives fall back to whole microseconds.
    us = ns / 1e3
    return f"{us / 1e3:.1f}ms" if us >= 1e3 else f"{us:.0f}us"


def _op_block(by_op: dict, tbo: dict, limit: int = 8) -> list[str]:
    """Per-op detail lines for a dispatch signal -- ``op  count  at call-site`` column-
    aligned within the block (alignment is per-block, not across blocks). The call-site
    is shown only when captured (``with_stack=True``)."""
    items = list(by_op.items())[:limit]
    if not items:
        return []
    ow = max(len(op) for op, _ in items)
    cw = max(len(_thousands(c)) for _, c in items)
    out = []
    for op, c in items:
        at = f"  at {_short_path(tbo[op])}" if op in tbo else ""
        out.append(f"    {op.ljust(ow)}  {_thousands(c).rjust(cw)}{at}")
    extra = len(by_op) - len(items)
    if extra > 0:
        out.append(f"    +{extra} more")
    return out


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
        return "  ".join(out).rstrip()  # trim padding on the trailing (left-aligned) column

    lines = [sep, fmt_row(headers), sep]
    lines += [fmt_row(r) for r in rows]
    lines.append(sep)
    return lines


class RBLNExplain:
    """A hidden-overhead explain region. Use via :func:`explain` as a context manager.

    All flow numbers are deltas over the region window, but the underlying counters are
    PROCESS-GLOBAL: the delta sums activity from ALL threads that ran during the region,
    not just the calling thread. Concurrent device work (e.g. a background transfer or
    worker thread) is therefore included in the delta — attribute accordingly when
    profiling a multi-threaded run (vLLM, DataLoader workers, ...). Memory is read as a
    process-wide level/high-water mark, not a per-region delta."""

    def __init__(self, with_stack: bool = False, *, trace: Optional[bool] = None) -> None:
        if trace is not None:  # deprecated back-compat alias for with_stack
            with_stack = trace
        self._b0 = self._d0 = self._rt0 = self._wall0 = self._f0 = self._r0 = self._fr0 = None
        self._bounces = self._dispatch = self._rt = self._fallback_by_op = self._recompile_by_op = None
        self._fallback_reasons = None
        self._trace = with_stack
        self._trace_by_op: Optional[dict] = None
        self._rt_timing: Optional[list[tuple[int, int]]] = None  # (B) per-primitive librbln (ns, calls)
        self._holds = False  # this region owns the global instrumentation (release exactly once)
        self._wall_ns = 0

    def start(self) -> RBLNExplain:
        global _active
        # Claim the process-global instrumentation. Regions must not overlap; a second
        # (nested or concurrent) region raises rather than silently corrupting the open
        # one's timer/trace/counters.
        with _region_lock:
            if _active:
                raise RuntimeError(
                    "torch.rbln.explain() regions must not overlap (nested or concurrent): the "
                    "(B) timer, trace and counters are process-global. Close the open region first."
                )
            _active = True
            self._holds = True
        # We now own the gate. Enable it and take the start baselines; if ANY of that raises,
        # release before propagating so a failed start never leaks the gate/guard (#2).
        try:
            if self._trace:
                _set_trace(True)
                _reset_trace()  # region-local captures (the C++ map dedups per op)
            # (B) off => one relaxed atomic load per boundary call, no clock read (ON==OFF).
            _rt_timing_reset()
            _rt_timing_enable(True)
            self._b0 = _read_bounces()
            self._d0 = _read_dispatch()
            self._f0 = _read_fallback_by_op()
            self._r0 = _read_recompile_by_op()
            self._fr0 = _read_fallback_reasons()
            self._rt0 = _read_runtime()
            self._wall0 = time.perf_counter_ns()
        except BaseException:
            self._release()
            raise
        return self

    def _release(self) -> None:
        """Disable the global instrumentation and release the region guard, exactly once.
        Called from stop()'s finally AND from a failed start() -- so neither a mid-readout
        error nor a start failure leaks the (B) timer / trace gate into the next region."""
        global _active
        with _region_lock:
            if not self._holds:
                return
            self._holds = False
            _rt_timing_enable(False)
            if self._trace:
                _set_trace(False)
            _active = False

    def stop(self) -> RBLNExplain:
        try:
            self._wall_ns = time.perf_counter_ns() - self._wall0
            self._rt_timing = _read_rt_timing()  # (B) region totals (reset at start)
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
                hb = [a - b for a, b in zip(rt1["hidden_bytes"], self._rt0["hidden_bytes"])]
                self._rt = {"hidden_count": hc, "hidden_bytes": hb}
                if "mem_cur" in rt1:
                    # memory is a process-level high-water gauge (a level, not a delta)
                    self._rt["mem_cur"] = rt1["mem_cur"]
                    self._rt["mem_peak"] = rt1["mem_peak"]
                if "host_sync_count" in rt1 and "host_sync_count" in self._rt0:
                    self._rt["host_sync_count"] = rt1["host_sync_count"] - self._rt0["host_sync_count"]
                    self._rt["host_sync_bytes"] = rt1["host_sync_bytes"] - self._rt0["host_sync_bytes"]
                if "host_sync_h2d_count" in rt1 and "host_sync_h2d_count" in self._rt0:
                    self._rt["host_sync_h2d_count"] = rt1["host_sync_h2d_count"] - self._rt0["host_sync_h2d_count"]
                    self._rt["host_sync_h2d_bytes"] = rt1["host_sync_h2d_bytes"] - self._rt0["host_sync_h2d_bytes"]
                if "reject_count" in rt1 and "reject_count" in self._rt0:
                    self._rt["reject_count"] = [a - b for a, b in zip(rt1["reject_count"], self._rt0["reject_count"])]
                    self._rt["reject_bytes"] = [a - b for a, b in zip(rt1["reject_bytes"], self._rt0["reject_bytes"])]
            else:
                self._rt = None
            self._trace_by_op = _read_trace_by_op() if self._trace else {}
        finally:
            self._release()  # always drop the gate + release the guard, even if a readout raised
        return self

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        # Like torch._dynamo.explain()'s ExplainOutput: the object renders as its report,
        # so `print(p)` works. NOT auto-printed on __exit__ (torch.profiler doesn't either).
        if self._bounces is None:  # region still open — stop() not called yet
            return f"<RBLNExplain active (trace={self._trace}); read the report after the region closes>"
        return self.report()

    __str__ = __repr__

    # -- readout -------------------------------------------------------------
    def dump(self) -> dict[str, Any]:
        if self._bounces is None:
            raise RuntimeError("region is still open -- exit the with-block or call stop() before dump()/report()")
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
        out["trace_by_op"] = dict(self._trace_by_op or {})  # (A) WHERE; {} unless with_stack=True
        if self._rt is not None:
            hc, hbytes = self._rt["hidden_count"], self._rt["hidden_bytes"]
            # (#3) the runtime hidden-reason axis is a positional ABI contract, not partially-
            # interpretable user data: report keys like src_not_on_device are only truthful if the
            # order/count/meaning line up 1:1 with _RUNTIME_REASONS. A length mismatch means this
            # torch-rbln build and the loaded librbln are out of sync -- fail loudly rather than
            # silently truncate (zip) or invent an "unattributed" bucket that report()/verdict()
            # would then have to false-interpret. The axis-length canary guards this in CI too.
            if len(hc) != len(_RUNTIME_REASONS):
                raise RuntimeError(
                    f"runtime hidden-reason axis drifted (runtime reports {len(hc)} reasons, "
                    f"this build names {len(_RUNTIME_REASONS)}); "
                    "torch-rbln and rebel-compiler/librbln are out of sync"
                )
            by_reason = {n: {"count": c, "bytes": b} for (n, _f), c, b in zip(_RUNTIME_REASONS, hc, hbytes)}
            out["runtime_residency"] = {
                "available": True,
                "total_count": sum(hc),
                "by_reason": by_reason,
            }
            if "host_sync_count" in self._rt:
                # REAL device->host this region (manager-emitted): a host bounce with 0 real
                # d2h was served on host (no device crossing). Covers synchronous + async-
                # fallback transfers; a deliberate pinned non_blocking copy takes the async
                # DMA path (uncounted by the runtime), out of scope by design (see docs §6).
                out["runtime_residency"]["real_host_sync_d2h"] = {
                    "count": self._rt["host_sync_count"],
                    "bytes": self._rt["host_sync_bytes"],
                }
            if "host_sync_h2d_count" in self._rt:
                # REAL host->device this region (manager-emitted): the lazy push at the
                # device-consume boundary -- glue that pushes host-latest data onto the device
                # for a graph to consume. Same scope caveat as real_host_sync_d2h above.
                out["runtime_residency"]["real_host_sync_h2d"] = {
                    "count": self._rt["host_sync_h2d_count"],
                    "bytes": self._rt["host_sync_h2d_bytes"],
                }
            if "reject_count" in self._rt:
                # WHY a fast v2v plan was rejected, grouped by who can act: user-
                # actionable reasons individually; every other reason collapsed into one
                # "internal" bucket (the runtime's internal classification is never named).
                user_rej: dict[str, dict[str, int]] = {}
                int_c = int_b = 0
                rc, rb = self._rt["reject_count"], self._rt["reject_bytes"]
                for i in range(len(rc)):
                    if rc[i] <= 0:
                        continue
                    aud = _REJECT_AUDIENCE[i] if i < len(_REJECT_AUDIENCE) else "internal"
                    if aud == "user":
                        user_rej[_REJECT_LABEL[i]] = {"count": rc[i], "bytes": rb[i]}
                    elif aud != "none":
                        int_c += rc[i]
                        int_b += rb[i]
                out["runtime_residency"]["reject"] = {
                    "user_actionable": user_rej,
                    "internal_fallback": {"count": int_c, "bytes": int_b},
                }
                # Diagnostic only -- NOT a stable surface. Raw per-index reject axis is
                # enum-order dependent (the runtime's internal classification), so it lives
                # under "debug" and must not be treated as a CI contract.
                out["runtime_residency"]["debug"] = {
                    "reject_raw_by_index": {i: {"count": rc[i], "bytes": rb[i]} for i in range(len(rc)) if rc[i] > 0}
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
        # (A) which fallback ops lack a CPU fast-path handler = the optimization
        # candidates (registered ones already bypass redispatchBoxed). Omitted when
        # the _C predates the registry query (can't tell handled from un-handled).
        unaccel: list[str] = []
        for op in out["cpu_fallback_by_op"]:
            if _read_fast_path_registered(op) is False:
                unaccel.append(op)
        if unaccel:
            out["cpu_fallback_unaccelerated"] = unaccel

        # (E) host CPU oversubscription (resource fact; amplifies all host overhead)
        out["host_threads"] = _host_thread_info()

        # (B) rebel-runtime (librbln) boundary time this region, per primitive.
        if self._rt_timing is not None:
            total_ns = sum(ns for ns, _c in self._rt_timing)
            by_prim = {name: {"ns": ns, "calls": cnt} for name, (ns, cnt) in zip(_RT_PRIMS, self._rt_timing) if cnt}
            out["rebel_runtime"] = {
                "total_ns": total_ns,
                "wall_fraction": (total_ns / self._wall_ns) if self._wall_ns else 0.0,
                "by_primitive": by_prim,
            }

        out["notes"] = _collect_notes(out)
        return out

    def verdict(self) -> dict[str, Any]:
        d = self.dump()
        hb, disp, rr = d["hidden_host_bounce"], d["dispatch"], d["runtime_residency"]
        reasons: list[str] = []
        if hb["total_count"] > 0:
            reasons.append(f"hidden host bounce (torch-side): {hb['total_count']}x, {_fmt_bytes(hb['total_bytes'])}")
        runtime_hidden = rr["total_count"] if rr["available"] else 0
        if runtime_hidden > 0:
            # Lead with the real d2h the manager performed to serve the fallback, as a
            # magnitude fact -- whether it is user-avoidable is the reject axis's job, not
            # a blanket label here.
            rd = rr.get("real_host_sync_d2h")
            if rd and rd["count"]:
                reasons.append(f"runtime-side physical d2h: {rd['count']}x, {_fmt_bytes(rd['bytes'])}")
            fix = dict(_RUNTIME_REASONS)
            for n, vv in rr["by_reason"].items():
                if vv["count"]:
                    reasons.append(f"runtime hidden d2h [{n}]: {vv['count']}x - {fix[n]}")
        if disp["recompile_miss"] > 0:
            reasons.append(f"recompile/cold-compile: {disp['recompile_miss']}x")
        if disp["cpu_fallback"] > 0:
            reasons.append(f"cpu_fallback: {disp['cpu_fallback']}x (ran on CPU)")
        # ``clean`` is a FACT (did any hidden event fire), NOT a severity grade. explain
        # does not colour-judge how bad an event is -- a host bounce can be free -- so a
        # RED/AMBER/GREEN would be a guess. Cost lives in the table (Bytes/Note) + reasons.
        clean = not reasons
        if clean:
            reasons.append("no hidden host bounce, no CPU fallback, no recompile")
        return {
            "clean": clean,
            "hidden_host_bounces": hb["total_count"],
            "hidden_host_bounce_bytes": hb["total_bytes"],
            "runtime_hidden_d2h": runtime_hidden if rr["available"] else None,
            "runtime_hidden_by_reason": rr.get("by_reason") if rr["available"] else None,
            "cpu_fallbacks": disp["cpu_fallback"],
            "recompiles": disp["recompile_miss"],
            "reasons": reasons,
            "notes": d.get("notes", []),
        }

    def report(self) -> str:
        """Verdict-first report. Each byte-carrying row's Note leads with the cost
        verdict (from the region-global physical-d2h witness); detail blocks are grouped
        under their parent signal; the full note prose is in dump()['notes'] / help()."""
        d, v = self.dump(), self.verdict()
        rr = d["runtime_residency"]

        # -- header: [clean]/[overhead: N] marker + region facts --------------
        if v["clean"]:
            mark = "[clean]"
        else:
            # N = how many KINDS of hidden signal fired (host bounce / runtime d2h /
            # cpu_fallback / recompile). A count, NOT a severity grade -- the tool never
            # ranks how bad; cost is read from the table Bytes/Note and the >> witness.
            nsig = sum(
                (
                    v["hidden_host_bounces"] > 0,
                    bool(v["runtime_hidden_d2h"]),
                    v["cpu_fallbacks"] > 0,
                    v["recompiles"] > 0,
                )
            )
            mark = f"[overhead: {nsig} signal{'' if nsig == 1 else 's'}]"
        head = f"{mark}  RBLN EXPLAIN   (region wall {_fmt_time(d['wall_ns'])}"
        if "device_memory" in d:
            # rebel BufferAllocator reserved footprint (incl. cached/idle buffers held
            # for reuse), a device-side high-water mark -- NOT host process RSS.
            head += f" | device mem {_fmt_bytes(d['device_memory']['peak_bytes'])} peak, reserved"
        head += ")"
        lines = [head]

        # -- context (host oversubscription, rebel-runtime share, h2d push):
        #    amplifiers/facts, each wrapped to stay self-contained and short (P1-5).
        #    Built for BOTH paths -- these are facts that happened regardless of whether
        #    any hidden signal fired, so a clean region still surfaces them (e.g. an h2d
        #    push is expected glue, not overhead -- it must not vanish just because the
        #    region is otherwise clean). --------------------------------------
        ctx: list[str] = []
        ht = d.get("host_threads") or {}
        if ht.get("oversubscribed"):
            cores = ht["cores"]
            ctx.append(
                f"  ! host oversubscription: {ht['intended_threads']} threads / {cores} "
                f"core{'' if cores == 1 else 's'} -> host latency may be inflated"
            )
            ctx.append("    (tune affinity / OMP_NUM_THREADS)")
        rtm = d.get("rebel_runtime")
        if rtm and rtm["total_ns"]:
            bp = sorted(rtm["by_primitive"].items(), key=lambda kv: -kv[1]["ns"])
            top = "  ".join(f"{n} {_fmt_dur(pv['ns'])}/{pv['calls']}" for n, pv in bp[:4])
            more = f"  +{len(bp) - 4} more" if len(bp) > 4 else ""
            ctx.append(
                f"  rebel runtime: {_fmt_time(rtm['total_ns'])} in librbln "
                f"({rtm['wall_fraction'] * 100:.1f}% of region wall)"
            )
            ctx.append(f"    {top}{more}")
        rhh = rr.get("real_host_sync_h2d") if rr.get("available") else None
        if rhh is not None and rhh["count"]:
            ctx.append(f"  host->device push (runtime): {rhh['count']} pushes, {_fmt_bytes(rhh['bytes'])}")
        if ctx:
            lines.append("")
            lines += ctx

        # -- clean path: say WHAT was checked, and that clean != fast ----------
        if v["clean"]:
            checked = ["host_bounce"]
            if rr.get("available"):  # only claim v2v_slow when the region could observe it
                checked.append("v2v_slow")
            checked += ["cpu_fallback", "recompile"]
            lines.append(f"  checked: {', '.join(checked)} -- none fired")
            lines.append('  note: clean = no hidden host overhead, not "fast"')
            if not rr.get("available"):
                lines.append(_RT_UNAVAIL)
            return "\n".join(lines)

        # -- cost verdict source: the region-global physical-d2h witness -------
        # It cannot attribute per row, so: witness==0 -> every byte row is low-cost;
        # witness>0 with exactly one byte row -> that row is the costly one; witness>0
        # with several -> rows defer to the >> line (no false per-row blame).
        phys = rr.get("real_host_sync_d2h") if rr.get("available") else None
        witness = phys["count"] if phys else 0
        bounce_sites = [(n, vv) for n, vv in d["hidden_host_bounce"]["by_site"].items() if vv["count"]]
        v2v_fired = bool(rr.get("available") and rr.get("total_count"))
        n_byte_rows = len(bounce_sites) + (1 if v2v_fired else 0)

        def _verdict() -> str:
            if witness == 0:
                return "no DMA (low cost)"
            if n_byte_rows == 1:
                return "real d2h DMA (costly)"
            return "see physical d2h below"

        # -- the signal table: fixed category order (host_bounce -> runtime ->
        #    dispatch); cost verdict leads the Note on every byte-carrying row --
        rows: list[list[str]] = []
        for name, vv in bounce_sites:
            short = _FIX_SHORT.get(name, "")
            note = f"{_verdict()} | {_fix(short)}" if short else _verdict()
            rows.append([f"host_bounce/{_site_label(name)}", _thousands(vv["count"]), _fmt_bytes(vv["bytes"]), note])
        if v2v_fired:
            # ONE event row. Bytes is host-path volume ONLY (one semantic); the physical
            # DMA magnitude, if any, lives on the >> line. v2v's remedy is per reject-cause,
            # shown in the detail block, so the row Note is just the cost verdict.
            host_path_bytes = sum(x.get("bytes", 0) for x in rr["by_reason"].values())
            rows.append(["runtime/v2v_slow", _thousands(rr["total_count"]), _fmt_bytes(host_path_bytes), _verdict()])
        disp = d["dispatch"]
        if disp["cpu_fallback"]:
            rows.append(
                ["dispatch/cpu_fallback", _thousands(disp["cpu_fallback"]), "--", _fix(_FIX_SHORT["cpu_fallback"])]
            )
        if disp["recompile_miss"]:
            rows.append(["dispatch/recompile", _thousands(disp["recompile_miss"]), "--", _fix(_FIX_SHORT["recompile"])])

        lines.append("")
        lines += _table(["Signal", "Count", "Bytes", "Note"], rows, ["l", "r", "r", "l"])
        lines.append("")

        tbo = d.get("trace_by_op") or {}  # op/site -> call-site, only with with_stack=True

        # -- detail blocks: grouped under their parent signal, in table order --
        for name, _vv in bounce_sites:  # (1) host_bounce: the bounce call-site (stack only)
            if name in tbo:
                lines.append(f"  host_bounce/{_site_label(name)}:")
                lines.append(f"    at {_short_path(tbo[name])}")
                lines.append("")
        if v2v_fired:  # (2) v2v_slow: the src-state axis + the reject (who-can-act) axis
            lines.append("  runtime/v2v_slow:")
            st = "  ".join(f"{n} {_thousands(x['count'])}" for n, x in rr["by_reason"].items() if x["count"])
            if st:
                lines.append(f"    state:  {st}")
            rj = rr.get("reject") or {}
            rparts = [
                f"{lbl} {_thousands(x['count'])} -> {_REJECT_FIX.get(lbl, '')}"
                for lbl, x in rj.get("user_actionable", {}).items()
            ]
            if rj.get("internal_fallback", {}).get("count"):
                rparts.append(
                    f"internal_fallback {_thousands(rj['internal_fallback']['count'])} "
                    "(runtime-internal, not user-fixable)"
                )
            if rparts:
                lines.append(f"    reject: {' | '.join(rparts)}")
            lines.append("")
        fbo = d.get("cpu_fallback_by_op") or {}
        if fbo:  # (3) cpu_fallback: per-op count + call-site (aligned), why, candidates
            fb_ns = disp.get("cpu_fallback_ns", 0)
            cost = f"  (sum {_fmt_time(fb_ns)} wall)" if fb_ns else ""
            lines.append(f"  dispatch/cpu_fallback:{cost}")
            lines += _op_block(fbo, tbo)
            fr = d.get("cpu_fallback_reasons") or {}
            if fr:
                lines.append("    why: " + ", ".join(f"{n} {_thousands(c)}" for n, c in fr.items()))
            unaccel = d.get("cpu_fallback_unaccelerated") or []
            if unaccel:
                shown = ", ".join(_short_op(o) for o in unaccel[:8]) + (
                    "" if len(unaccel) <= 8 else f", +{len(unaccel) - 8} more"
                )
                lines.append(f"    candidates (no fast-path handler): {shown}")
            lines.append("")
        rbo = d.get("recompile_by_op") or {}
        if rbo:  # (4) recompile
            lines.append("  dispatch/recompile:")
            lines += _op_block(rbo, tbo)
            lines.append("")

        # -- the single most important line, promoted out of the pile + scoped -
        if bounce_sites or v2v_fired:
            if witness:
                scope = " (see runtime/v2v_slow)" if v2v_fired else ""
                lines.append(
                    f"  >> physical d2h (real transfer): {phys['count']} copies, {_fmt_bytes(phys['bytes'])}"
                    f" -- real device crossing{scope}"
                )
            else:
                lines.append("  >> physical d2h (real transfer): 0 -- everything above served on host, no crossing")
            lines.append("")

        if (fbo or rbo) and not tbo:
            lines.append("  where? -> rerun with explain(with_stack=True)")
        if not rr["available"]:
            lines.append(_RT_UNAVAIL)
        lines.append("  (detail: p.help(signal) | raw: p.dump())")
        return "\n".join(lines)

    def help(self, signal: Optional[str] = None) -> str:
        """Full remedy prose. No arg: the fix for every fired signal. With a
        signal (bare cause or report label like 'dispatch/cpu_fallback'): just
        that one."""
        fixes = self.dump().get("notes", [])
        if signal is None:
            return "\n".join(f"{f['signal']}: {f['fix']}" for f in fixes) or "no hidden overhead"
        key = signal.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if key in _REMEDY:
            return _REMEDY[key]
        rfix = dict(_RUNTIME_REASONS)
        if key in rfix:
            return rfix[key]
        if key in _REJECT_FIX:
            return _REJECT_FIX[key]
        if key == "internal_fallback":
            return "runtime-internal fallback; not user-fixable (a notification, not a to-do)"
        return next((f["fix"] for f in fixes if signal in f["signal"]), f"no remedy for '{signal}'")

    def diff(self, other: RBLNExplain) -> RBLNDiff:
        """Compare this region with another one YOU placed (self -> other), e.g.
        an early call vs a later steady call. explain cannot tell a one-time cost
        from a recurring one on its own (it does not know your run structure);
        placing the two regions is how YOU supply that. See :class:`RBLNDiff`."""
        # Errors are UI: a bare assert vanishes under -O and its message doesn't say what
        # to do. Raise something actionable instead.
        if self._bounces is None:
            raise RuntimeError("region 'a' is still open -- exit its with-block or call stop() before diff()")
        if other._bounces is None:
            raise RuntimeError("region 'b' is still open -- exit its with-block or call stop() before diff()")
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
            # The authoritative real device->host transfer (the v2v_slow magnitude). Tracked
            # so a recurring internal-fallback cost shows up in the diff, not just its count.
            da_d2h = (ra.get("real_host_sync_d2h") or {}).get("count", 0)
            db_d2h = (rb.get("real_host_sync_d2h") or {}).get("count", 0)
            if da_d2h or db_d2h:
                rows.append(("runtime/physical_d2h", da_d2h, db_d2h))
            ha = (ra.get("real_host_sync_h2d") or {}).get("count", 0)
            hbv = (rb.get("real_host_sync_h2d") or {}).get("count", 0)
            if ha or hbv:
                rows.append(("runtime/h2d_push", ha, hbv))
        rows.append(("cpu_fallback", da["dispatch"]["cpu_fallback"], db["dispatch"]["cpu_fallback"]))
        rows.append(("recompile", da["dispatch"]["recompile_miss"], db["dispatch"]["recompile_miss"]))
        return rows

    @staticmethod
    def _sig_bytes(dump: dict[str, Any], name: str) -> Optional[int]:
        """Host-traffic byte magnitude for a signal in one region's dump, or None if the
        signal carries no byte measure (cpu_fallback/recompile are counts, not transfers).
        Mirrors the single-region row magnitude: a real device->host transfer if one
        happened, else the host-served copy volume -- so the diff can rank by COST, not
        just recurrence (a 12 KB host-served loop and a 12 GB one read identically by count)."""
        if name == "host_bounce":
            return dump["hidden_host_bounce"]["total_bytes"]
        rr = dump.get("runtime_residency") or {}
        if not rr.get("available"):
            return None
        if name == "runtime/v2v_slow":
            phys = rr.get("real_host_sync_d2h") or {}
            if phys.get("count"):
                return phys["bytes"]
            return sum(v.get("bytes", 0) for v in rr["by_reason"].values())
        if name == "runtime/physical_d2h":
            return (rr.get("real_host_sync_d2h") or {}).get("bytes", 0)
        if name == "runtime/h2d_push":
            return (rr.get("real_host_sync_h2d") or {}).get("bytes", 0)
        return None

    def dump(self) -> dict[str, Any]:
        """{'signals': {name: {'a','b'[,'a_bytes','b_bytes']}}, 'persists'|'gone'|'appeared':
        [names], 'persists_by_op': [{signal, op, count, at}], 'device_memory': {...}}.
        ``persists`` (present in ``b``) is the actionable, recurring set. Byte-carrying
        signals also carry ``a_bytes``/``b_bytes`` (the host-traffic magnitude) so a
        recurring cost can be ranked by size, not just by recurrence."""
        out: dict[str, Any] = {"signals": {}, "persists": [], "gone": [], "appeared": [], "persists_by_op": []}
        for name, av, bv in self._rows():
            sig: dict[str, int] = {"a": av, "b": bv}
            ab, bb = self._sig_bytes(self._a, name), self._sig_bytes(self._b, name)
            if ab is not None or bb is not None:
                sig["a_bytes"], sig["b_bytes"] = ab or 0, bb or 0
            out["signals"][name] = sig
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
            mark = ">> persists" if bv > 0 else ("gone in B" if av > 0 else "")
            # Byte magnitude per region for byte-carrying signals ("--" for count-only
            # signals like cpu_fallback) -> a persisting cost shows its SIZE, so you can
            # tell a recurring 12 KB host-served loop from a 12 GB one at a glance.
            ab, bb = vv.get("a_bytes"), vv.get("b_bytes")
            rows.append([name, str(av), str(bv), _fmt_bytes(ab) if ab else "--", _fmt_bytes(bb) if bb else "--", mark])
        lines += _table(["Signal", "A", "B", "A bytes", "B bytes", ""], rows, ["l", "r", "r", "r", "r", "l"])
        pbo = d.get("persists_by_op") or []
        if pbo:
            lines.append("")
            lines.append("  >> persists across your two points -> recurring overhead, act on these:")
            for e in pbo[:8]:
                at = f"  @ {e['at']}" if e.get("at") else ""
                lines.append(f"    {e['signal']}: {e['op']} {e['count']}{at}")
        elif not d["persists"]:
            lines.append("")
            lines.append("  nothing persisted into B (no recurring overhead across your two points)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()

    __str__ = __repr__


def explain(with_stack: bool = False, *, trace: Optional[bool] = None) -> RBLNExplain:
    """Return a hidden-overhead explain region, usable as a context manager.

    With ``with_stack=True`` (opt-in; OFF by default, so a plain region adds nothing to
    any path), the FIRST time each op falls back / recompiles its Python call-site
    is captured (deduped per op) and shown as an ``at <file:line(func)>`` line in
    the report — telling you WHERE in your model the hidden overhead originates.

    ``trace=`` is a deprecated back-compat alias for ``with_stack=`` (torch names this
    feature ``with_stack`` in ``torch.profiler.profile``); it still works."""
    if trace is not None:
        with_stack = trace
    return RBLNExplain(with_stack=with_stack)


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
    with_stack: bool = False,
    trace: Optional[bool] = None,
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
    if trace is not None:  # deprecated back-compat alias for with_stack
        with_stack = trace
    cold = RBLNExplain(with_stack=with_stack).start()
    try:
        fn()
    finally:
        cold.stop()
    for _ in range(max(0, warmup)):
        fn()
    warm = RBLNExplain(with_stack=with_stack).start()
    try:
        fn()
    finally:
        warm.stop()
    if as_diff:
        return cold.diff(warm)
    return (cold, warm) if return_cold else warm
