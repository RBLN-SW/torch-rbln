"""``torch.rbln.profile`` — surface HIDDEN torch-rbln overhead, with CAUSE.

A normal PyTorch op (``a.copy_(b)``, ``torch.cat``, an indexing write, a forward
pass) can silently round-trip the host (NPU->CPU->NPU), fall back to a CPU
kernel, recompile, or leave the NPU idle — for reasons the user never asked for
and cannot see. This profiler counts those hidden events, attributes the cause,
and renders a torch.profiler-style report.

Signals (every counter sits on an already-slow point — a host DMA, a job submit,
a host-blocking wait, an alloc — never the fast device path; reads are lazy, so
wrapping a run with profile() does not change its latency):

  host bounce (torch-side, aten intent)  : non-direct copy_ / strided fallback / etc.
  runtime host traffic (rebel leaf, bytes): d2h + h2v physical DMA, counted once each
  runtime hidden-d2h cause (v2v slow)     : why a device v2v fell to host (src state)
  dispatch                                : cpu_fallback, recompile/miss, warm-hit
  command streams (D)                     : CS submissions
  device-idle proxy (C)                   : region wall - host-blocked-on-device
  device memory (E)                       : current live + peak high-water

Two complementary, non-overlapping witnesses (a transfer is counted once): the
aten-layer "intent" counters and the runtime-leaf "physical byte" counters live
in separate tiers and are never summed.

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

# prof_get_scalars order (must match profiler_stats.h):
#   0 h2v_bytes 1 h2v_count 2 d2h_bytes 3 d2h_count 4 cs_count 5 host_wait_ns 6 mem_cur 7 mem_peak
_N_SCALARS = 8


def _read_bounces() -> list[tuple[int, int]]:
    import torch_rbln._C as _C

    return list(_C._profiler_dump_bounces())


def _read_dispatch() -> tuple[int, int, int, int, int, int]:
    import torch_rbln._C as _C

    return tuple(_C._dispatch_shim_diag_dump())


# --- runtime (rebel-compiler) counters via ctypes from the loaded librbln -----
_rt_fns: Optional[dict] = None
_rt_resolved = False


def _runtime_fns():
    global _rt_fns, _rt_resolved
    if _rt_resolved:
        return _rt_fns
    _rt_resolved = True
    for loader in (lambda: ctypes.CDLL(None), lambda: ctypes.CDLL("librbln.so")):
        try:
            lib = loader()
            hidden = lib.rbln_prof_get_v2v_hidden_d2h
            scalars = lib.rbln_prof_get_scalars
        except (OSError, AttributeError):
            continue
        hidden.restype = None
        hidden.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        scalars.restype = None
        scalars.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        _rt_fns = {"hidden": hidden, "scalars": scalars}
        return _rt_fns
    _rt_fns = None
    return None


def _read_runtime() -> Optional[dict]:
    """Snapshot of runtime counters, or None if the loaded librbln lacks them."""
    fns = _runtime_fns()
    if fns is None:
        return None
    n = len(_RUNTIME_REASONS)
    rc = (ctypes.c_uint64 * n)()
    rb = (ctypes.c_uint64 * n)()
    fns["hidden"](rc, rb, n)
    sc = (ctypes.c_uint64 * _N_SCALARS)()
    fns["scalars"](sc, _N_SCALARS)
    return {
        "hidden_count": [int(rc[i]) for i in range(n)],
        "scalars": [int(sc[i]) for i in range(_N_SCALARS)],
    }


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
        self._b0 = self._d0 = self._rt0 = self._wall0 = None
        self._bounces = self._dispatch = self._rt = None
        self._wall_ns = 0

    def start(self) -> RBLNProfile:
        self._b0 = _read_bounces()
        self._d0 = _read_dispatch()
        self._rt0 = _read_runtime()
        self._wall0 = time.perf_counter_ns()
        return self

    def stop(self) -> RBLNProfile:
        self._wall_ns = time.perf_counter_ns() - self._wall0
        b1, d1, rt1 = _read_bounces(), _read_dispatch(), _read_runtime()
        self._bounces = [(c1 - c0, by1 - by0) for (c0, by0), (c1, by1) in zip(self._b0, b1)]
        self._dispatch = tuple(x1 - x0 for x0, x1 in zip(self._d0, d1))
        if self._rt0 is not None and rt1 is not None:
            hc = [a - b for a, b in zip(rt1["hidden_count"], self._rt0["hidden_count"])]
            s0, s1 = self._rt0["scalars"], rt1["scalars"]
            flow = [s1[i] - s0[i] for i in range(6)]  # h2v/d2h bytes+count, cs, host_wait (delta)
            self._rt = {"hidden_count": hc, "flow": flow, "mem_cur": s1[6], "mem_peak": s1[7]}
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
        if self._rt is not None:
            h2v_b, h2v_c, d2h_b, d2h_c, cs, host_wait = self._rt["flow"]
            idle = max(0, self._wall_ns - host_wait)
            out["runtime_residency"] = {
                "available": True,
                "total_count": sum(self._rt["hidden_count"]),
                "by_reason": {n: {"count": c} for (n, _f), c in zip(_RUNTIME_REASONS, self._rt["hidden_count"])},
            }
            out["runtime_host_traffic"] = {
                "h2v_bytes": h2v_b,
                "h2v_count": h2v_c,
                "d2h_bytes": d2h_b,
                "d2h_count": d2h_c,
                "total_bytes": h2v_b + d2h_b,
            }
            out["command_streams"] = cs
            out["device_idle"] = {"host_wait_ns": host_wait, "idle_proxy_ns": idle, "wall_ns": self._wall_ns}
            out["device_memory"] = {"current_bytes": self._rt["mem_cur"], "peak_bytes": self._rt["mem_peak"]}
            out["pending_runtime_signals"] = ["fragmentation breakdown", "finer host-sync cause (dtype/align/chunks)"]
        else:
            out["runtime_residency"] = {"available": False}
            out["pending_runtime_signals"] = [
                "hidden_d2h / residency (STATE)",
                "host traffic bytes",
                "command streams (D)",
                "device idle (C)",
                "memory gauge (E)",
            ]
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
        # runtime physical host traffic (leaf byte tier)
        rt = d.get("runtime_host_traffic")
        if rt:
            rows.append(
                ["runtime/d2h (device->host)", str(rt["d2h_count"]), _fmt_bytes(rt["d2h_bytes"]), "physical DMA, leaf"]
            )
            rows.append(
                ["runtime/h2v (host->device)", str(rt["h2v_count"]), _fmt_bytes(rt["h2v_bytes"]), "physical DMA, leaf"]
            )
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
        if "command_streams" in d:
            rows.append(["dispatch/command_streams", str(d["command_streams"]), "-", "CS (C++ submit paths)"])
        if not rows:
            rows.append(["(clean)", "0", "-", "no hidden overhead"])

        lines += _table(["Signal", "Count", "Bytes", "Note"], rows, ["l", "r", "r", "l"])

        # device-idle proxy + memory gauge (torch.cuda.memory_summary-style line)
        if "device_idle" in d:
            di = d["device_idle"]
            if di["host_wait_ns"] > 0:
                lines.append(
                    f"  device-busy (host blocked on device): {_fmt_ms(di['host_wait_ns'])} of "
                    f"{_fmt_ms(di['wall_ns'])} wall  (idle proxy {_fmt_ms(di['idle_proxy_ns'])})"
                )
            else:
                lines.append(
                    "  device-busy/idle: not captured on the executed path "
                    "(CS/wait instrumented on C++ submit paths; TVM graph-run pending)"
                )
        if "device_memory" in d:
            m = d["device_memory"]
            lines.append(
                f"  device memory: current {_fmt_bytes(m['current_bytes'])} | peak {_fmt_bytes(m['peak_bytes'])}"
            )
        if not rr["available"]:
            lines.append("  note: runtime signals not exposed by the loaded librbln (install a recent rebel-compiler)")
        return "\n".join(lines)


def profile() -> RBLNProfile:
    """Return a profiling region usable as a context manager."""
    return RBLNProfile()
