"""Measure B vs baseline when tensor addresses vary every call.

Rationale: the earlier benchmark reused the same `a`, `b` tensors across
iterations. If rebel's runtime skips CS patching when addresses are unchanged,
that would artificially flatter C_ON's hot-path numbers. This benchmark
allocates fresh tensors each iteration so the runtime must re-patch addresses
every call.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

import torch
import torch_rbln  # noqa: F401
from _kernel_toggle import set_enabled


def time_add_varying(size: int, iters: int) -> list[float]:
    """Per-iteration latency in µs, allocating fresh inputs each call."""
    device = torch.device("rbln:0")
    lat = []
    for i in range(iters):
        va = float(i % 7)
        vb = float((i * 3 + 1) % 11)
        a = torch.full((size,), va, dtype=torch.float16, device=device)
        b = torch.full((size,), vb, dtype=torch.float16, device=device)
        t0 = time.perf_counter_ns()
        _ = torch.add(a, b)
        t1 = time.perf_counter_ns()
        lat.append((t1 - t0) / 1000.0)
    return lat


def time_add_fixed(a, b, iters: int) -> list[float]:
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = torch.add(a, b)
        t1 = time.perf_counter_ns()
        lat.append((t1 - t0) / 1000.0)
    return lat


def summarize(lat):
    srt = sorted(lat)
    n = len(srt)
    return {
        "min": srt[0],
        "p50": srt[n // 2],
        "p99": srt[min(n - 1, int(n * 0.99))],
        "mean": statistics.mean(srt),
    }


def main() -> int:
    sizes = [16, 256, 1024, 16384, 262144, 1048576]
    warmup = 30
    iters = 300

    print(
        f"{'size':>8}  {'mode':>6}  {'pattern':>9}  "
        f"{'min':>9}  {'p50':>9}  {'p99':>9}  {'mean':>9}",
        flush=True,
        file=sys.stderr,
    )
    for size in sizes:
        dev = torch.device("rbln:0")

        for label, b_enabled in [("C_ON", True), ("C_OFF", False)]:
            set_enabled(b_enabled)

            # warm-up once
            a_w = torch.ones(size, dtype=torch.float16, device=dev)
            b_w = torch.ones(size, dtype=torch.float16, device=dev)
            for _ in range(warmup):
                _ = torch.add(a_w, b_w)

            lat_fixed = time_add_fixed(a_w, b_w, iters)
            lat_vary = time_add_varying(size, iters)

            s_fix = summarize(lat_fixed)
            s_var = summarize(lat_vary)

            for patt, s in [("fixed", s_fix), ("varying", s_var)]:
                print(
                    f"{size:>8d}  {label:>6s}  {patt:>9s}  "
                    f"{s['min']:>8.2f}us  {s['p50']:>8.2f}us  {s['p99']:>8.2f}us  {s['mean']:>8.2f}us",
                    flush=True,
                    file=sys.stderr,
                )

            del a_w, b_w
            gc.collect()

        print(flush=True, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
