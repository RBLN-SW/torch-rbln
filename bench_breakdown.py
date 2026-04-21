"""Break down the B_ON steady-state cost for torch.add into stages via the
timing counters exported from torch_rbln/_C.*.so.

Stages (run_cached template, all in rebel runtime after our guard+find):
  alloc       -> at::empty for output tensor
  build_maps  -> std::map<uint32_t, uint64_t> for dev_in / dev_out
  prepare_in  -> PyRblnSyncRuntime::PrepareInputs
  prepare_out -> PyRblnSyncRuntime::PrepareOutputs
  run         -> PyRblnSyncRuntime::Run (device launch + wait)
  total       -> sum of above (time inside run_cached)
  bench_wall  -> wall-clock per-call time measured in Python (includes
                 ATen dispatcher entry, our guard/find, return path)

Requires the C++ lib built with ``-DTORCH_RBLN_B_TIMING=ON``; otherwise the
counters return zeros.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

import torch
import torch_rbln  # noqa: F401
from _b_toggle import STAGES, read_counters, set_b

def run_one(size: int, pool_n: int, warmup: int, iters: int) -> dict:
    device = torch.device("rbln:0")

    # Build pool.
    pool = [
        (
            torch.full((size,), float(k % 7) + 1.0, dtype=torch.float16, device=device),
            torch.full((size,), float((k * 3 + 1) % 11) + 1.0, dtype=torch.float16, device=device),
        )
        for k in range(pool_n)
    ]

    # Warm-up on first pair (also triggers compile + cache).
    for _ in range(warmup):
        _ = torch.add(*pool[0])

    # Reset counters after warmup — we only want measurement iters.
    read_counters()

    wall_t0 = time.perf_counter_ns()
    for i in range(iters):
        a, b = pool[i % pool_n]
        _ = torch.add(a, b)
    wall_total_ns = time.perf_counter_ns() - wall_t0

    cnt = read_counters()
    n = cnt["n_calls"]
    assert n == iters, f"n_calls={n} expected {iters}"

    avg_ns = {k: cnt[k] / n for k in STAGES if k != "n_calls"}
    avg_ns["bench_wall"] = wall_total_ns / n
    # "python_extra" = what the bench loop sees minus the C++ kernel total.
    # Includes ATen dispatcher entry/exit, py::object return path, and Python
    # loop overhead (pool index, tuple unpack).
    avg_ns["python_extra"] = avg_ns["bench_wall"] - avg_ns["total"]
    return avg_ns


def main() -> int:
    size = 1024
    pool_n = 16
    warmup = 100
    iters = 2000

    # Use B_ON only — we're breaking down the B path itself.
    set_b(True)

    print(f"size={size}  pool={pool_n}  warmup={warmup}  iters={iters}",
          flush=True, file=sys.stderr)
    print(flush=True, file=sys.stderr)

    stats = run_one(size, pool_n, warmup, iters)

    order = [
        "alloc", "build_maps",
        "prepare_in", "prepare_out", "run", "total",
        "python_extra", "bench_wall",
    ]
    maxlabel = max(len(k) for k in order)

    total_us = stats["total"] / 1000.0
    wall_us = stats["bench_wall"] / 1000.0

    print(f"{'stage':<{maxlabel}}  {'avg_ns':>10}  {'avg_us':>9}  {'% of kernel total':>18}  {'% of bench wall':>16}",
          flush=True, file=sys.stderr)
    for k in order:
        ns = stats[k]
        us = ns / 1000.0
        pct_total = 100.0 * ns / stats["total"] if stats["total"] > 0 else 0.0
        pct_wall = 100.0 * ns / stats["bench_wall"] if stats["bench_wall"] > 0 else 0.0
        print(
            f"{k:<{maxlabel}}  {ns:>10,.0f}  {us:>8.2f}us  {pct_total:>17.1f}%  {pct_wall:>15.1f}%",
            flush=True,
            file=sys.stderr,
        )

    print(flush=True, file=sys.stderr)
    print(
        f"kernel total (C++): {total_us:.2f}us,  bench wall (Py loop): {wall_us:.2f}us,  "
        f"delta (Py+dispatcher): {wall_us - total_us:.2f}us",
        flush=True,
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
