"""Isolate the CS-patching + DMA cost inside PrepareInputs / PrepareOutputs.

Mechanism: rebel's ``PatchingManager::RelocateIOAddress`` has an early-return
(``cs_patcher.cc:340``) when ``addrs == current_addr``. If we reuse the same
tensors, the CS patching path hits that no-op and skips the host-side patch +
H2D DMA. Reshape to fresh addresses each call forces the DMA path.

The difference between the two timings is the CS-patch+DMA cost per op.

Scenarios:
  FIXED  — reuse one pair (a, b) and one output buffer. All 3 slots'
           ``current_addr`` match the incoming addr → early return → no DMA.
  CHURN  — pool of 16 input pairs, fresh output each call. Addresses change
           every call → CS patch + DMA path runs.

For outputs the fresh ``at::empty`` per call means output address always
differs call-to-call in both scenarios. The FIXED scenario's output is
pre-allocated and reused to make the comparison clean.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

import torch
import torch_rbln  # noqa: F401
from _kernel_toggle import STAGES, read_counters, set_enabled  # noqa: F401


def run_fixed(size: int, iters: int, warmup: int) -> dict:
    """All inputs and all outputs stay at the same address across iters.

    Since at::empty is called inside the kernel for outputs, we can't force
    stable output addresses from the Python side — but the kernel's allocator
    is a caching allocator, and an immediately-freed output of a fixed shape
    will usually be handed the same slot back on the next alloc. To maximise
    the chance, we keep exactly one output tensor alive at a time (no pool).
    """
    dev = torch.device("rbln:0")
    a = torch.full((size,), 2.0, dtype=torch.float16, device=dev)
    b = torch.full((size,), 3.0, dtype=torch.float16, device=dev)

    for _ in range(warmup):
        out = torch.add(a, b)
        del out  # free immediately so allocator can reuse the slot
    read_counters()

    wall_t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = torch.add(a, b)
        del out
    wall_total_ns = time.perf_counter_ns() - wall_t0

    return _package(read_counters(), wall_total_ns, iters)


def run_churn(size: int, iters: int, warmup: int) -> dict:
    dev = torch.device("rbln:0")
    pool_n = 16
    pool = [
        (
            torch.full((size,), float(k % 7) + 1.0, dtype=torch.float16, device=dev),
            torch.full((size,), float((k * 3 + 1) % 11) + 1.0, dtype=torch.float16, device=dev),
        )
        for k in range(pool_n)
    ]

    for i in range(warmup):
        _ = torch.add(*pool[i % pool_n])
    read_counters()

    wall_t0 = time.perf_counter_ns()
    for i in range(iters):
        _ = torch.add(*pool[i % pool_n])
    wall_total_ns = time.perf_counter_ns() - wall_t0

    return _package(read_counters(), wall_total_ns, iters)


def _package(cnt, wall_total_ns, iters):
    n = cnt["n_calls"]
    d = {k: (cnt[k] / n) for k in STAGES if k != "n_calls"}
    d["bench_wall"] = wall_total_ns / iters
    d["python_extra"] = d["bench_wall"] - d["total"]
    d["n_calls"] = n
    return d


def fmt_row(label, d):
    # "our_cpp" here is the portion of kernel time spent in our C++ outside
    # the rebel runtime calls — alloc + std::map construction inside
    # run_cached<N>, plus (untracked) guard/find in the per-op stub.
    our_cpp_tracked = (d["alloc"] + d["build_maps"]) / 1000.0
    our_py = d["python_extra"] / 1000.0
    return (
        f"{label:<6}  "
        f"our_cpp={our_cpp_tracked:>5.2f}us  "
        f"prepare_in={d['prepare_in']/1000:>7.2f}us  "
        f"prepare_out={d['prepare_out']/1000:>7.2f}us  "
        f"run={d['run']/1000:>7.2f}us  "
        f"py_over={our_py:>5.2f}us  "
        f"wall={d['bench_wall']/1000:>7.2f}us"
    )


def main() -> int:
    size = 1024
    warmup = 100
    iters = 2000

    set_enabled(True)
    print(f"size={size}  warmup={warmup}  iters={iters}", flush=True, file=sys.stderr)
    print(flush=True, file=sys.stderr)

    d_fix = run_fixed(size, iters, warmup)
    print(fmt_row("FIXED", d_fix), flush=True, file=sys.stderr)
    d_churn = run_churn(size, iters, warmup)
    print(fmt_row("CHURN", d_churn), flush=True, file=sys.stderr)

    print(flush=True, file=sys.stderr)
    dp_in = (d_churn["prepare_in"] - d_fix["prepare_in"]) / 1000.0
    dp_out = (d_churn["prepare_out"] - d_fix["prepare_out"]) / 1000.0
    dp_total = (d_churn["total"] - d_fix["total"]) / 1000.0
    pct = 100.0 * dp_total / (d_churn["total"] / 1000.0)
    print(
        f"delta (CS patch + DMA cost per op, steady state):\n"
        f"   prepare_in:  +{dp_in:>7.2f}us\n"
        f"   prepare_out: +{dp_out:>7.2f}us\n"
        f"   TOTAL extra: +{dp_total:>7.2f}us  ({pct:.1f}% of CHURN total)",
        flush=True,
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
