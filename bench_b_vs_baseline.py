"""Benchmark Option B (C++ hot path) vs baseline (#3 + #19 + #20) for torch.add.

The B path is toggled at runtime via the ``c10_rbln_set_b_add_enabled`` C ABI
entry point exported from ``libc10_rbln.so``. When disabled, ``aten::add.Tensor``
calls fall through to the legacy composite-explicit-autograd path, which
allocates an output and invokes ``add.out`` → ``add_out_rbln`` (Python fast path
from #19) → ``DynamoRuntime.run`` (trimmed by #20).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")  # silence INFO spam

import torch
import torch_rbln  # noqa: F401

# Load libc10_rbln.so to flip the B toggle.
_LIBC10_RBLN = ctypes.CDLL(
    str(Path(torch_rbln.__file__).parent / "lib" / "libc10_rbln.so"),
    mode=ctypes.RTLD_GLOBAL,
)
_LIBC10_RBLN.c10_rbln_set_b_add_enabled.argtypes = [ctypes.c_int]
_LIBC10_RBLN.c10_rbln_set_b_add_enabled.restype = None
_LIBC10_RBLN.c10_rbln_get_b_add_enabled.argtypes = []
_LIBC10_RBLN.c10_rbln_get_b_add_enabled.restype = ctypes.c_int


def set_b(enabled: bool) -> None:
    _LIBC10_RBLN.c10_rbln_set_b_add_enabled(1 if enabled else 0)
    got = _LIBC10_RBLN.c10_rbln_get_b_add_enabled()
    assert got == (1 if enabled else 0), f"toggle failed: got {got}"


def time_add(a: torch.Tensor, b: torch.Tensor, iters: int) -> list[float]:
    """Return per-iteration latencies in microseconds. Host-side wall time."""
    lat_us = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = torch.add(a, b)
        t1 = time.perf_counter_ns()
        lat_us.append((t1 - t0) / 1000.0)
    return lat_us


def correctness_check(a: torch.Tensor, b: torch.Tensor) -> None:
    got = torch.add(a, b).detach().cpu()
    ref = (a.detach().cpu() + b.detach().cpu()).to(torch.float16)
    # fp16 add on rbln should match fp16 add on cpu bit-for-bit for our inputs
    # (ones + twos with small counts); if not, loosen to allclose.
    if not torch.equal(got, ref):
        # Diff diagnostic
        max_err = (got.float() - ref.float()).abs().max().item()
        raise RuntimeError(f"correctness FAIL: max_abs_err={max_err}")


def summarize(label: str, size: int, lat: list[float]) -> dict:
    srt = sorted(lat)
    n = len(srt)
    p50 = srt[n // 2]
    p99 = srt[min(n - 1, int(n * 0.99))]
    mn = srt[0]
    return {
        "label": label,
        "size": size,
        "n": n,
        "min_us": mn,
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": statistics.mean(srt),
    }


def fmt_row(r: dict) -> str:
    return (
        f"{r['label']:<10s} size={r['size']:>8d}  "
        f"min={r['min_us']:>7.2f}us  "
        f"p50={r['p50_us']:>7.2f}us  "
        f"p99={r['p99_us']:>7.2f}us  "
        f"mean={r['mean_us']:>7.2f}us  "
        f"n={r['n']}"
    )


def run_suite(sizes: list[int], warmup: int, iters: int) -> list[dict]:
    device = torch.device("rbln:0")
    results = []
    for size in sizes:
        a = torch.ones(size, dtype=torch.float16, device=device)
        b = torch.full((size,), 2.0, dtype=torch.float16, device=device)

        for label, b_enabled in [("B_ON", True), ("B_OFF", False)]:
            set_b(b_enabled)

            # Correctness before timing for each config.
            correctness_check(a, b)

            # Warm-up (compile on first call if miss; settle caches).
            for _ in range(warmup):
                _ = torch.add(a, b)

            lat = time_add(a, b, iters)
            r = summarize(label, size, lat)
            results.append(r)
            print(fmt_row(r), flush=True)

        # Free per-size tensors before moving on.
        del a, b
        gc.collect()

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[16, 256, 1024, 16384, 262144, 1048576, 4194304],
    )
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    # Sanity: require the toggle to be exported.
    set_b(True)
    set_b(False)
    set_b(True)

    print(f"sizes={args.sizes}  warmup={args.warmup}  iters={args.iters}", flush=True)
    print(f"{'label':<10s} {'size':>13s}  {'min':>11s}  {'p50':>11s}  {'p99':>11s}  {'mean':>11s}", flush=True)

    rows = run_suite(args.sizes, args.warmup, args.iters)

    # Side-by-side speedup summary.
    print(flush=True)
    print("=== B vs baseline speedup (p50) ===", flush=True)
    for size in args.sizes:
        b_on = next(r for r in rows if r["size"] == size and r["label"] == "B_ON")
        b_off = next(r for r in rows if r["size"] == size and r["label"] == "B_OFF")
        speedup = b_off["p50_us"] / b_on["p50_us"] if b_on["p50_us"] > 0 else float("inf")
        saved = b_off["p50_us"] - b_on["p50_us"]
        print(
            f"size={size:>8d}  B_OFF p50={b_off['p50_us']:>7.2f}us  "
            f"B_ON p50={b_on['p50_us']:>7.2f}us  "
            f"speedup={speedup:.2f}x  saved={saved:>7.2f}us",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
