"""Measure the pure CS-patching cost in B by cycling through pre-allocated
tensors. No per-iter allocation so tensor alloc does not contaminate timing.

Three patterns:
  same  — reuse one pair (rebel's "IO address not changed" fast path)
  pair2 — alternate two pairs
  pool  — rotate over 16 pairs (addresses permute fully)

If the runtime had O(1) address check + relocation cost, all three should
produce similar p50 on B_ON. If patching is expensive, `pool` should be
noticeably slower than `same`.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

import torch
import torch_rbln  # noqa: F401

_LIB = ctypes.CDLL(
    str(Path(torch_rbln.__file__).parent / "lib" / "libc10_rbln.so"),
    mode=ctypes.RTLD_GLOBAL,
)
_LIB.c10_rbln_set_b_add_enabled.argtypes = [ctypes.c_int]
_LIB.c10_rbln_set_b_add_enabled.restype = None


def set_b(enabled: bool) -> None:
    _LIB.c10_rbln_set_b_add_enabled(1 if enabled else 0)


def make_pool(size: int, n_pairs: int):
    dev = torch.device("rbln:0")
    return [
        (
            torch.full((size,), float(k % 7), dtype=torch.float16, device=dev),
            torch.full((size,), float((k * 3 + 1) % 11), dtype=torch.float16, device=dev),
        )
        for k in range(n_pairs)
    ]


def time_cycle(pool, iters: int) -> list[float]:
    lat = []
    n = len(pool)
    for i in range(iters):
        a, b = pool[i % n]
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
    sizes = [1024, 16384, 262144]
    patterns = [("same", 1), ("pair2", 2), ("pool", 16)]
    warmup = 50
    iters = 500

    print(
        f"{'size':>8}  {'mode':>5}  {'pattern':>6}  "
        f"{'min':>9}  {'p50':>9}  {'p99':>9}  {'mean':>9}",
        flush=True,
        file=sys.stderr,
    )
    for size in sizes:
        for mode_label, b_enabled in [("B_ON", True), ("B_OFF", False)]:
            set_b(b_enabled)

            # warm-up with first pool pair
            warm_pool = make_pool(size, 1)
            for _ in range(warmup):
                _ = torch.add(*warm_pool[0])

            for patt_name, n_pairs in patterns:
                pool = make_pool(size, n_pairs)
                lat = time_cycle(pool, iters)
                s = summarize(lat)
                print(
                    f"{size:>8d}  {mode_label:>5s}  {patt_name:>6s}  "
                    f"{s['min']:>8.2f}us  {s['p50']:>8.2f}us  "
                    f"{s['p99']:>8.2f}us  {s['mean']:>8.2f}us",
                    flush=True,
                    file=sys.stderr,
                )
        print(flush=True, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
