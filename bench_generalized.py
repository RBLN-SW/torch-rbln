"""Correctness + speedup for all V1 Option B kernels (add/sub/mul/div/neg).

For each op:
  1. Correctness under address churn (pool cycling) for B_ON and B_OFF.
  2. Steady-state latency (p50/p99) for B_ON and B_OFF at one size.

The goal is to confirm the same template works across op arities (binary and
unary) and the speedup is consistent with the add-only numbers.
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
_LIB.c10_rbln_set_b_enabled.argtypes = [ctypes.c_int]
_LIB.c10_rbln_set_b_enabled.restype = None


def set_b(enabled: bool) -> None:
    _LIB.c10_rbln_set_b_enabled(1 if enabled else 0)


def make_pool_binary(size: int, n: int):
    dev = torch.device("rbln:0")
    # Offset b values away from zero so div doesn't divide by zero.
    return [
        (
            torch.full((size,), float(k % 7) + 1.0, dtype=torch.float16, device=dev),
            torch.full((size,), float((k * 3 + 1) % 11) + 1.0, dtype=torch.float16, device=dev),
        )
        for k in range(n)
    ]


def make_pool_unary(size: int, n: int):
    dev = torch.device("rbln:0")
    return [
        torch.full((size,), float(k % 7) - 3.0, dtype=torch.float16, device=dev)
        for k in range(n)
    ]


OP_TABLE = [
    # (name, fn, reference_fn, arity, make_pool_fn)
    ("add", torch.add, torch.add, 2, make_pool_binary),
    ("sub", torch.sub, torch.sub, 2, make_pool_binary),
    ("mul", torch.mul, torch.mul, 2, make_pool_binary),
    ("div", torch.div, torch.div, 2, make_pool_binary),
    ("neg", torch.neg, torch.neg, 1, make_pool_unary),
]


def summarize(lat):
    srt = sorted(lat)
    n = len(srt)
    return {
        "min": srt[0],
        "p50": srt[n // 2],
        "p99": srt[min(n - 1, int(n * 0.99))],
        "mean": statistics.mean(srt),
    }


def correctness_check(name, fn, ref_fn, pool, arity, iters: int) -> list[str]:
    errs = []
    n = len(pool)
    for i in range(iters):
        item = pool[i % n]
        if arity == 2:
            a, b = item
            got = fn(a, b).detach().cpu()
            # For div on fp16 use allclose with small rtol since fp16 div can
            # drift slightly between device and CPU.
            if name == "div":
                ref = ref_fn(a.detach().cpu().float(), b.detach().cpu().float()).to(torch.float16)
                if not torch.allclose(got, ref, rtol=1e-2, atol=1e-2):
                    max_err = (got.float() - ref.float()).abs().max().item()
                    errs.append(f"[{name} i={i}] div mismatch max_err={max_err}")
            else:
                ref = ref_fn(a.detach().cpu().float(), b.detach().cpu().float()).to(torch.float16)
                if not torch.equal(got, ref):
                    max_err = (got.float() - ref.float()).abs().max().item()
                    errs.append(f"[{name} i={i}] mismatch max_err={max_err}")
        else:
            (a,) = (item,)
            got = fn(a).detach().cpu()
            ref = ref_fn(a.detach().cpu().float()).to(torch.float16)
            if not torch.equal(got, ref):
                max_err = (got.float() - ref.float()).abs().max().item()
                errs.append(f"[{name} i={i}] mismatch max_err={max_err}")
        if len(errs) >= 3:
            break
    return errs


def time_op(fn, pool, arity, iters: int) -> list[float]:
    lat = []
    n = len(pool)
    for i in range(iters):
        item = pool[i % n]
        if arity == 2:
            a, b = item
            t0 = time.perf_counter_ns()
            _ = fn(a, b)
            t1 = time.perf_counter_ns()
        else:
            a = item
            t0 = time.perf_counter_ns()
            _ = fn(a)
            t1 = time.perf_counter_ns()
        lat.append((t1 - t0) / 1000.0)
    return lat


def main() -> int:
    size = 1024
    pool_n = 16
    warmup = 30
    iters = 500
    corr_iters = 50

    print(
        f"{'op':<6} {'mode':<6} {'min':>9} {'p50':>9} {'p99':>9} {'mean':>9} {'corr':>6}",
        flush=True,
        file=sys.stderr,
    )

    any_corr_fail = False
    results = {}
    for name, fn, ref_fn, arity, make_pool in OP_TABLE:
        # warm-up: trigger compile + cache for this op (with 1 pool item).
        pool_warm = make_pool(size, 1)
        for _ in range(warmup):
            if arity == 2:
                _ = fn(*pool_warm[0])
            else:
                _ = fn(pool_warm[0])

        # pool for measurement — fresh tensors.
        pool = make_pool(size, pool_n)

        for label, enabled in [("B_ON", True), ("B_OFF", False)]:
            set_b(enabled)
            errs = correctness_check(name, fn, ref_fn, pool, arity, corr_iters)
            corr = "OK" if not errs else f"FAIL({len(errs)})"
            if errs:
                any_corr_fail = True
                for e in errs:
                    print("  " + e, flush=True, file=sys.stderr)
            lat = time_op(fn, pool, arity, iters)
            s = summarize(lat)
            results[(name, label)] = s
            print(
                f"{name:<6} {label:<6} "
                f"{s['min']:>7.2f}us {s['p50']:>7.2f}us {s['p99']:>7.2f}us {s['mean']:>7.2f}us "
                f"{corr:>6}",
                flush=True,
                file=sys.stderr,
            )

    print(flush=True, file=sys.stderr)
    print("=== speedup (p50, pool cycling) ===", flush=True, file=sys.stderr)
    for name, _fn, _ref, _arity, _mp in OP_TABLE:
        on = results[(name, "B_ON")]["p50"]
        off = results[(name, "B_OFF")]["p50"]
        sp = off / on if on > 0 else float("inf")
        saved = off - on
        print(
            f"{name:<6} B_OFF={off:>7.2f}us  B_ON={on:>7.2f}us  speedup={sp:.2f}x  saved={saved:>7.2f}us",
            flush=True,
            file=sys.stderr,
        )

    return 1 if any_corr_fail else 0


if __name__ == "__main__":
    sys.exit(main())
