"""Compare Prepare* vs Update*Addr paths for aten::add.Tensor.

Both paths should produce correct output; the Update* path skips the vmem
manager wrapping (SetDeviceAllocConfiguration, EnsureSyncedOnPhysicalView,
MarkPhysicalViewIsUpdated, GetDeviceAddrs) and hands ``{data_ptr}`` straight
to the runtime's CS/GCE patcher.

Assumption under test: for device-resident rbln tensors whose physical view is
already latest (steady state), the vmem wrapping is a pure no-op cost and
Update* produces identical results.
"""

from __future__ import annotations

import ctypes
import os
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
_LIB.c10_rbln_set_b_use_update_addr.argtypes = [ctypes.c_int]
_LIB.c10_rbln_set_b_use_update_addr.restype = None
_LIB.c10_rbln_hp_read_and_reset.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_LIB.c10_rbln_hp_read_and_reset.restype = None

STAGES = ["n_calls", "guard", "find", "alloc", "build_maps",
          "prepare_in", "prepare_out", "run", "total"]


def set_b(enabled: bool) -> None:
    _LIB.c10_rbln_set_b_enabled(1 if enabled else 0)


def set_update(enabled: bool) -> None:
    _LIB.c10_rbln_set_b_use_update_addr(1 if enabled else 0)


def read_counters() -> dict[str, int]:
    buf = (ctypes.c_uint64 * len(STAGES))()
    _LIB.c10_rbln_hp_read_and_reset(buf)
    return {name: int(buf[i]) for i, name in enumerate(STAGES)}


def make_pool(size: int, n: int):
    dev = torch.device("rbln:0")
    return [
        (
            torch.full((size,), float(k % 7) + 1.0, dtype=torch.float16, device=dev),
            torch.full((size,), float((k * 3 + 1) % 11) + 1.0, dtype=torch.float16, device=dev),
        )
        for k in range(n)
    ]


def run_mode(label: str, use_update: bool, pool, iters: int, warmup: int) -> dict:
    set_update(use_update)

    # warmup on the measurement pool so any first-time init happens now.
    for i in range(warmup):
        a, b = pool[i % len(pool)]
        _ = torch.add(a, b)

    # Correctness sample.
    corr_errors = 0
    for i in range(20):
        a, b = pool[i % len(pool)]
        got = torch.add(a, b).detach().cpu()
        ref = (a.detach().cpu().float() + b.detach().cpu().float()).to(torch.float16)
        if not torch.equal(got, ref):
            corr_errors += 1

    # Drain counters accumulated during warmup + correctness.
    read_counters()

    wall_t0 = time.perf_counter_ns()
    for i in range(iters):
        a, b = pool[i % len(pool)]
        _ = torch.add(a, b)
    wall_total_ns = time.perf_counter_ns() - wall_t0

    cnt = read_counters()
    n = cnt["n_calls"]
    avg = {k: (cnt[k] / n if n > 0 else 0) for k in STAGES if k != "n_calls"}
    avg["bench_wall"] = wall_total_ns / iters
    avg["python_extra"] = avg["bench_wall"] - avg["total"]
    avg["corr_errors"] = corr_errors
    avg["label"] = label
    return avg


def print_row(r: dict) -> None:
    print(
        f"{r['label']:<10s}  "
        f"guard={r['guard']/1000:>5.2f}us  "
        f"find={r['find']/1000:>5.2f}us  "
        f"alloc={r['alloc']/1000:>5.2f}us  "
        f"maps={r['build_maps']/1000:>5.2f}us  "
        f"in={r['prepare_in']/1000:>7.2f}us  "
        f"out={r['prepare_out']/1000:>7.2f}us  "
        f"run={r['run']/1000:>7.2f}us  "
        f"total={r['total']/1000:>7.2f}us  "
        f"wall={r['bench_wall']/1000:>7.2f}us  "
        f"corr={r['corr_errors']}",
        flush=True,
        file=sys.stderr,
    )


def main() -> int:
    size = 1024
    pool_n = 16
    warmup = 100
    iters = 2000

    pool = make_pool(size, pool_n)
    set_b(True)

    print(f"size={size}  pool={pool_n}  warmup={warmup}  iters={iters}",
          flush=True, file=sys.stderr)
    print(flush=True, file=sys.stderr)

    r_prep = run_mode("PREPARE", False, pool, iters, warmup)
    r_upd = run_mode("UPDATE", True, pool, iters, warmup)

    print_row(r_prep)
    print_row(r_upd)
    print(flush=True, file=sys.stderr)

    # Summary
    saved_in = (r_prep["prepare_in"] - r_upd["prepare_in"]) / 1000.0
    saved_out = (r_prep["prepare_out"] - r_upd["prepare_out"]) / 1000.0
    saved_total = (r_prep["total"] - r_upd["total"]) / 1000.0
    print(
        f"delta: in -{saved_in:.2f}us  out -{saved_out:.2f}us  total -{saved_total:.2f}us",
        flush=True,
        file=sys.stderr,
    )
    if r_upd["corr_errors"] > 0:
        print(
            f"!! UPDATE path corrupted {r_upd['corr_errors']}/20 correctness checks",
            flush=True,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
