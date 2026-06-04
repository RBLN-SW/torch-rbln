"""Fine-grained crossover sweep for d2d copy.

Three axes:
  outer_count    — number of v2v entries a strided copy expands into
  inner_block    — bytes per contiguous inner block
  stride_factor  — controls the view_span / numel ratio

For each cell, time `dst.copy_(view)` where:
  parent shape = (outer_count, stride_factor, inner_block_elems)
  view = parent[:, 0, :]  → shape (outer_count, inner_block_elems)
  dst  = contig empty of same shape on rbln device

Common_inner_start on (view, dst):
  view strides = (stride_factor * inner_block_elems, 1)
  dst  strides = (inner_block_elems, 1)
  → inner_start = 1, outer_count = parent.size(0), inner = inner_block_elems
  view_span = (outer_count - 1) * stride_factor * inner_block_elems + inner_block_elems
            ≈ stride_factor * total_numel (for outer_count >> 1)
"""

import argparse
import json
import sys
import time

import rebel  # noqa: F401
import torch

DTYPE = torch.float16
ELEM_SIZE = 2


def sync():
    torch.empty(1, dtype=torch.int8, device="rbln").cpu()


def time_call(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_cell(outer_count, inner_elems, stride_factor, iters=10):
    parent = torch.empty((outer_count, stride_factor, inner_elems), dtype=DTYPE, device="rbln")
    view = parent[:, 0, :]
    dst = torch.empty((outer_count, inner_elems), dtype=DTYPE, device="rbln")
    return time_call(lambda: dst.copy_(view), iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-total-mb", type=int, default=16,
                    help="Skip cells whose total view bytes exceed this many MB")
    args = ap.parse_args()

    # Grid: dense around the suspected knees.
    outers = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384]
    inner_elems_list = [1, 64, 512, 1024, 2048, 4096, 8192, 16384, 32768]  # 2 B .. 64 KB
    stride_factors = [2, 16, 256]

    rows = []
    print(f"# label={args.label}", file=sys.stderr)
    print(f"{'outer':>6}  {'inner_elems':>11}  {'inner_B':>8}  {'sf':>4}  "
          f"{'numel_B':>10}  {'span_B':>10}  {'us':>11}")
    for sf in stride_factors:
        for inner in inner_elems_list:
            for outer in outers:
                total_bytes = outer * inner * ELEM_SIZE
                if total_bytes > args.max_total_mb * (1 << 20):
                    continue
                # Parent storage = outer * sf * inner * 2 bytes; bench tensor must fit.
                parent_bytes = outer * sf * inner * ELEM_SIZE
                if parent_bytes > 256 * (1 << 20):  # cap at 256 MB
                    continue
                us = bench_cell(outer, inner, sf)
                view_span = ((outer - 1) * sf * inner + inner) * ELEM_SIZE
                rows.append({
                    "outer": outer, "inner_elems": inner, "sf": sf,
                    "inner_bytes": inner * ELEM_SIZE,
                    "numel_bytes": total_bytes,
                    "view_span": view_span,
                    "us": us,
                })
                print(f"{outer:>6d}  {inner:>11d}  {inner*ELEM_SIZE:>8d}  {sf:>4d}  "
                      f"{total_bytes:>10d}  {view_span:>10d}  {us:>11.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
