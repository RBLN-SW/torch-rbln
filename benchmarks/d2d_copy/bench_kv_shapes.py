"""KV-cache shape sweep for sub-block d2d copy.

Varies num_kv_heads, head_size, block_size, n_tokens. For each (shape, n)
combination, measures `dst.copy_(src)` where:

  kv shape: (2, num_blocks, num_kv_heads, 1, block_size, head_size)
  src/dst: kv[:, src_idx, :, :, :n, :]  /  kv[:, dst_idx, :, :, :n, :]

Reports the geometry that strided_v2v_copy would see (computed in Python
matching common_inner_start logic): outer_count, inner block bytes, and view
span. Run once per build configuration under test.
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


def time_call(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6


def contig_suffix_start(sizes, strides):
    """Replicates RBLNStrideUtils.h::contig_suffix_start (single-stride form)."""
    rank = len(sizes)
    expected = 1
    j = rank
    for i in range(rank - 1, -1, -1):
        if sizes[i] == 1:
            j = i
            continue
        if strides[i] == 0:
            break
        if strides[i] == expected:
            expected *= sizes[i]
            j = i
        else:
            break
    return j


def common_inner_start(sizes, src_strides, dst_strides):
    """Replicates RBLNStrideUtils.h::common_inner_start."""
    rank = len(sizes)
    expected = 1
    j = rank
    for i in range(rank - 1, -1, -1):
        if sizes[i] == 1:
            j = i
            continue
        if src_strides[i] == 0 or dst_strides[i] == 0:
            break
        if src_strides[i] == expected and dst_strides[i] == expected:
            expected *= sizes[i]
            j = i
        else:
            break
    return j


def storage_span_bytes(sizes, strides, elem_size):
    """Replicates at::detail::computeStorageNbytes."""
    if any(s == 0 for s in sizes):
        return 0
    span = 0
    for sz, st in zip(sizes, strides):
        if sz > 1:
            span += (sz - 1) * st
    return (span + 1) * elem_size


def bench_shape(num_blocks, num_kv_heads, block_size, head_size, n_tokens, iters=10):
    kv = torch.empty(
        (2, num_blocks, num_kv_heads, 1, block_size, head_size),
        dtype=DTYPE,
        device="rbln",
    )
    src = kv[:, 0, :, :, :n_tokens, :]
    dst = kv[:, 1, :, :, :n_tokens, :]
    us = time_call(lambda: dst.copy_(src), iters=iters)

    # Geometry that strided_v2v_copy would see.
    sizes = list(src.shape)
    src_strides = list(src.stride())
    dst_strides = list(dst.stride())
    inner_start = common_inner_start(sizes, src_strides, dst_strides)
    outer_count = 1
    for i in range(inner_start):
        outer_count *= sizes[i]
    inner_elems = 1
    for i in range(inner_start, len(sizes)):
        inner_elems *= sizes[i]
    src_span = storage_span_bytes(sizes, src_strides, src.element_size())
    dst_span = storage_span_bytes(sizes, dst_strides, src.element_size())

    return {
        "num_blocks": num_blocks,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "head_size": head_size,
        "n_tokens": n_tokens,
        "outer_count": outer_count,
        "inner_elems": inner_elems,
        "inner_bytes": inner_elems * ELEM_SIZE,
        "max_view_span": max(src_span, dst_span),
        "us": us,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    num_blocks = 16
    head_sizes = [64, 128]
    num_kv_heads_list = [2, 4, 8]
    block_sizes = [1024, 4096]

    rows = []
    print(f"# label={args.label}", file=sys.stderr)
    print(f"{'kv_heads':>8}  {'head':>5}  {'block':>5}  {'n_tok':>5}  "
          f"{'outer':>6}  {'inner_B':>9}  {'view_span':>11}  {'us':>10}")
    for block_size in block_sizes:
        # Pick a few n_tokens spanning sub-block, mid, near-full, full.
        n_tokens_list = sorted({128, block_size // 8, block_size // 4, block_size // 2,
                                block_size - 128, block_size})
        for num_kv_heads in num_kv_heads_list:
            for head_size in head_sizes:
                for n in n_tokens_list:
                    # Skip degenerate n.
                    if n < 1 or n > block_size:
                        continue
                    row = bench_shape(num_blocks, num_kv_heads, block_size, head_size, n)
                    rows.append({"label": args.label, **row})
                    print(f"{num_kv_heads:>8d}  {head_size:>5d}  {block_size:>5d}  {n:>5d}  "
                          f"{row['outer_count']:>6d}  {row['inner_bytes']:>9d}  "
                          f"{row['max_view_span']:>11d}  {row['us']:>10.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
