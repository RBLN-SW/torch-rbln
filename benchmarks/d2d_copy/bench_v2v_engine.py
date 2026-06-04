"""Probe: does the new strided v2v engine reach the fast path on the sub-block
KV slicing geometry?

The sub-block KV workload is `kv[:, dst, :, :, :n, :] = kv[:, src, :, :, :n, :]`,
which dispatches to `aten::copy_`. Whether it reaches strided_v2v_copy depends
on the routing in RBLNCopy.cpp.

torch-rbln registers native v2v kernels for cat / index_select / index_copy /
repeat_interleave that always drive the engine. This script compares
`aten::copy_` against `index_copy_` on the same geometry, so it doubles as an
engine-reachability probe independent of how copy_ is routed.

Geometry:
  kv[:, dst, :, :, :n, :] = src_block   # equivalent to
  kv.index_copy_(1, torch.tensor([dst], device='rbln'), src_block.unsqueeze(1))
where src_block has shape (2, 8, 1, n, 128).
"""

import time

import rebel  # noqa: F401
import torch


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


def make_kv():
    return torch.empty(
        (2, 16, 8, 1, 1024, 128), dtype=torch.float16, device="rbln"
    )


def bench_copy_via_slice(n):
    kv = make_kv()
    src = kv[:, 0, :, :, :n, :]
    dst = kv[:, 1, :, :, :n, :]
    return time_call(lambda: dst.copy_(src))


def bench_index_copy_full_block(n):
    """index_copy_ over the dim-1 axis. n_tokens=block_size=1024 (no inner slice).

    src must be a freshly allocated contig tensor matching the full kv[:, [dst], ...]
    geometry. This expresses 'overwrite the dst block with a copy of the src block'
    via the new native index_copy_ kernel that goes through strided_v2v_copy.
    """
    kv = make_kv()
    src_block = torch.empty((2, 1, 8, 1, 1024, 128), dtype=torch.float16, device="rbln")
    idx = torch.tensor([1], dtype=torch.long, device="rbln")
    return time_call(lambda: kv.index_copy_(1, idx, src_block))


def bench_cat_full_block(n):
    """cat along dim=1 with two halves of the kv tensor. Produces a fresh tensor
    so this is a different operation — just to see what cat costs on this geometry."""
    kv = make_kv()
    a = kv[:, :8, ...]
    b = kv[:, 8:, ...]
    return time_call(lambda: torch.cat([a, b], dim=1))


def bench_index_select(n):
    """index_select reads from a single dim-1 index. Times the read side only."""
    kv = make_kv()
    idx = torch.tensor([0], dtype=torch.long, device="rbln")
    return time_call(lambda: kv.index_select(1, idx))


def main():
    print("v2v engine reach test\n")
    print("All numbers in us/op. kv shape (2,16,8,1,1024,128) fp16, ~64 MB total.")
    print()

    for n in (128, 896, 1024):
        t_copy = bench_copy_via_slice(n)
        print(f"  n={n:4d}  aten::copy_ via slice  : {t_copy:9.1f} us  (host-bounce fallback)")

    print()
    t_ic = bench_index_copy_full_block(0)
    print(f"  full-block index_copy_ (dim=1)   : {t_ic:9.1f} us")
    t_cat = bench_cat_full_block(0)
    print(f"  full cat (dim=1, two halves)     : {t_cat:9.1f} us")
    t_isel = bench_index_select(0)
    print(f"  index_select (dim=1, k=1)        : {t_isel:9.1f} us")


if __name__ == "__main__":
    main()
