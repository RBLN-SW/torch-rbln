"""Reproduce worst-case d2d copy patterns for strided_v2v_copy.

Routing a same-device same-shape same-dtype rbln→rbln copy through
strided_v2v_copy walks outer dims and enqueues one v2v entry per outer
iteration. For stride patterns where the largest jointly-contiguous inner
block is small, outer_count blows up and the engine loses badly to the
host-bounce baseline. Any routing heuristic must send these to host.

Cases:
  - transpose (2D): strides (1, N) on src, (N, 1) on dst → inner_block=1 elem,
    outer_count = N*M.
  - permute (3D): rotate axes so the innermost contig dim is no longer last.
  - unfold-like strided: every-other element along inner dim.
"""

import time

import rebel  # noqa: F401
import torch


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


def case_transpose(n, iters=5):
    """src = x.t(), dst = contig empty_like. Expected fan-out: n*n entries
    of 1 element each (inner_block_elems = 1)."""
    x = torch.empty((n, n), dtype=torch.float16, device="rbln")
    y = x.t()
    z = torch.empty_like(y.contiguous())  # contig same shape
    # y has strides (1, n); z has strides (n, 1)
    return time_call(lambda: z.copy_(y), iters=iters)


def case_permute_3d(shape, perm, iters=5):
    """Permute then copy_ to a contig same-shape tensor."""
    x = torch.empty(shape, dtype=torch.float16, device="rbln")
    y = x.permute(*perm)
    z = torch.empty(y.shape, dtype=torch.float16, device="rbln")
    return time_call(lambda: z.copy_(y), iters=iters)


def case_strided_inner(n, stride_inner, iters=5):
    """Slice every k-th element along the innermost dim. Inner contig block = 1."""
    x = torch.empty((n, n * stride_inner), dtype=torch.float16, device="rbln")
    y = x[:, ::stride_inner]  # shape (n, n), inner stride = stride_inner
    z = torch.empty((n, n), dtype=torch.float16, device="rbln")
    return time_call(lambda: z.copy_(y), iters=iters)


def main():
    print("Worst-case d2d copy patterns\n")

    for n in (64, 256, 1024, 2048):
        nbytes = n * n * 2
        t = case_transpose(n)
        print(f"  transpose {n}x{n} fp16  ({nbytes/1024:6.0f} KB)  : {t:10.1f} us")

    print()
    for shape, perm in [((128, 128, 128), (1, 0, 2)), ((128, 128, 128), (2, 0, 1)), ((256, 256, 64), (1, 2, 0))]:
        nbytes = shape[0] * shape[1] * shape[2] * 2
        t = case_permute_3d(shape, perm)
        print(f"  permute {shape} {perm} fp16 ({nbytes/1024:6.0f} KB) : {t:10.1f} us")

    print()
    for n, k in [(256, 2), (256, 4), (1024, 2), (1024, 4)]:
        nbytes = n * n * 2
        t = case_strided_inner(n, k)
        print(f"  strided inner [:,::{k}] of ({n}, {n*k}) fp16 ({nbytes/1024:6.0f} KB) : {t:10.1f} us")


if __name__ == "__main__":
    main()
