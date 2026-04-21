"""Stress-test Option B for rebel-runtime CS (command stream) patching.

Concern: rebel's runtime stamps input/output addresses into the compiled command
stream during PrepareInputs/PrepareOutputs. If any of that "re-patch on address
change" logic lived in the Python DynamoRuntime.run wrapper that Option B
bypasses, we'd see wrong results when tensor addresses vary between calls.

This test compares Option B vs the legacy Python path across several address
patterns that would expose a stuck-address bug:

  - fresh allocations every call (fresh `a`, `b`, output)
  - reused allocation, different values (write into same tensor each call)
  - interleaved fresh / reused (cache-line of addresses churns)
  - randomly permuted allocation order (address can go backward)

Each call's result is compared against a CPU reference computed from the
actual input values. A single mismatch on B_ON that does not reproduce on
B_OFF indicates the CS patching assumption in B is wrong.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "ERROR")

import torch
import torch_rbln  # noqa: F401

_LIBC10_RBLN = ctypes.CDLL(
    str(Path(torch_rbln.__file__).parent / "lib" / "libc10_rbln.so"),
    mode=ctypes.RTLD_GLOBAL,
)
# NB: the toggle is still named *_b_add_enabled in the current build; the
# refactor that renames to *_b_enabled is WIP.
_LIBC10_RBLN.c10_rbln_set_b_add_enabled.argtypes = [ctypes.c_int]
_LIBC10_RBLN.c10_rbln_set_b_add_enabled.restype = None


def set_b(enabled: bool) -> None:
    _LIBC10_RBLN.c10_rbln_set_b_add_enabled(1 if enabled else 0)


def _compare(a_rbln, b_rbln, out_rbln, label: str, i: int) -> tuple[bool, str]:
    a_cpu = a_rbln.detach().cpu()
    b_cpu = b_rbln.detach().cpu()
    ref = (a_cpu.float() + b_cpu.float()).to(torch.float16)
    got = out_rbln.detach().cpu()
    if torch.equal(got, ref):
        return True, ""
    max_err = (got.float() - ref.float()).abs().max().item()
    # Find first mismatch index for diagnostic.
    diff_idx = int((got != ref).nonzero(as_tuple=False)[0, 0])
    return False, (
        f"[{label} i={i}] MISMATCH at idx={diff_idx}: "
        f"got={got[diff_idx].item()} ref={ref[diff_idx].item()} "
        f"a_ptr=0x{a_rbln.data_ptr():x} b_ptr=0x{b_rbln.data_ptr():x} "
        f"out_ptr=0x{out_rbln.data_ptr():x} max_abs_err={max_err}"
    )


def scenario_fresh(size: int, n_iters: int, label: str) -> list[str]:
    """Fresh allocations every call — addresses change each iteration."""
    device = torch.device("rbln:0")
    errs = []
    for i in range(n_iters):
        # Vary values so CPU ref is different per call — ensures we're not
        # accidentally getting "correct" by re-reading stale output.
        va = float(i % 7)
        vb = float((i * 3 + 1) % 11)
        a = torch.full((size,), va, dtype=torch.float16, device=device)
        b = torch.full((size,), vb, dtype=torch.float16, device=device)
        c = torch.add(a, b)
        ok, msg = _compare(a, b, c, label, i)
        if not ok:
            errs.append(msg)
            if len(errs) >= 3:
                break
    return errs


def scenario_reused_buffers(size: int, n_iters: int, label: str) -> list[str]:
    """Same `a` and `b` tensors, values rewritten in-place each call."""
    device = torch.device("rbln:0")
    errs = []
    a = torch.zeros(size, dtype=torch.float16, device=device)
    b = torch.zeros(size, dtype=torch.float16, device=device)
    for i in range(n_iters):
        va = float(i % 7)
        vb = float((i * 3 + 1) % 11)
        a.fill_(va)
        b.fill_(vb)
        c = torch.add(a, b)
        ok, msg = _compare(a, b, c, label, i)
        if not ok:
            errs.append(msg)
            if len(errs) >= 3:
                break
    return errs


def scenario_alternating(size: int, n_iters: int, label: str) -> list[str]:
    """Alternate between two distinct tensor pairs (addresses oscillate)."""
    device = torch.device("rbln:0")
    errs = []
    pair0 = (
        torch.full((size,), 1.0, dtype=torch.float16, device=device),
        torch.full((size,), 2.0, dtype=torch.float16, device=device),
    )
    pair1 = (
        torch.full((size,), 4.0, dtype=torch.float16, device=device),
        torch.full((size,), 8.0, dtype=torch.float16, device=device),
    )
    for i in range(n_iters):
        a, b = pair0 if (i % 2 == 0) else pair1
        c = torch.add(a, b)
        ok, msg = _compare(a, b, c, label, i)
        if not ok:
            errs.append(msg)
            if len(errs) >= 3:
                break
    return errs


def scenario_random_addrs(size: int, n_iters: int, label: str, seed: int = 0) -> list[str]:
    """Keep a pool of tensors alive; pick random ones per call so addresses
    permute unpredictably between iterations."""
    device = torch.device("rbln:0")
    rng = random.Random(seed)
    pool_size = 16
    pool = [
        (
            torch.full((size,), float(k), dtype=torch.float16, device=device),
            torch.full((size,), float(k * 2 + 1), dtype=torch.float16, device=device),
        )
        for k in range(pool_size)
    ]
    errs = []
    for i in range(n_iters):
        a, b = pool[rng.randrange(pool_size)]
        c = torch.add(a, b)
        ok, msg = _compare(a, b, c, label, i)
        if not ok:
            errs.append(msg)
            if len(errs) >= 3:
                break
    return errs


SCENARIOS = [
    ("fresh", scenario_fresh),
    ("reused", scenario_reused_buffers),
    ("alternating", scenario_alternating),
    ("random", scenario_random_addrs),
]


def main() -> int:
    sizes = [64, 1024, 16384, 262144]
    iters = 50
    print(
        f"{'scenario':<12} {'size':>8} {'mode':>6} {'pass':>6} {'fail':>6}",
        flush=True,
        file=sys.stderr,
    )
    any_fail = False
    for size in sizes:
        for name, fn in SCENARIOS:
            for mode_name, enabled in [("B_ON", True), ("B_OFF", False)]:
                set_b(enabled)
                # Warm-up once so compile/caching happens for this shape
                # before the measurement scenarios.
                dev = torch.device("rbln:0")
                _ = torch.add(
                    torch.ones(size, dtype=torch.float16, device=dev),
                    torch.ones(size, dtype=torch.float16, device=dev),
                )
                errs = fn(size, iters, f"{name}/{mode_name}")
                passed = iters - len(errs)
                print(
                    f"{name:<12} {size:>8} {mode_name:>6} {passed:>6} {len(errs):>6}",
                    flush=True,
                    file=sys.stderr,
                )
                for e in errs:
                    print("  " + e, flush=True, file=sys.stderr)
                if errs:
                    any_fail = True

    print(flush=True, file=sys.stderr)
    if any_fail:
        print(
            "RESULT: at least one scenario failed — CS patching assumption in B is wrong",
            flush=True,
            file=sys.stderr,
        )
        return 1
    print("RESULT: all scenarios passed for B_ON and B_OFF", flush=True, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
