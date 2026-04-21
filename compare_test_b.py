"""Quick correctness test for Option B add kernel.

Run with: .venv/bin/python compare_test_b.py
"""

import os
import sys

os.environ.setdefault("TORCH_RBLN_LOG_LEVEL", "INFO")

import torch
import torch_rbln  # noqa: F401 — triggers library load + kernel registration


def main() -> int:
    device = torch.device("rbln:0")
    a = torch.ones(64, dtype=torch.float16, device=device)
    b = torch.ones(64, dtype=torch.float16, device=device) * 2

    # First call — cache miss path, triggers torch.compile + rebel compile
    c1 = torch.add(a, b)
    print("first call ok, c1[:4] =", c1.detach().cpu()[:4].tolist(), flush=True, file=sys.stderr)

    # Second call — cache hit, C++ hot path
    c2 = torch.add(a, b)
    print("second call ok, c2[:4] =", c2.detach().cpu()[:4].tolist(), flush=True, file=sys.stderr)

    # Reference CPU result
    ref = torch.ones(64, dtype=torch.float16) + torch.ones(64, dtype=torch.float16) * 2
    c1_cpu = c1.detach().cpu()
    c2_cpu = c2.detach().cpu()
    if not torch.equal(c1_cpu, ref):
        print("MISMATCH c1 vs ref:", c1_cpu[:8].tolist(), "vs", ref[:8].tolist(), file=sys.stderr)
        return 1
    if not torch.equal(c2_cpu, ref):
        print("MISMATCH c2 vs ref:", c2_cpu[:8].tolist(), "vs", ref[:8].tolist(), file=sys.stderr)
        return 1
    print("PASS — add kernel B produces correct results on hit and miss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
