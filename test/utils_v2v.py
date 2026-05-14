"""Shared helpers for v2v-kernel test suites under test/python/test_*_v2v.py.

Each test file under that directory used to redefine the same four or five
helpers (`_to_dev`, `_eq`, `_close`, `_arange`, the `DEVICE` constant, the
env-var setup) — collected here once so the test files are short and
diff-friendly.

Import as:

    from test.utils_v2v import DEVICE, ENGINE_DTYPES, arange, close, eq, to_dev

Importing this module also handles env-var defaults for stand-alone runs
(`python test_foo_v2v.py`); under pytest the same env vars are set by
`test/conftest.py` before tests start.
"""

from __future__ import annotations

import os


# Must run before `import torch_rbln` for stand-alone executions; under pytest
# conftest imports torch_rbln earlier, so these are best-effort no-ops there.
os.environ.setdefault("TORCH_RBLN_EAGER_MALLOC", "1")
os.environ.setdefault("TORCH_RBLN_DEPLOY", "ON")

import torch

import torch_rbln  # noqa: F401  # binds the RBLN device


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("rbln:0")
CPU = torch.device("cpu")

# Dtypes covered by the v2v engine path (no upstream cast). float dtypes
# stress different element sizes; ints confirm bit-exact correctness.
ENGINE_DTYPES: list[torch.dtype] = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
]


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------


def to_dev(x: torch.Tensor) -> torch.Tensor:
    """Clone `x` onto the RBLN device, preserving dtype/shape."""
    out = torch.empty_like(x, device=DEVICE)
    out.copy_(x)
    return out


def arange(shape, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic CPU ramp shaped `shape`. 0-D (`shape == ()`) returns the
    scalar `0`."""
    n = 1
    for s in shape:
        n *= s
    if not shape:
        return torch.tensor(0, dtype=dtype)
    return torch.arange(n, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def eq(actual_dev: torch.Tensor, expected_cpu: torch.Tensor) -> None:
    """Bitwise equality after pulling the device tensor back to CPU. Fails
    with a tensor dump if shape/dtype differ or any element diverges."""
    actual_cpu = actual_dev.cpu()
    assert actual_cpu.shape == expected_cpu.shape, (
        f"shape mismatch: device={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)}"
    )
    assert actual_cpu.dtype == expected_cpu.dtype, (
        f"dtype mismatch: device={actual_cpu.dtype} expected={expected_cpu.dtype}"
    )
    assert torch.equal(actual_cpu, expected_cpu), f"bitwise mismatch:\n  device={actual_cpu}\n  expected={expected_cpu}"


def close(
    actual_dev: torch.Tensor,
    expected_cpu: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Tolerance-based comparison via `torch.testing.assert_close`. Use for
    cross-dtype paths where bit-exactness isn't expected."""
    actual_cpu = actual_dev.cpu()
    assert actual_cpu.shape == expected_cpu.shape, (
        f"shape mismatch: device={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)}"
    )
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=atol, rtol=rtol)
