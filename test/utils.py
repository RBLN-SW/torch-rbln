import multiprocessing
import os
import random

import numpy as np
import pytest
import torch

from torch_rbln._internal.device_arch_utils import is_atom_device, is_rebel_device
from torch_rbln._internal.ops_utils import SupportedDtypes


SUPPORTED_DTYPES = list(SupportedDtypes.dispatch)

_DEFAULT_DISTRIBUTED_MASTER_PORT = "29604"


def configure_rbln_network_for_autoport_tests() -> None:
    """Set ``RBLN_*`` IP defaults for autoport tests that spawn fresh Python processes.

    Must run in the parent test process **before** ``subprocess.run`` so children
    inherit ``RBLN_LOCAL_IP``, ``RBLN_ROOT_IP``, and optionally probed
    ``RBLN_RDMA_IP`` when ``torch_rbln`` / ``librbln`` load in the subprocess.
    See :mod:`torch_rbln._internal.rdma_env`.
    """
    from torch_rbln._internal.rdma_env import apply_default_rbln_network_environment

    apply_default_rbln_network_environment()


def configure_master_port_for_rccl_tests(default_port: str = _DEFAULT_DISTRIBUTED_MASTER_PORT) -> None:
    """Apply MASTER_PORT policy for RBLN distributed tests.

    The process group's TCP store listens on MASTER_PORT, through which rank 0 hands the others
    the RCCL group id; RCCL picks its own ports into that id. A default port is set without
    clobbering an existing MASTER_PORT, so concurrent runs keep apart by setting their own.
    """
    os.environ.setdefault("MASTER_PORT", default_port)


def assert_device_computed_dtype(dtype: torch.dtype) -> None:
    """Fail unless the device advertises `dtype` as one it computes on.

    Only fp16/bf16 are (``get_device_capability()``). A tensor of another dtype still lives in
    device memory, but its ops run on the host through the CPU fallback, so an ordering test
    written on it would exercise host copies instead of device work and pass with a broken fence.
    """
    advertised = torch.accelerator.get_device_capability()["supported_dtypes"]
    assert dtype in advertised, f"{dtype} is not computed on the device; advertised: {advertised}"


#: The relative accuracy of the NPU's division: two dlfloat16 ulps (the measured worst case is about 1.28 * 2**-9).
DLFLOAT16_DIVISION_ACCURACY = 2.0**-8


def to_dlfloat16(x: torch.Tensor) -> torch.Tensor:
    """``x`` rounded to dlfloat16 (1 sign, 6 exponent and 9 mantissa bits) as the NPU rounds a value
    of ``x.dtype`` it computes on, as float32.

    Both round to nearest, but a float16 tie goes to even and a float32 tie away from zero; every
    bfloat16 is exact. Covers dlfloat16's normal range, which holds every finite float16.
    """
    bits = x.float().view(torch.int32)
    tie = (bits >> 14) & 1 if x.dtype == torch.float16 else 1
    return ((bits + 0x1FFF + tie) & ~0x3FFF).view(torch.float32)


def rounding_division_bounds(x: torch.Tensor, y: torch.Tensor, rounding_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The interval ``(lo, hi)`` the NPU's ``torch.div(x, y, rounding_mode=rounding_mode)`` lies in.

    The NPU divides the dlfloat16 operands to within ``DLFLOAT16_DIVISION_ACCURACY`` and rounds
    the quotient as ``rounding_mode`` says, either before or after holding it in ``x.dtype``.
    """
    rounding = {"floor": torch.floor, "trunc": torch.trunc}[rounding_mode]
    quotient = to_dlfloat16(x).double() / to_dlfloat16(y).double()
    ends = (quotient * (1 - DLFLOAT16_DIVISION_ACCURACY), quotient * (1 + DLFLOAT16_DIVISION_ACCURACY))
    candidates = torch.stack([rounding(held).to(x.dtype) for end in ends for held in (end, end.to(x.dtype).double())])
    return candidates.amin(0), candidates.amax(0)


def set_deterministic_seeds(seed: int):
    """Set deterministic seeds for reproducibility.

    Note: In multiprocessing contexts (e.g., mp.spawn), each child process
    starts with a fresh random state. This function must be called within
    each spawned process to ensure reproducibility, as seeds set in the
    parent process are not inherited by child processes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def requires_devices(num_devices):
    """Pytest marker to skip test if device count is less than required.

    Note: This replaces `deviceCountAtLeast` from `torch.testing._internal`,
    which must NOT be used here. During collection, `deviceCountAtLeast`
    triggers `PrivateUse1TestBase.setUpClass()`, mutating `device_type` from
    `"privateuse1"` to `"rbln"` at the class level. This breaks
    `instantiate_device_type_tests(only_for="privateuse1")` for all files
    collected after the mutation, silently dropping most tests.
    """
    device_count = torch.rbln.device_count()
    return pytest.mark.skipif(
        device_count < num_devices,
        reason=f"Requires at least {num_devices} devices, found {device_count}",
    )


def requires_physical_devices(num_devices):
    """Pytest marker to skip test if this process sees fewer NPUs than required."""
    physical_device_count = torch.rbln.physical_device_count()
    return pytest.mark.skipif(
        physical_device_count < num_devices,
        reason=f"Requires at least {num_devices} physical devices, found {physical_device_count}",
    )


def xfail_atom(reason: str):
    """Strict xfail on ATOM, inert elsewhere.

    An unexpected pass fails the suite, signalling the marker can be removed
    once ATOM gains support.
    """
    return pytest.mark.xfail(condition=is_atom_device(), reason=reason, strict=True)


def xfail_rebel(reason: str):
    """Strict xfail on REBEL, inert elsewhere.

    An unexpected pass fails the suite, signalling the marker can be removed
    once REBEL gains support.
    """
    return pytest.mark.xfail(condition=is_rebel_device(), reason=reason, strict=True)


def spawn_target_with_clean_exit(rank: int, test_func, *args) -> None:
    """Run ``test_func`` under ``mp.spawn`` and force a clean exit on success.

    mp.spawn reports worker exit code as ProcessExitedException on SIGSEGV.
    Python teardown in a worker that has compiled any module via rebel-compiler
    segfaults in unloaded JIT .so destructors, so skip Python shutdown on
    success. Exceptions still propagate to mp.spawn for normal failure
    reporting.

    Use as the first positional argument to ``mp.spawn`` and pass the real
    target plus its args via ``args=(test_func, *test_args)``.
    """
    test_func(rank, *args)
    os._exit(0)


def run_in_isolated_process(func, *args):
    """Run `func` in a freshly spawned process and propagate failures.

    Useful when a test requires a clean process state (e.g. fresh device
    counters, singleton re-initialization, or module-level C++ state).
    The "spawn" start method guarantees no inherited state from the parent.

    `func` and every element of `args` must be picklable (module-level
    functions, primitive types, dataclasses, etc.).
    """
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=func, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{func.__name__} failed with exit code {p.exitcode}")
