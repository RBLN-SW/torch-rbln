import multiprocessing
import os
import random
import sys

import numpy as np
import pytest
import torch


SUPPORTED_DTYPES = [torch.float16]

_DEFAULT_DISTRIBUTED_MASTER_PORT = "29604"


def configure_master_port_for_rccl_tests(default_port: str = _DEFAULT_DISTRIBUTED_MASTER_PORT) -> None:
    """Apply MASTER_PORT policy for RBLN distributed tests.

    When RCCL_PORT_GEN=1, RCCL uses the autoport / unique-id init path; the
    process group still needs MASTER_PORT for the TCP store. In that case the
    user must set MASTER_PORT in the environment. Otherwise we pick a free
    port for each test invocation so back-to-back distributed tests in the
    same suite don't collide if a previous test's TCP store hasn't fully
    released the socket yet.
    """
    if os.environ.get("RCCL_PORT_GEN") == "1":
        if not os.environ.get("MASTER_PORT"):
            print(
                "RCCL_PORT_GEN=1 requires MASTER_PORT to be set in the environment "
                f"(e.g. export MASTER_PORT=29500). Unset RCCL_PORT_GEN or set it to "
                f"empty to use the test default ({default_port}).",
                file=sys.stderr,
            )
            sys.exit(1)
        return
    # Always rebind to a free port. The previous behaviour kept a fixed port
    # via setdefault, which bit us when an earlier test left the socket in
    # TIME_WAIT or a worker process leaked it.
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    os.environ["MASTER_PORT"] = str(free_port)


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


def requires_logical_devices(num_devices):
    """Pytest marker to skip test if logical device count is less than required.

    Note: This replaces `deviceCountAtLeast` from `torch.testing._internal`,
    which must NOT be used here. During collection, `deviceCountAtLeast`
    triggers `PrivateUse1TestBase.setUpClass()`, mutating `device_type` from
    `"privateuse1"` to `"rbln"` at the class level. This breaks
    `instantiate_device_type_tests(only_for="privateuse1")` for all files
    collected after the mutation, silently dropping most tests.
    """
    logical_device_count = torch.rbln.device_count()
    return pytest.mark.skipif(
        logical_device_count < num_devices,
        reason=f"Requires at least {num_devices} logical devices, found {logical_device_count}",
    )


def requires_physical_devices(num_devices):
    """Pytest marker to skip test if physical device count is less than required."""
    physical_device_count = torch.rbln.physical_device_count()
    return pytest.mark.skipif(
        physical_device_count < num_devices,
        reason=f"Requires at least {num_devices} physical devices, found {physical_device_count}",
    )


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
