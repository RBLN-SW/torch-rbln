# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 8: c10d distributed on RBLN (`rbln-ccl` backend).

Ported from walkthrough_guide/8.c10d_rbln_ccl.py.

Verifies that PyTorch distributed (c10d) works with RBLN using the
`rbln-ccl` backend:
- `dist.init_process_group(backend="rbln-ccl")`.
- `all_reduce` with SUM.
- `broadcast` from rank 0 to all ranks.
- `all_gather` — each rank contributes one tensor; all ranks receive
  the full list.

Multi-process launch uses `torch.multiprocessing.spawn`; each rank runs on
`rbln:{rank}`. The test requires at least two logical RBLN devices.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import configure_master_port_for_rccl_tests, requires_logical_devices


def _setup_environment(rank: int, world_size: int) -> None:
    """Match the distributed setup used by `test/distributed/test_process_group.py`."""
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.rbln.set_device(rank)


def _run_all_reduce_sum(rank: int, world_size: int, backend: str) -> None:
    """Per-rank body for the `all_reduce` SUM check."""
    _setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"rbln:{rank}")

    try:
        base_value = rank + 1.0
        tensor = torch.full([64], base_value, dtype=torch.float16, device=device)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Ranks are 0..world_size-1, so values are 1..world_size.
        # SUM == world_size * (world_size + 1) / 2.
        expected_sum = world_size * (world_size + 1) / 2.0
        assert tensor[0] == expected_sum, (
            f"all_reduce SUM failed on rank {rank}: expected {expected_sum}, got {tensor[0]}"
        )
    finally:
        dist.destroy_process_group()


def _run_broadcast(rank: int, world_size: int, backend: str) -> None:
    """Per-rank body for the broadcast-from-rank-0 check."""
    _setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"rbln:{rank}")

    try:
        if rank == 0:
            buf = torch.full([32], 42.0, dtype=torch.float16, device=device)
        else:
            buf = torch.zeros(32, dtype=torch.float16, device=device)

        dist.broadcast(buf, src=0)

        assert buf[0] == 42.0, f"broadcast failed on rank {rank}: buf[0]={buf[0]}"
    finally:
        dist.destroy_process_group()


def _run_all_gather(rank: int, world_size: int, backend: str) -> None:
    """Per-rank body for the `all_gather` check."""
    _setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"rbln:{rank}")

    try:
        local = torch.full([4], float(rank), dtype=torch.float16, device=device)
        gathered = [torch.zeros(4, dtype=torch.float16, device=device) for _ in range(world_size)]

        dist.all_gather(gathered, local)

        for other_rank, t in enumerate(gathered):
            assert t[0] == float(other_rank), (
                f"all_gather failed on rank {rank}: "
                f"expected gathered[{other_rank}][0]={float(other_rank)}, got {t[0]}"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@requires_logical_devices(2)
class TestC10dRBLNCCL(TestCase):
    """Walkthrough example 8: c10d collectives with the `rbln-ccl` backend."""

    def setUp(self):
        # Match the env setup used by test/distributed/test_process_group.py.
        os.environ["RCCL_FORCE_EXPORT_MEM"] = "1"
        os.environ["RBLN_ROOT_IP"] = "127.0.0.1"
        os.environ["RBLN_LOCAL_IP"] = "127.0.0.1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        configure_master_port_for_rccl_tests()

        self.backend = "rbln-ccl"
        self.world_size = min(torch.rbln.device_count(), 2)

    def _spawn(self, fn):
        """Launch `fn(rank, world_size, backend)` across `self.world_size` ranks."""
        mp.spawn(
            fn,
            args=(self.world_size, self.backend),
            nprocs=self.world_size,
            join=True,
        )

    def test_all_reduce_sum(self):
        """`all_reduce` with SUM should sum the rank-specific values across ranks."""
        self._spawn(_run_all_reduce_sum)

    def test_broadcast_from_rank0(self):
        """`broadcast` from rank 0 should deliver the source value to every rank."""
        self._spawn(_run_broadcast)

    def test_all_gather(self):
        """`all_gather` should give every rank the full list of per-rank tensors."""
        self._spawn(_run_all_gather)


instantiate_device_type_tests(TestC10dRBLNCCL, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
