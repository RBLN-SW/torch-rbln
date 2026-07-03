# Owner(s): ["module: PrivateUse1"]

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test.utils import configure_master_port_for_rccl_tests, requires_logical_devices, spawn_target_with_clean_exit


def _run_allreduce_with_unset_control_plane_ips(rank: int, world_size: int, backend: str) -> None:
    # The IPs must be unset on entry: this is the production path that the
    # autoport helper masks by pre-setting them. Do NOT set them here.
    assert not os.environ.get("RBLN_ROOT_IP"), "test precondition: RBLN_ROOT_IP must be unset"
    assert not os.environ.get("RBLN_LOCAL_IP"), "test precondition: RBLN_LOCAL_IP must be unset"

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.rbln.set_device(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        # The fix defaults both to loopback during process-group construction.
        assert os.environ.get("RBLN_ROOT_IP") == "127.0.0.1"
        assert os.environ.get("RBLN_LOCAL_IP") == "127.0.0.1"

        tensor = torch.full([64], rank + 1.0, dtype=torch.float16, device=torch.device(f"rbln:{rank}"))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        assert tensor[0] == expected, f"all_reduce failed on rank {rank}: expected={expected}, actual={tensor[0]}"
    finally:
        dist.destroy_process_group()


@pytest.mark.single_worker
@pytest.mark.test_set_ci
@requires_logical_devices(2)
def test_rbln_ccl_defaults_control_plane_ips_when_unset(monkeypatch):
    """rbln-ccl must init + run a collective with RBLN_ROOT_IP/RBLN_LOCAL_IP unset.

    Reproduces the production failure: the autoport helper that pre-sets the IPs
    is test-only, so a normal run leaves them unset and rank 0's rcclGetUniqueId
    fails, taking every peer down with an opaque c10d recv error. The test
    passing is also the empirical proof that rccl reads these at process-group
    construction time, not at librbln load.
    """
    monkeypatch.delenv("RBLN_ROOT_IP", raising=False)
    monkeypatch.delenv("RBLN_LOCAL_IP", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("RCCL_FORCE_EXPORT_MEM", "1")
    monkeypatch.setenv("TORCH_RBLN_C10D_ASYNC", "0")
    configure_master_port_for_rccl_tests()

    world_size = 2
    mp.spawn(
        spawn_target_with_clean_exit,
        args=(_run_allreduce_with_unset_control_plane_ips, world_size, "rbln-ccl"),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
