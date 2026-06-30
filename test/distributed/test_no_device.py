# Owner(s): ["module: PrivateUse1"]

"""Distributed / DTensor support with no RBLN device (``device_count() == 0``).

A sharded (tensor-parallel) model must be traceable / compilable on a host with
no NPU: the process group, the DTensor ``DeviceMesh`` built on it, and DTensors
placed on that mesh must all come up without hardware. Collectives are captured
symbolically during graph capture and only execute at runtime on real NPUs.

These contracts only manifest with zero devices, so they skip when this host has
NPUs. The companion ``test_device_mesh_with_npu_is_unaffected`` is the count > 0
regression and runs only when a device is present.
"""

import os
import unittest

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from test.utils import (
    configure_master_port_for_rccl_tests,
    requires_logical_devices,
    requires_physical_devices,
    spawn_target_with_clean_exit,
)


_HAS_NPU = torch.rbln.device_count() > 0
_NEEDS_NO_NPU = "requires a host with no physical NPU (device_count() == 0)"


# ---- spawn workers (module-level so they are picklable by mp.spawn) ---------


def _mesh_worker(rank: int, world_size: int) -> None:
    """Build an "rbln" DeviceMesh of the given world size and check its shape."""
    dist.init_process_group(backend="rbln-ccl", rank=rank, world_size=world_size)
    mesh = init_device_mesh("rbln", (world_size,))
    assert tuple(mesh.shape) == (world_size,), mesh.shape
    dist.destroy_process_group()


def _dtensor_worker(rank: int, world_size: int) -> None:
    """Shard a (meta) tensor over an "rbln" mesh — the tensor-parallel pattern.

    Meta tensors keep tracing/compilation off-device (real RBLN tensors require
    hardware); the DTensor and its sharding propagate without an NPU.
    """
    from torch.distributed.tensor import distribute_tensor, Shard

    dist.init_process_group(backend="rbln-ccl", rank=rank, world_size=world_size)
    mesh = init_device_mesh("rbln", (world_size,))
    with torch.device("meta"):
        full = torch.randn(8, 8)
    dtensor = distribute_tensor(full, mesh, [Shard(0)])
    assert tuple(dtensor.shape) == (8, 8), dtensor.shape
    dist.destroy_process_group()


def _no_device_mesh_worker(rank: int, world_size: int) -> None:
    """Like ``_mesh_worker`` but first asserts the NPUs really are hidden."""
    assert torch.rbln.device_count() == 0, "RBLN_DEVICES forcing did not yield 0 devices"
    _mesh_worker(rank, world_size)


def _no_device_dtensor_worker(rank: int, world_size: int) -> None:
    """Like ``_dtensor_worker`` but first asserts the NPUs really are hidden."""
    assert torch.rbln.device_count() == 0, "RBLN_DEVICES forcing did not yield 0 devices"
    _dtensor_worker(rank, world_size)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
class TestNoDeviceDistributed(TestCase):
    """Process group / DeviceMesh / DTensor with ``device_count() == 0``."""

    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        configure_master_port_for_rccl_tests()

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_process_group_constructs_with_world_size_gt_1(self):
        """A world_size > 1 ProcessGroupRBLN must construct with no NPU (RCCL init skipped)."""
        import torch_rbln._C

        torch_rbln._C._c10d_rbln_init()
        store = dist.HashStore()
        pg = torch_rbln._C._distributed_c10d.ProcessGroupRBLN(store, 0, 2, 0, [], gloo_backend=None)
        try:
            self.assertEqual(pg.get_size(), 2)
            self.assertEqual(pg.get_rank(), 0)
        finally:
            del pg

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_device_mesh_world_size_1_without_npu(self):
        """init_device_mesh("rbln", (1,)) must build with no NPU.

        is_initialized() reports True on a no-device host, so DeviceMesh skips its
        auto-select path (which would otherwise divide by device_count()==0 or call
        set_device(), both of which fail with no NPU).
        """
        mp.spawn(spawn_target_with_clean_exit, args=(_mesh_worker, 1), nprocs=1, join=True)

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_device_mesh_world_size_2_without_npu(self):
        """The real tensor-parallel case: a world_size=2 "rbln" DeviceMesh across
        two processes must build with no NPU."""
        mp.spawn(spawn_target_with_clean_exit, args=(_mesh_worker, 2), nprocs=2, join=True)

    @unittest.skipIf(_HAS_NPU, _NEEDS_NO_NPU)
    def test_dtensor_tensor_parallel_without_npu(self):
        """A DTensor sharded over a world_size=2 "rbln" mesh must build with no NPU."""
        mp.spawn(spawn_target_with_clean_exit, args=(_dtensor_worker, 2), nprocs=2, join=True)

    @requires_physical_devices(1)
    def test_device_mesh_world_size_2_forced_no_device(self):
        """The no-device mesh path, exercised on a real-NPU CI host by hiding the
        NPUs with a nonexistent ``RBLN_DEVICES`` filter (spawned children re-read
        the env and see ``device_count() == 0``)."""
        with pytest.MonkeyPatch.context() as ctx:
            ctx.setenv("RBLN_DEVICES", "99999")
            ctx.delenv("LOCAL_RANK", raising=False)
            mp.spawn(spawn_target_with_clean_exit, args=(_no_device_mesh_worker, 2), nprocs=2, join=True)

    @requires_physical_devices(1)
    def test_dtensor_tensor_parallel_forced_no_device(self):
        """The no-device DTensor (tensor-parallel) path, exercised on a real-NPU
        CI host by hiding the NPUs."""
        with pytest.MonkeyPatch.context() as ctx:
            ctx.setenv("RBLN_DEVICES", "99999")
            ctx.delenv("LOCAL_RANK", raising=False)
            mp.spawn(spawn_target_with_clean_exit, args=(_no_device_dtensor_worker, 2), nprocs=2, join=True)

    @requires_logical_devices(1)
    def test_device_mesh_with_npu_is_unaffected(self):
        """Regression: the device-module additions don't break a real-device mesh."""
        mp.spawn(spawn_target_with_clean_exit, args=(_mesh_worker, 1), nprocs=1, join=True)


if __name__ == "__main__":
    run_tests()
