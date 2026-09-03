# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.physical_layout(tensor)``.

A device tensor's bytes live in one device-memory area per (node, chiplet) the
runtime placed it on. ``physical_layout`` reports those areas -- the byte
footprint of a tensor per chiplet -- for a tensor whose device allocation has
been materialized (a bound buffer or a compiled-op output).
"""

from __future__ import annotations

import math

import pytest
import torch

import torch_rbln  # noqa: F401  # binds the RBLN device + torch.rbln namespace
from test.utils import requires_logical_devices


DEVICE = torch.device("rbln:0")


@pytest.mark.test_set_ci
def test_api_visible():
    assert callable(torch.rbln.physical_layout)


@pytest.mark.test_set_ci
def test_cpu_tensor_raises():
    with pytest.raises(RuntimeError, match="RBLN tensor"):
        torch.rbln.physical_layout(torch.empty(16))


@pytest.mark.test_set_ci
@requires_logical_devices(1)
class TestPhysicalLayoutOnDevice:
    def test_bound_flat_buffer_is_one_shard(self):
        nbytes = 1 << 20
        t = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
        torch.rbln.bind_device_memory(t)

        layout = torch.rbln.physical_layout(t)

        assert layout.logical_shape == (nbytes,)
        assert layout.logical_dtype == torch.uint8
        # flat 1:1 binding: no transform, so the physical view is the logical one
        assert layout.physical_shape == (nbytes,)
        assert layout.physical_dtype == torch.uint8
        assert layout.physical_itemsize == 1
        assert len(layout.shards) == 1
        (shard,) = layout.shards
        assert shard.node_id == 0
        assert shard.nbytes == nbytes
        assert isinstance(shard.device_addr, int)
        assert shard.shape == ()  # flat byte buffer: not a tensor placement
        assert layout.nbytes_per_chiplet() == {(shard.node_id, shard.chiplet_id): nbytes}

    def test_compiled_op_output_covers_its_bytes(self):
        out = torch.matmul(
            torch.randn(128, 512, dtype=torch.float16, device=DEVICE),
            torch.randn(512, 64, dtype=torch.float16, device=DEVICE),
        )

        layout = torch.rbln.physical_layout(out)

        assert layout.logical_shape == (128, 64)
        assert layout.shards
        assert all(s.nbytes > 0 for s in layout.shards)

        physical_numel = math.prod(layout.physical_shape)
        physical_itemsize = layout.physical_itemsize  # physical_dtype may be a runtime-only name
        assert physical_numel >= out.numel()
        # each shard's shape is the slice of the physical tensor that fills its area
        for shard in layout.shards:
            assert len(shard.shape) == len(layout.physical_shape)
            assert math.prod(shard.shape) * physical_itemsize == shard.nbytes
        # every area holds whole physical elements; sharded or replicated, the total is a
        # multiple of one physical copy
        total = sum(layout.nbytes_per_chiplet().values())
        assert total % (physical_numel * physical_itemsize) == 0

    def test_unbound_tensor_raises(self):
        t = torch.empty(16, dtype=torch.float16, device=DEVICE)
        with pytest.raises(RuntimeError, match="no device memory"):
            torch.rbln.physical_layout(t)

    def test_interior_view_raises(self):
        t = torch.empty(64, dtype=torch.uint8, device=DEVICE)
        torch.rbln.bind_device_memory(t)
        with pytest.raises(RuntimeError, match="whole storage"):
            torch.rbln.physical_layout(t[8:32])
