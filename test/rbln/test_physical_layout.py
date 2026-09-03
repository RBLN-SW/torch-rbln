# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.physical_layout(tensor)``.

A device tensor's bytes live in one device-memory area per (node, chiplet) the
runtime placed it on. ``physical_layout`` reports those areas -- the byte
footprint of a tensor per chiplet and the slice of the physical tensor each one
holds -- for a tensor whose device allocation has been materialized.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  # binds the RBLN device + torch.rbln namespace


@pytest.mark.test_set_ci
def test_api_visible():
    assert callable(torch.rbln.physical_layout)


@pytest.mark.test_set_ci
def test_cpu_tensor_raises():
    with pytest.raises(RuntimeError, match="RBLN tensor"):
        torch.rbln.physical_layout(torch.empty(16))


@pytest.mark.test_set_ci
class TestPhysicalLayout(TestCase):
    def test_bound_flat_buffer_is_one_shard(self, device):
        nbytes = 1 << 20
        t = torch.empty(nbytes, dtype=torch.uint8, device=device)
        torch.rbln.bind_device_memory(t)

        layout = torch.rbln.physical_layout(t)

        self.assertEqual(layout.logical_shape, (nbytes,))
        self.assertEqual(layout.logical_dtype, torch.uint8)
        # flat 1:1 binding has no tensor transform: nothing physical to describe
        self.assertIsNone(layout.physical_shape)
        self.assertIsNone(layout.physical_dtype)
        self.assertIsNone(layout.physical_itemsize)
        self.assertEqual(len(layout.shards), 1)
        (shard,) = layout.shards
        self.assertEqual(shard.node_id, 0)
        self.assertEqual(shard.nbytes, nbytes)
        self.assertIsInstance(shard.device_addr, int)
        self.assertIsNone(shard.shape)
        self.assertEqual(layout.nbytes_per_chiplet(), {(shard.node_id, shard.chiplet_id): nbytes})

    def test_compiled_op_output_covers_its_bytes(self, device):
        out = torch.matmul(
            torch.randn(128, 512, dtype=torch.float16, device=device),
            torch.randn(512, 64, dtype=torch.float16, device=device),
        )

        layout = torch.rbln.physical_layout(out)

        self.assertEqual(layout.logical_shape, (128, 64))
        self.assertTrue(layout.shards)
        self.assertIsNotNone(layout.physical_shape)
        self.assertIsNotNone(layout.physical_itemsize)  # set even when physical_dtype is None (DLFloat16)
        self.assertGreaterEqual(math.prod(layout.physical_shape), out.numel())
        # each shard's shape is the slice of the physical tensor that fills its area
        for shard in layout.shards:
            self.assertEqual(len(shard.shape), len(layout.physical_shape))
            self.assertEqual(math.prod(shard.shape) * layout.physical_itemsize, shard.nbytes)
        # sharded or replicated, the total is a multiple of one physical copy
        total = sum(layout.nbytes_per_chiplet().values())
        self.assertEqual(total % (math.prod(layout.physical_shape) * layout.physical_itemsize), 0)

    def test_unbound_tensor_raises(self, device):
        t = torch.empty(16, dtype=torch.float16, device=device)
        with self.assertRaisesRegex(RuntimeError, "no device memory"):
            torch.rbln.physical_layout(t)

    def test_interior_view_raises(self, device):
        t = torch.empty(64, dtype=torch.uint8, device=device)
        torch.rbln.bind_device_memory(t)
        with self.assertRaisesRegex(RuntimeError, "whole storage"):
            torch.rbln.physical_layout(t[8:32])


instantiate_device_type_tests(TestPhysicalLayout, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
