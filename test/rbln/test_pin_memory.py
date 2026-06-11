# Owner(s): ["module: PrivateUse1"]

"""
Test suite for pinned host memory with the RBLN backend.

Covers the standard `pin_memory` UX while RBLN is the active accelerator:
1. `torch.tensor(..., pin_memory=True)` and `torch.empty(..., pin_memory=True)`
2. `Tensor.pin_memory()` / `Tensor.is_pinned()`
3. DataLoader with `pin_memory=True`
4. `non_blocking=True` copies from/to pinned host tensors
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestPinMemory(TestCase):
    """Pinned host allocation and is_pinned semantics (no device transfer involved)."""

    def test_tensor_factory_pin_memory(self):
        # Repro of vllm-rbln-internal#122: must not raise NotImplementedError.
        t = torch.tensor([[1], [2], [3]], dtype=torch.long, device="cpu", pin_memory=True)
        self.assertEqual(t.device.type, "cpu")
        self.assertTrue(t.is_pinned())
        self.assertEqual(t.cpu(), torch.tensor([[1], [2], [3]], dtype=torch.long))

    def test_empty_factory_pin_memory(self):
        t = torch.empty(64, 64, pin_memory=True)
        self.assertEqual(t.device.type, "cpu")
        self.assertTrue(t.is_pinned())

    def test_pin_memory_method(self):
        t = torch.randn(16, 16)
        self.assertFalse(t.is_pinned())
        pinned = t.pin_memory()
        self.assertTrue(pinned.is_pinned())
        self.assertEqual(pinned, t)
        # pin_memory() on an already-pinned tensor must be a no-op alias.
        self.assertIs(pinned.pin_memory(), pinned)

    def test_view_of_pinned_is_pinned(self):
        t = torch.randn(8, 8, pin_memory=True)
        self.assertTrue(t[2:5].is_pinned())
        self.assertTrue(t.view(-1)[7:].is_pinned())

    def test_zero_size_tensor(self):
        t = torch.empty(0, pin_memory=True)
        self.assertEqual(t.numel(), 0)

    def test_pageable_is_not_pinned(self):
        self.assertFalse(torch.randn(8).is_pinned())

    def test_pinned_freed_then_pageable_not_pinned(self):
        # After a pinned tensor is freed its address must leave the registry.
        for _ in range(4):
            t = torch.empty(1024, pin_memory=True)
            self.assertTrue(t.is_pinned())
            del t
            self.assertFalse(torch.empty(1024).is_pinned())

    def test_dataloader_pin_memory(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(32, dtype=torch.float32).reshape(8, 4))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, pin_memory=True)
        batches = [b[0] for b in loader]
        self.assertTrue(all(b.is_pinned() for b in batches))
        self.assertEqual(torch.cat(batches), torch.arange(32, dtype=torch.float32).reshape(8, 4))


@pytest.mark.test_set_ci
class TestPinMemoryCopy(TestCase):
    """Pinned host <-> RBLN device transfers, including the non_blocking path."""

    @dtypes(*SUPPORTED_DTYPES)
    def test_h2d_from_pinned(self, device, dtype):
        src = torch.randn(64, 64).to(dtype).pin_memory()
        dev = src.to(device)
        self.assertEqual(dev.cpu(), src)

    @dtypes(*SUPPORTED_DTYPES)
    def test_h2d_from_pinned_non_blocking(self, device, dtype):
        src = torch.randn(64, 64).to(dtype).pin_memory()
        dev = src.to(device, non_blocking=True)
        torch.rbln.synchronize()
        self.assertEqual(dev.cpu(), src)

    @dtypes(*SUPPORTED_DTYPES)
    def test_d2h_to_pinned_non_blocking(self, device, dtype):
        src = torch.randn(64, 64).to(dtype)
        dev = src.to(device)
        dst = torch.empty_like(src, pin_memory=True)
        dst.copy_(dev, non_blocking=True)
        torch.rbln.synchronize()
        self.assertTrue(dst.is_pinned())
        self.assertEqual(dst, src)

    def test_pinned_roundtrip_unaligned(self, device):
        # Odd byte count exercises the unaligned D2H bounce path under pinned dst.
        src = torch.randn(4097).to(torch.float16).pin_memory()
        dev = src.to(device)
        dst = torch.empty_like(src, pin_memory=True)
        dst.copy_(dev, non_blocking=True)
        torch.rbln.synchronize()
        self.assertEqual(dst, src)

    def test_pageable_non_blocking_safe_without_sync(self, device):
        # CUDA semantics: a non_blocking copy to pageable host downgrades to
        # sync, so reading the result immediately (no synchronize) is correct.
        src = torch.randn(256, 256)
        dev = src.to(device)
        host = dev.to("cpu", non_blocking=True)
        self.assertEqual(host, src)

    def test_pageable_non_blocking_h2d(self, device):
        src = torch.randn(128, 128)
        dev = src.to(device, non_blocking=True)
        self.assertEqual(dev.cpu(), src)


instantiate_device_type_tests(TestPinMemoryCopy, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
