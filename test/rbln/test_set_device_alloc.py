# Owner(s): ["module: PrivateUse1"]

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

import torch_rbln  # noqa: F401


class TestPlacementInput(TestCase):
    def test_cpu_tensor_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "tensor must be an RBLN tensor"):
            torch.rbln.set_device_alloc(torch.empty(4, 8), [torch.rbln.DevicePlacement(0, 0)])


@pytest.mark.npu(count=1)
class TestSetDeviceAlloc(TestCase):
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("num_chiplets", [1, 2])
    def test_round_trip(self, device, dtype, num_chiplets):
        if torch.rbln.get_device_properties(device).num_chiplet < num_chiplets:
            self.skipTest("requires two chiplets")
        data = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8).to(dtype)
        tensor = torch.empty(data.shape, dtype=dtype, device=device)
        rows = 4 // num_chiplets
        placements = [
            torch.rbln.DevicePlacement(0, c, ((1, c * rows, (c + 1) * rows),)) for c in reversed(range(num_chiplets))
        ]
        torch.rbln.set_device_alloc(tensor, placements)
        # A lazy tensor cannot be used as a layout reference. This also checks
        # that set_device_alloc materialized the physical view before any copy.
        target = torch.empty_like(tensor)
        torch.rbln.set_device_layout_like(target, tensor)
        tensor.copy_(data)
        target.copy_(tensor)
        self.assertEqual(target.cpu(), data)
        self.assertEqual(tensor.cpu(), data)

    def test_physical_dtype(self, device):
        data = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        tensor = torch.empty(data.shape, dtype=torch.float32, device=device)
        torch.rbln.set_device_alloc(tensor, [torch.rbln.DevicePlacement(0, 0)], physical_dtype=torch.float16)
        tensor.copy_(data)
        self.assertEqual(tensor.cpu(), data)

    @parametrize("view_kind", ["offset", "narrow", "transpose"])
    def test_partial_or_noncontiguous_storage_rejected(self, device, view_kind):
        tensor = torch.empty((4, 8), device=device)
        if view_kind == "offset":
            view = tensor[1:]
        elif view_kind == "narrow":
            view = tensor[:2]
        else:
            view = tensor.t()
        with self.assertRaisesRegex(RuntimeError, "whole storage"):
            torch.rbln.set_device_alloc(view, [torch.rbln.DevicePlacement(0, 0)])

    def test_empty_tensor_rejected(self, device):
        with self.assertRaisesRegex(RuntimeError, "tensor must not be empty"):
            torch.rbln.set_device_alloc(torch.empty(0, device=device), [torch.rbln.DevicePlacement(0, 0)])

    @parametrize("invalid", ["empty", "gap", "overlap", "axis", "duplicate"])
    def test_runtime_rejection_preserves_data(self, device, invalid):
        place = torch.rbln.DevicePlacement
        placements = {
            "empty": [],
            "gap": [place(0, 0, ((0, 0, 2),))],
            "overlap": [place(0, 0, ((0, 0, 3),)), place(0, 1, ((0, 2, 4),))],
            "axis": [place(0, 0, ((2, 0, 1),))],
            "duplicate": [place(0, 0, ((0, 0, 2),)), place(0, 0, ((0, 2, 4),))],
        }[invalid]
        data = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        tensor = torch.empty(data.shape, dtype=data.dtype, device=device)
        torch.rbln.set_device_alloc(tensor, [place(0, 0)])
        tensor.copy_(data)
        with self.assertRaisesRegex(RuntimeError, "rbln_set_device_alloc failed"):
            torch.rbln.set_device_alloc(tensor, placements)
        self.assertEqual(tensor.cpu(), data)


instantiate_device_type_tests(TestSetDeviceAlloc, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
