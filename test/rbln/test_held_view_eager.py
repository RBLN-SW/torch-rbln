# Owner(s): ["module: PrivateUse1"]

import os
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln._internal.compile_cache import clear_rbln_compile_cache


def _pv_hash(tensor):
    from rebel._C import vmem  # noqa: PLC0415

    return vmem.get_pv_hash(tensor.data_ptr())


@pytest.mark.npu(count=1)
class TestHeldViewEager(TestCase):
    def setUp(self):
        super().setUp()
        clear_rbln_compile_cache()
        torch._dynamo.reset()

    def tearDown(self):
        clear_rbln_compile_cache()
        torch._dynamo.reset()
        super().tearDown()

    @parametrize("flat_first", [True, False])
    def test_transpose_keeps_placed_input(self, device, flat_first):
        # The compiler takes a cut only when every chiplet holds a piece.
        chiplets = torch.rbln.get_device_properties(device).num_chiplet
        if chiplets < 2:
            self.skipTest("requires two chiplets")
        shape = (2 * chiplets, 8, 64, 128)
        data = torch.randn(shape).to(torch.bfloat16)
        expected = data.transpose(1, 2).contiguous()
        with mock.patch.dict(os.environ):
            os.environ.pop("RBLN_DECLARE_HELD_PV", None)
            flat = data.to(device)
            placed = torch.empty(shape, dtype=torch.bfloat16, device=device)
            torch.rbln.set_device_alloc(
                placed, [torch.rbln.DevicePlacement(0, c, ((0, 2 * c, 2 * c + 2),)) for c in range(chiplets)]
            )
            placed.copy_(data)
            placed_hash = _pv_hash(placed)
            self.assertIsNotNone(placed_hash)

            order = (flat, placed) if flat_first else (placed, flat)
            for _ in range(2):
                for tensor in order:
                    self.assertEqual(tensor.transpose(1, 2).contiguous().cpu(), expected)

            self.assertEqual(_pv_hash(placed), placed_hash)


instantiate_device_type_tests(TestHeldViewEager, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
