# Owner(s): ["module: PrivateUse1"]
"""The graphs torch.compile makes of one module hold its weights once on a device: a graph that
holds a weight as another graph does binds that graph's tensor of it instead of writing its own.
"""

import rbln
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


IN, HIDDEN = 1024, 4096


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.Linear(IN, HIDDEN)
        self.down = torch.nn.Linear(HIDDEN, IN)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _free() -> int:
    return rbln.Device(torch.rbln.current_device()).memory_info().free


class TestGraphWeightSharing(TestCase):
    def test_graphs_of_one_module_hold_its_weights_once(self):
        model = Mlp().eval()
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        on_device = Mlp().eval().to("rbln")
        compiled = torch.compile(on_device, backend="rbln", dynamic=False)
        with torch.no_grad():
            first = torch.randn(4, IN)
            self.assertEqual(compiled(first.to("rbln")).cpu(), model(first), atol=5e-2, rtol=5e-2)
            free = _free()
            second = torch.randn(8, IN)
            self.assertEqual(compiled(second.to("rbln")).cpu(), model(second), atol=5e-2, rtol=5e-2)
        self.assertLess(free - _free(), weights // 2)


if __name__ == "__main__":
    run_tests()
