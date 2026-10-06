# Owner(s): ["module: PrivateUse1"]
"""State a graph reads is written to the device again once it changes, and state the program was
built for is checked instead: a new tensor of the same value runs, another value is refused.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class Gather(torch.nn.Module):
    """Gathers rows of its weight by an index it holds, which the program is built for."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn(3, 64))
        self.register_buffer("index", torch.tensor([2, 0, 1]))

    def forward(self, x):
        return x + self.weight[self.index]


@pytest.mark.test_set_ci
class TestGraphState(TestCase):
    def test_state_the_program_is_built_for_is_checked_when_it_changes(self):
        torch._dynamo.reset()
        model = Gather().to("rbln")
        compiled = torch.compile(model, backend="rbln", dynamic=False)
        x = torch.randn(3, 64)
        want = x + model.weight.detach().cpu()[[2, 0, 1]]
        with torch.no_grad():
            self.assertEqual(compiled(x.to("rbln")).cpu(), want, atol=2e-2, rtol=2e-2)
            model.index = torch.tensor([2, 0, 1], device="rbln")
            self.assertEqual(compiled(x.to("rbln")).cpu(), want, atol=2e-2, rtol=2e-2)
            model.weight.mul_(-1)
            self.assertEqual(compiled(x.to("rbln")).cpu(), x - want + x, atol=2e-2, rtol=2e-2)
            model.index = torch.tensor([0, 1, 2], device="rbln")
            with self.assertRaisesRegex(Exception, "differs from the value"):
                compiled(x.to("rbln"))


if __name__ == "__main__":
    run_tests()
