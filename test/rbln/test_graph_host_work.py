# Owner(s): ["module: PrivateUse1"]
"""Host work around a graph's device work runs on each call: what the host makes of the inputs and
state before the run, such as an embedding gather, and what it makes of the run's results after it.
"""

import copy

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class Embedding(torch.nn.Module):
    """Gathers rows of a table the host holds, which a linear layer on the device reads."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(32, 64)
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, ids):
        return self.linear(self.emb(ids))


class Tangent(torch.nn.Module):
    """A linear layer on the device, whose result the host takes the tangent of."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        return torch.tan(self.linear(x) * 0.3)


@pytest.mark.test_set_ci
class TestGraphHostWork(TestCase):
    def test_host_work_before_the_run_reads_the_input_and_the_state(self):
        torch._dynamo.reset()
        model = Embedding().eval()
        reference = copy.deepcopy(model)
        compiled = torch.compile(model.to("rbln"), backend="rbln", dynamic=False)
        with torch.no_grad():
            for seed in range(3):
                ids = torch.randint(0, 32, (1, 4), generator=torch.Generator().manual_seed(seed))
                want = reference(ids)
                self.assertEqual(compiled(ids.to("rbln")).cpu(), want, atol=2e-2, rtol=2e-2)

    def test_host_work_before_the_run_follows_the_state(self):
        torch._dynamo.reset()
        model = Embedding().eval()
        reference = copy.deepcopy(model)
        compiled = torch.compile(model.to("rbln"), backend="rbln", dynamic=False)
        ids = torch.tensor([[3, 1, 4, 1]])
        with torch.no_grad():
            compiled(ids.to("rbln"))
            model.emb.weight.mul_(-1)
            reference.emb.weight.mul_(-1)
            self.assertEqual(compiled(ids.to("rbln")).cpu(), reference(ids), atol=2e-2, rtol=2e-2)

    def test_host_work_after_the_run_makes_the_result(self):
        torch._dynamo.reset()
        model = Tangent().eval()
        reference = copy.deepcopy(model)
        compiled = torch.compile(model.to("rbln"), backend="rbln", dynamic=False)
        with torch.no_grad():
            for seed in range(3):
                x = torch.randn(4, 64, generator=torch.Generator().manual_seed(seed))
                want = reference(x)
                got = compiled(x.to("rbln"))
                self.assertEqual(got.device.type, "rbln")
                self.assertEqual(got.cpu(), want, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
