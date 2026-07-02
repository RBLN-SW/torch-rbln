# Owner(s): ["module: PrivateUse1"]

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestForeachOptimizerOps(TestCase):
    rbln_device = torch.device("rbln:0")

    def _reset_graph_counter(self):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 0)

    def test_foreach_optimizer_ops_do_not_compile_per_tensor_rbln_graphs(self):
        params = [
            torch.ones((8, 64), dtype=torch.bfloat16, device=self.rbln_device),
            torch.ones((64, 8), dtype=torch.bfloat16, device=self.rbln_device),
        ]

        self._reset_graph_counter()
        torch._foreach_mul_(params, 0.999)

        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 0)

    def test_foreach_lerp_does_not_mutate_readonly_tensor_list(self):
        grads = [
            torch.randn((8, 64), dtype=torch.bfloat16, device=self.rbln_device),
            torch.randn((64, 8), dtype=torch.bfloat16, device=self.rbln_device),
        ]
        exp_avgs = [torch.zeros_like(g) for g in grads]
        grads_before = [g.detach().cpu().clone() for g in grads]

        torch._foreach_lerp_(exp_avgs, grads, 0.1)

        for grad, before in zip(grads, grads_before):
            self.assertEqual(grad.cpu(), before)

    def test_adamw_rank8_zero_grad_stays_finite(self):
        param = torch.nn.Parameter(torch.randn((8, 64), dtype=torch.bfloat16, device=self.rbln_device))
        param.grad = torch.zeros_like(param)
        optimizer = torch.optim.AdamW([param], lr=1e-5, foreach=True)

        optimizer.step()

        self.assertTrue(bool(torch.isfinite(param.detach().cpu()).all().item()))


if __name__ == "__main__":
    run_tests()
