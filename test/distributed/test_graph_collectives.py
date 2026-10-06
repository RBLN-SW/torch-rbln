# Owner(s): ["module: PrivateUse1"]
"""A graph torch.compile hands the rbln backend runs its collectives over the communicator of the
process group they name, the default group's or one made after it, as vLLM's tensor parallel
layers reduce their partial sums inside the model graph.
"""

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase

from test.utils import configure_master_port_for_rccl_tests, spawn_target_with_clean_exit


FEATURES, UNITS = 512, 256


class RowParallel(torch.nn.Module):
    """A linear whose input features are split across the group; each rank sums the parts."""

    def __init__(self, weight: torch.Tensor, group):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.group = group

    def forward(self, x):
        y = x @ self.weight.T
        dist.all_reduce(y, group=self.group)
        return y


def run_row_parallel(rank: int, world_size: int, sub_group: bool) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.rbln.set_device(rank)
    dist.init_process_group(backend="rbln-ccl", rank=rank, world_size=world_size)
    try:
        group = dist.new_group(list(range(world_size)), backend="rbln-ccl") if sub_group else None
        torch.manual_seed(0)
        weight = (torch.randn(UNITS, FEATURES) / 16).to(torch.float16)
        x = torch.randn(1, 64, FEATURES).to(torch.float16)
        part = slice(rank * FEATURES // world_size, (rank + 1) * FEATURES // world_size)
        model = RowParallel(weight[:, part].contiguous(), group).to(f"rbln:{rank}")
        compiled = torch.compile(model, backend="rbln", dynamic=False)
        out = compiled(x[..., part].contiguous().to(f"rbln:{rank}")).cpu().float()
        ref = x.float() @ weight.float().T
        cosine = float(out.flatten() @ ref.flatten() / out.norm() / ref.norm())
        assert cosine > 0.999, f"rank {rank}: cosine {cosine}"
    finally:
        dist.destroy_process_group()


@pytest.mark.single_worker
@pytest.mark.test_set_ci
class TestGraphCollectives(TestCase):
    def setUp(self):
        env = mock.patch.dict(
            os.environ, {"RBLN_ROOT_IP": "127.0.0.1", "RBLN_LOCAL_IP": "127.0.0.1", "MASTER_ADDR": "127.0.0.1"}
        )
        env.start()
        self.addCleanup(env.stop)
        configure_master_port_for_rccl_tests()
        self.world_size = min(torch.rbln.device_count(), 2)

    @parametrize("sub_group", [False, True])
    def test_a_graph_reduces_over_its_group(self, sub_group):
        if self.world_size < 2:
            self.skipTest("Requires two NPUs")
        mp.spawn(
            spawn_target_with_clean_exit,
            args=(run_row_parallel, self.world_size, sub_group),
            nprocs=self.world_size,
            join=True,
        )


instantiate_parametrized_tests(TestGraphCollectives)


if __name__ == "__main__":
    run_tests()
