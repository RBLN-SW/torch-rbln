# Owner(s): ["module: PrivateUse1"]

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, subtest, TestCase

from test.utils import configure_master_port_for_rccl_tests, set_deterministic_seeds


def setup_environment(rank: int, world_size: int) -> None:
    """Setup environment variables for distributed testing."""
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.rbln.set_device(rank)


def run_default_group_allreduce_test(rank: int, world_size: int, backend: str) -> None:
    """Test default group allreduce with tensor parallel linear."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:

        class TensorParallelLinear(nn.Module):
            def __init__(self, input_size, output_size, world_size, rank):
                super().__init__()
                self.input_size = input_size
                self.output_size = output_size
                self.world_size = world_size
                self.rank = rank

                self.linear = nn.Linear(input_size, output_size, bias=False)

                self.start_col = rank * (output_size // world_size)
                self.end_col = (rank + 1) * (output_size // world_size)

                with torch.no_grad():
                    mask = torch.zeros_like(self.linear.weight)
                    mask[self.start_col : self.end_col, :] = 1.0
                    self.linear.weight.data *= mask

            def forward(self, x):
                output = self.linear(x)
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
                return output

        class SequentialLinear(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.linear = nn.Linear(input_size, output_size, bias=False)

            def forward(self, x):
                return self.linear(x)

        if world_size != 2:
            return

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        input_size = 1024
        output_size = 256
        batch_size = 64

        tp_model = TensorParallelLinear(input_size, output_size, world_size, rank).to(device).half()

        if rank == 0:
            seq_model = SequentialLinear(input_size, output_size).to(device).half()
            tp_weight = tp_model.linear.weight.data.clone()
            dist.all_reduce(tp_weight, op=dist.ReduceOp.SUM)
            seq_model.linear.weight.data = tp_weight
        else:
            tp_weight = tp_model.linear.weight.data.clone()
            dist.all_reduce(tp_weight, op=dist.ReduceOp.SUM)

        set_deterministic_seeds(123)
        input_data = torch.randn(batch_size, input_size, device=device, dtype=torch.float16)

        tp_output = tp_model(input_data)

        if rank == 0:
            seq_output = seq_model(input_data)
            diff = torch.abs(tp_output - seq_output)
            max_diff = torch.max(diff).item()
            assert max_diff < 1e-3, f"Results don't match. Max difference: {max_diff}"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_pp2_test(rank: int, world_size: int, backend: str) -> None:
    """Test PP=2 pipeline parallel example."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        DTYPE = torch.float16

        class SequentialModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layer1 = nn.Linear(input_size, hidden_size, bias=False, dtype=DTYPE)
                self.layer2 = nn.Linear(hidden_size, output_size, bias=False, dtype=DTYPE)

            def forward(self, x):
                x = self.layer1(x)
                x = torch.relu(x)
                x = self.layer2(x)
                return x

        class PipelineStage(nn.Module):
            def __init__(self, stage_id, input_size, hidden_size, output_size):
                super().__init__()
                self.stage_id = stage_id

                if stage_id == 0:
                    self.layer = nn.Linear(input_size, hidden_size, bias=False, dtype=DTYPE)
                    self.has_activation = True
                elif stage_id == 1:
                    self.layer = nn.Linear(hidden_size, output_size, bias=False, dtype=DTYPE)
                    self.has_activation = False
                else:
                    raise ValueError(f"Invalid stage_id: {stage_id}")

            def forward(self, x):
                x = self.layer(x)
                if self.has_activation:
                    x = torch.relu(x)
                return x

        class PipelineParallelModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, rank, world_size):
                super().__init__()
                self.rank = rank
                self.world_size = world_size

                if rank == 0:
                    self.stage = PipelineStage(0, input_size, hidden_size, output_size)
                    self.next_rank = 1
                    self.prev_rank = None
                elif rank == 1:
                    self.stage = PipelineStage(1, input_size, hidden_size, output_size)
                    self.next_rank = None
                    self.prev_rank = 0
                else:
                    raise ValueError(f"Invalid rank for PP=2: {rank}")

            def forward(self, x):
                if self.rank == 0:
                    output = self.stage(x)
                    if self.next_rank is not None:
                        dist.send(output, dst=self.next_rank)
                    return output
                elif self.rank == 1:
                    if self.prev_rank is not None:
                        batch_size = x.shape[0]
                        hidden_size = self.stage.layer.in_features
                        input_shape = (batch_size, hidden_size)
                        received_tensor = torch.zeros(input_shape, dtype=DTYPE, device=x.device)
                        dist.recv(received_tensor, src=self.prev_rank)
                        output = self.stage(received_tensor)
                    else:
                        output = self.stage(x)
                    return output

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        assert world_size == 2, f"This example requires exactly 2 ranks, got {world_size}"

        input_size = 128
        hidden_size = 256
        output_size = 64
        batch_size = 2
        device = torch.device("rbln", rank)

        set_deterministic_seeds(42)
        input_tensor = torch.randn(batch_size, input_size, dtype=DTYPE, device=device)

        if rank == 0:
            sequential_model = SequentialModel(input_size, hidden_size, output_size).to(device)
            with torch.no_grad():
                nn.init.xavier_uniform_(sequential_model.layer1.weight)
                nn.init.xavier_uniform_(sequential_model.layer2.weight)
            with torch.no_grad():
                sequential_output = sequential_model(input_tensor)
        else:
            sequential_model = SequentialModel(input_size, hidden_size, output_size).to(device)

        pp_model = PipelineParallelModel(input_size, hidden_size, output_size, rank, world_size).to(device)

        for name, param in sequential_model.named_parameters():
            dist.broadcast(param.data, src=0)

        if rank == 0:
            pp_model.stage.layer.weight.data.copy_(sequential_model.layer1.weight.data)
        elif rank == 1:
            pp_model.stage.layer.weight.data.copy_(sequential_model.layer2.weight.data)

        pp_output = pp_model(input_tensor)

        if rank == 0:
            pp_output_received = torch.empty(sequential_output.shape, device=device, dtype=DTYPE)
            dist.recv(pp_output_received, src=1)
            is_same_relaxed = torch.allclose(pp_output_received, sequential_output, atol=1e-2)
            assert is_same_relaxed, "PP model output does not match Sequential model output"
        else:
            dist.send(pp_output, dst=0)

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_single_linear_allgather_default_group_test(rank: int, world_size: int, backend: str) -> None:
    """Test single linear layer with allgather, default group."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:

        def tensor_to_view_list(tensor):
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return [tensor[i] for i in range(tensor.shape[0])]

        DTYPE = torch.float16

        IN_DIM_0 = 1
        IN_DIM_1 = 128
        OUT_DIM_1 = 128

        assert world_size == 2, f"This test requires exactly 2 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.out_features_per_partition = out_features // world_size
                self.linear = nn.Linear(in_features, self.out_features_per_partition, dtype=DTYPE)

            def forward(self, x):
                src_rank_in_group = dist.get_global_rank(self.group, 0)
                dist.broadcast(x, src=src_rank_in_group, group=self.group)
                partial_output = self.linear(x)

                if not partial_output.is_contiguous():
                    partial_output = partial_output.contiguous()

                gathered_output_shape = (self.world_size,) + partial_output.shape
                t = torch.empty(gathered_output_shape, device=device, dtype=DTYPE)
                gathered_outputs = tensor_to_view_list(t)

                dist.all_gather(gathered_outputs, partial_output, group=self.group)
                all_outputs = torch.cat(gathered_outputs, dim=1)
                return all_outputs

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(IN_DIM_1, OUT_DIM_1).to(device=device)
            data = torch.randn((IN_DIM_0, IN_DIM_1), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [
                TensorParallelLinear(IN_DIM_1, OUT_DIM_1, tensor_world_size, rank, tensor_group),
            ]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            l1_bias = sequential_model.layers[0].bias.data.clone()
            split_size = l1_weight.size(0) // tensor_world_size

            l1_weight_split_0 = l1_weight[0:split_size, :]
            l1_bias_split_0 = l1_bias[0:split_size]
            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)
            model.layers[0].linear.bias.data.copy_(l1_bias_split_0)

            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            l1_bias_split_1 = l1_bias[split_size : 2 * split_size]

            dist.send(l1_weight_split_1, dst=1, group=tensor_groups[0])
            dist.send(l1_bias_split_1, dst=1, group=tensor_groups[0])

        elif rank == 1:
            l1_weight = torch.empty((OUT_DIM_1 // 2, IN_DIM_1), device=device, dtype=DTYPE)
            l1_bias = torch.empty(OUT_DIM_1 // 2, device=device, dtype=DTYPE)

            dist.recv(l1_weight, src=0, group=tensor_groups[0])
            dist.recv(l1_bias, src=0, group=tensor_groups[0])

            model.layers[0].linear.weight.data.copy_(l1_weight)
            model.layers[0].linear.bias.data.copy_(l1_bias)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)
        elif rank == 1:
            data_dummy = torch.empty((IN_DIM_0, IN_DIM_1), device=device, dtype=DTYPE)
            output_tp_pp = model(data_dummy)

        if rank == 0:
            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_two_linear_bias_default_group_send_recv_test(rank: int, world_size: int, backend: str) -> None:
    """Test two linear layers with bias, default group, send/recv."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        DTYPE = torch.float16

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group, is_output_layer=False):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.is_output_layer = is_output_layer

                if not is_output_layer:
                    self.out_features_per_partition = out_features // world_size
                    self.linear = nn.Linear(in_features, self.out_features_per_partition, dtype=DTYPE)
                else:
                    self.in_features_per_partition = in_features // world_size
                    self.linear = nn.Linear(self.in_features_per_partition, out_features, dtype=DTYPE)

            def forward(self, x):
                if not self.is_output_layer:
                    src_rank_in_group = dist.get_global_rank(self.group, 0)
                    dist.broadcast(x, src=src_rank_in_group, group=self.group)
                    output = self.linear(x)
                    return output
                else:
                    weight_output = torch.nn.functional.linear(x, self.linear.weight, bias=None)
                    dist.all_reduce(weight_output, op=dist.ReduceOp.SUM, group=self.group)
                    output = weight_output + self.linear.bias
                    return output

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, dtype=DTYPE),
                    nn.Linear(hidden_dim_2, hidden_dim_3, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        assert world_size == 4, f"This test requires exactly 4 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(512, 2048, 256).to(device=device)
            data = torch.randn((64, 512), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [TensorParallelLinear(512, 2048, tensor_world_size, rank, tensor_group)]
            model = PipelineStage(layers, rank).to(device=device)
        else:
            layers = [TensorParallelLinear(2048, 256, tensor_world_size, rank, tensor_group, True)]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            l1_bias = sequential_model.layers[0].bias.data.clone()
            split_size = l1_weight.size(0) // tensor_world_size

            l1_weight_split_0 = l1_weight[0:split_size, :]
            if not l1_weight_split_0.is_contiguous():
                l1_weight_split_0 = l1_weight_split_0.contiguous()
            l1_bias_split_0 = l1_bias[0:split_size]
            if not l1_bias_split_0.is_contiguous():
                l1_bias_split_0 = l1_bias_split_0.contiguous()
            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)
            model.layers[0].linear.bias.data.copy_(l1_bias_split_0)

            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            if not l1_weight_split_1.is_contiguous():
                l1_weight_split_1 = l1_weight_split_1.contiguous()
            l1_bias_split_1 = l1_bias[split_size : 2 * split_size]
            if not l1_bias_split_1.is_contiguous():
                l1_bias_split_1 = l1_bias_split_1.contiguous()

            dist.send(l1_weight_split_1, dst=1)
            dist.send(l1_bias_split_1, dst=1)

            l2_weight = sequential_model.layers[1].weight.data.clone()
            l2_bias = sequential_model.layers[1].bias.data.clone()
            split_size = l2_weight.size(1) // tensor_world_size

            l2_weight_split_0 = l2_weight[:, 0:split_size]
            if not l2_weight_split_0.is_contiguous():
                l2_weight_split_0 = l2_weight_split_0.contiguous()
            dist.send(l2_weight_split_0, dst=2)
            dist.send(l2_bias, dst=2)

            l2_weight_split_1 = l2_weight[:, split_size : 2 * split_size]
            if not l2_weight_split_1.is_contiguous():
                l2_weight_split_1 = l2_weight_split_1.contiguous()
            dist.send(l2_weight_split_1, dst=3)
            dist.send(l2_bias, dst=3)

        elif rank == 1:
            l1_weight = torch.empty((1024, 512), device=device, dtype=DTYPE)
            l1_bias = torch.empty(1024, device=device, dtype=DTYPE)
            dist.recv(l1_weight, src=0)
            dist.recv(l1_bias, src=0)
            model.layers[0].linear.weight.data.copy_(l1_weight)
            model.layers[0].linear.bias.data.copy_(l1_bias)

        elif rank == 2:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)
            dist.recv(l2_weight, src=0)
            dist.recv(l2_bias, src=0)
            model.layers[0].linear.weight.data.copy_(l2_weight)
            model.layers[0].linear.bias.data.copy_(l2_bias)

        elif rank == 3:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)
            dist.recv(l2_weight, src=0)
            dist.recv(l2_bias, src=0)
            model.layers[0].linear.weight.data.copy_(l2_weight)
            model.layers[0].linear.bias.data.copy_(l2_bias)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)
            dist.send(output_tp_pp, dst=2)
            dist.send(output_seq, dst=2)
        elif rank == 1:
            data_dummy = torch.empty((64, 512), device=device, dtype=DTYPE)
            output_tp_pp = model(data_dummy)
            dist.send(output_tp_pp, dst=3)
        elif rank == 2:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)
            dist.recv(output_tp_pp_recv, src=0)
            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)
        elif rank == 3:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)
            dist.recv(output_tp_pp_recv, src=1)
            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)

        if rank == 2:
            output_seq = torch.empty((64, 256), device=device, dtype=DTYPE)
            dist.recv(output_seq, src=0)

            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_two_linear_no_bias_default_group_send_recv_test(rank: int, world_size: int, backend: str) -> None:
    """Test two linear layers without bias, default group, send/recv."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        DTYPE = torch.float16

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group, is_output_layer=False):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.is_output_layer = is_output_layer

                if not is_output_layer:
                    self.out_features_per_partition = out_features // world_size
                    self.linear = nn.Linear(in_features, self.out_features_per_partition, bias=False, dtype=DTYPE)
                else:
                    self.in_features_per_partition = in_features // world_size
                    self.linear = nn.Linear(self.in_features_per_partition, out_features, bias=False, dtype=DTYPE)

            def forward(self, x):
                if not self.is_output_layer:
                    src_rank_in_group = dist.get_global_rank(self.group, 0)
                    dist.broadcast(x, src=src_rank_in_group, group=self.group)
                    output = self.linear(x)
                    return output
                else:
                    weight_output = torch.nn.functional.linear(x, self.linear.weight, bias=None)
                    dist.all_reduce(weight_output, op=dist.ReduceOp.SUM, group=self.group)
                    return weight_output

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, bias=False, dtype=DTYPE),
                    nn.Linear(hidden_dim_2, hidden_dim_3, bias=False, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        assert world_size == 4, f"This test requires exactly 4 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(512, 2048, 256).to(device=device)
            data = torch.randn((64, 512), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [TensorParallelLinear(512, 2048, tensor_world_size, rank, tensor_group)]
            model = PipelineStage(layers, rank).to(device=device)
        else:
            layers = [TensorParallelLinear(2048, 256, tensor_world_size, rank, tensor_group, True)]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            split_size = l1_weight.size(0) // tensor_world_size

            l1_weight_split_0 = l1_weight[0:split_size, :]
            if not l1_weight_split_0.is_contiguous():
                l1_weight_split_0 = l1_weight_split_0.contiguous()
            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)

            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            if not l1_weight_split_1.is_contiguous():
                l1_weight_split_1 = l1_weight_split_1.contiguous()
            dist.send(l1_weight_split_1, dst=1)

            l2_weight = sequential_model.layers[1].weight.data.clone()
            split_size = l2_weight.size(1) // tensor_world_size

            l2_weight_split_0 = l2_weight[:, 0:split_size]
            if not l2_weight_split_0.is_contiguous():
                l2_weight_split_0 = l2_weight_split_0.contiguous()
            dist.send(l2_weight_split_0, dst=2)

            l2_weight_split_1 = l2_weight[:, split_size : 2 * split_size]
            if not l2_weight_split_1.is_contiguous():
                l2_weight_split_1 = l2_weight_split_1.contiguous()
            dist.send(l2_weight_split_1, dst=3)

        elif rank == 1:
            l1_weight = torch.empty((1024, 512), device=device, dtype=DTYPE)
            dist.recv(l1_weight, src=0)
            model.layers[0].linear.weight.data.copy_(l1_weight)

        elif rank == 2:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            dist.recv(l2_weight, src=0)
            model.layers[0].linear.weight.data.copy_(l2_weight)

        elif rank == 3:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            dist.recv(l2_weight, src=0)
            model.layers[0].linear.weight.data.copy_(l2_weight)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)
            dist.send(output_tp_pp, dst=2)
            dist.send(output_seq, dst=2)
        elif rank == 1:
            data_dummy = torch.empty((64, 512), device=device, dtype=DTYPE)
            output_tp_pp = model(data_dummy)
            dist.send(output_tp_pp, dst=3)
        elif rank == 2:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)
            dist.recv(output_tp_pp_recv, src=0)
            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)
        elif rank == 3:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)
            dist.recv(output_tp_pp_recv, src=1)
            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)

        if rank == 2:
            output_seq = torch.empty((64, 256), device=device, dtype=DTYPE)
            dist.recv(output_seq, src=0)

            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_two_linear_bias_subgroup_send_recv_test(rank: int, world_size: int, backend: str) -> None:
    """Test two linear layers with bias, subgroup, send/recv."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        DTYPE = torch.float16

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group, is_output_layer=False):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.is_output_layer = is_output_layer

                if not is_output_layer:
                    self.out_features_per_partition = out_features // world_size
                    self.linear = nn.Linear(in_features, self.out_features_per_partition, dtype=DTYPE)
                else:
                    self.in_features_per_partition = in_features // world_size
                    self.linear = nn.Linear(self.in_features_per_partition, out_features, dtype=DTYPE)

            def forward(self, x):
                if not self.is_output_layer:
                    src_rank_in_group = dist.get_global_rank(self.group, 0)
                    dist.broadcast(x, src=src_rank_in_group, group=self.group)
                    output = self.linear(x)
                    return output
                else:
                    weight_output = torch.nn.functional.linear(x, self.linear.weight, bias=None)
                    dist.all_reduce(weight_output, op=dist.ReduceOp.SUM, group=self.group)
                    output = weight_output + self.linear.bias
                    return output

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, dtype=DTYPE),
                    nn.Linear(hidden_dim_2, hidden_dim_3, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        assert world_size == 4, f"This test requires exactly 4 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        pipe_groups = [dist.new_group(ranks=[i, i + tensor_world_size]) for i in range(tensor_world_size)]
        pipe_group = pipe_groups[rank % tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(512, 2048, 256).to(device=device)
            data = torch.randn((64, 512), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [TensorParallelLinear(512, 2048, tensor_world_size, rank, tensor_group)]
            model = PipelineStage(layers, rank).to(device=device)
        else:
            layers = [TensorParallelLinear(2048, 256, tensor_world_size, rank, tensor_group, True)]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            l1_bias = sequential_model.layers[0].bias.data.clone()

            # Use tensor slicing for manual weight splitting
            split_size = l1_weight.size(0) // tensor_world_size

            # Slice for rank 0 (first half)
            l1_weight_split_0 = l1_weight[0:split_size, :]
            if not l1_weight_split_0.is_contiguous():
                l1_weight_split_0 = l1_weight_split_0.contiguous()
            l1_bias_split_0 = l1_bias[0:split_size]
            if not l1_bias_split_0.is_contiguous():
                l1_bias_split_0 = l1_bias_split_0.contiguous()
            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)
            model.layers[0].linear.bias.data.copy_(l1_bias_split_0)

            # Slice for rank 1 (second half)
            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            if not l1_weight_split_1.is_contiguous():
                l1_weight_split_1 = l1_weight_split_1.contiguous()
            l1_bias_split_1 = l1_bias[split_size : 2 * split_size]
            if not l1_bias_split_1.is_contiguous():
                l1_bias_split_1 = l1_bias_split_1.contiguous()

            dist.send(l1_weight_split_1, group_dst=1, group=tensor_group)
            dist.send(l1_bias_split_1, group_dst=1, group=tensor_group)

            l2_weight = sequential_model.layers[1].weight.data.clone()
            l2_bias = sequential_model.layers[1].bias.data.clone()

            # Use tensor slicing for manual weight splitting (dim=1 for column-wise split)
            split_size = l2_weight.size(1) // tensor_world_size

            # Slice for rank 2 (first half of columns)
            l2_weight_split_0 = l2_weight[:, 0:split_size]
            if not l2_weight_split_0.is_contiguous():
                l2_weight_split_0 = l2_weight_split_0.contiguous()

            dist.send(l2_weight_split_0, group_dst=1, group=pipe_group)
            dist.send(l2_bias, group_dst=1, group=pipe_group)

            # Slice for rank 3 (second half of columns)
            l2_weight_split_1 = l2_weight[:, split_size : 2 * split_size]
            if not l2_weight_split_1.is_contiguous():
                l2_weight_split_1 = l2_weight_split_1.contiguous()

            dist.send(l2_weight_split_1, dst=3)
            dist.send(l2_bias, dst=3)

        elif rank == 1:
            l1_weight = torch.empty((1024, 512), device=device, dtype=DTYPE)
            l1_bias = torch.empty(1024, device=device, dtype=DTYPE)

            dist.recv(l1_weight, group_src=0, group=tensor_group)
            dist.recv(l1_bias, group_src=0, group=tensor_group)

            model.layers[0].linear.weight.data.copy_(l1_weight)
            model.layers[0].linear.bias.data.copy_(l1_bias)

        elif rank == 2:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, group_src=0, group=pipe_group)
            dist.recv(l2_bias, group_src=0, group=pipe_group)

            model.layers[0].linear.weight.data.copy_(l2_weight)
            model.layers[0].linear.bias.data.copy_(l2_bias)

        elif rank == 3:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, src=0)
            dist.recv(l2_bias, src=0)

            model.layers[0].linear.weight.data.copy_(l2_weight)
            model.layers[0].linear.bias.data.copy_(l2_bias)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)

            # Use pipe group for rank 0 -> rank 2 communication
            dist.send(output_tp_pp, group_dst=1, group=pipe_group)
            dist.send(output_seq, group_dst=1, group=pipe_group)
        elif rank == 1:
            data_dummy = torch.empty((64, 512), device=device, dtype=DTYPE)

            output_tp_pp = model(data_dummy)

            # Use pipe group for rank 1 -> rank 3 communication
            dist.send(output_tp_pp, group_dst=1, group=pipe_group)
        elif rank == 2:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_tp_pp_recv, group_src=0, group=pipe_group)

            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)
        elif rank == 3:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)

            # Use pipe group for rank 3 <- rank 1 communication
            dist.recv(output_tp_pp_recv, group_src=0, group=pipe_group)

            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)

        if rank == 2:
            output_seq = torch.empty((64, 256), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_seq, group_src=0, group=pipe_group)

            # Move to CPU for comparison to avoid RBLN compilation of allclose
            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()

            # Check with different tolerance levels
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_two_linear_bias_relu_subgroup_send_recv_test(rank: int, world_size: int, backend: str) -> None:
    """Test two linear layers with bias and ReLU, subgroup, send/recv."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        DTYPE = torch.float16

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group, is_output_layer=False):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.is_output_layer = is_output_layer

                if not is_output_layer:
                    self.out_features_per_partition = out_features // world_size
                    self.linear = nn.Linear(in_features, self.out_features_per_partition, dtype=DTYPE)
                else:
                    self.in_features_per_partition = in_features // world_size
                    self.linear = nn.Linear(self.in_features_per_partition, out_features, dtype=DTYPE)

            def forward(self, x):
                if not self.is_output_layer:
                    src_rank_in_group = dist.get_global_rank(self.group, 0)
                    dist.broadcast(x, src=src_rank_in_group, group=self.group)
                    output = self.linear(x)
                    return output
                else:
                    weight_output = torch.nn.functional.linear(x, self.linear.weight, bias=None)
                    dist.all_reduce(weight_output, op=dist.ReduceOp.SUM, group=self.group)
                    output = weight_output + self.linear.bias
                    return output

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, dtype=DTYPE),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_2, hidden_dim_3, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        assert world_size == 4, f"This test requires exactly 4 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        pipe_groups = [dist.new_group(ranks=[i, i + tensor_world_size]) for i in range(tensor_world_size)]
        pipe_group = pipe_groups[rank % tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(512, 2048, 256).to(device=device)
            data = torch.randn((64, 512), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [TensorParallelLinear(512, 2048, tensor_world_size, rank, tensor_group)]
            model = PipelineStage(layers, rank).to(device=device)
        else:
            layers = [nn.ReLU(), TensorParallelLinear(2048, 256, tensor_world_size, rank, tensor_group, True)]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            l1_bias = sequential_model.layers[0].bias.data.clone()

            # Use tensor slicing for manual weight splitting
            split_size = l1_weight.size(0) // tensor_world_size

            l1_weight_split_0 = l1_weight[0:split_size, :]
            if not l1_weight_split_0.is_contiguous():
                l1_weight_split_0 = l1_weight_split_0.contiguous()
            l1_bias_split_0 = l1_bias[0:split_size]
            if not l1_bias_split_0.is_contiguous():
                l1_bias_split_0 = l1_bias_split_0.contiguous()
            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)
            model.layers[0].linear.bias.data.copy_(l1_bias_split_0)

            # Slice for rank 1 (second half)
            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            if not l1_weight_split_1.is_contiguous():
                l1_weight_split_1 = l1_weight_split_1.contiguous()
            l1_bias_split_1 = l1_bias[split_size : 2 * split_size]
            if not l1_bias_split_1.is_contiguous():
                l1_bias_split_1 = l1_bias_split_1.contiguous()

            dist.send(l1_weight_split_1, group_dst=1, group=tensor_group)
            dist.send(l1_bias_split_1, group_dst=1, group=tensor_group)

            l2_weight = sequential_model.layers[2].weight.data.clone()
            l2_bias = sequential_model.layers[2].bias.data.clone()

            # Use tensor slicing for manual weight splitting (dim=1 for column-wise split)
            split_size = l2_weight.size(1) // tensor_world_size

            # Slice for rank 2 (first half of columns)
            l2_weight_split_0 = l2_weight[:, 0:split_size]
            if not l2_weight_split_0.is_contiguous():
                l2_weight_split_0 = l2_weight_split_0.contiguous()

            dist.send(l2_weight_split_0, group_dst=1, group=pipe_group)
            dist.send(l2_bias, group_dst=1, group=pipe_group)

            # Slice for rank 3 (second half of columns)
            l2_weight_split_1 = l2_weight[:, split_size : 2 * split_size]
            if not l2_weight_split_1.is_contiguous():
                l2_weight_split_1 = l2_weight_split_1.contiguous()

            dist.send(l2_weight_split_1, dst=3)
            dist.send(l2_bias, dst=3)

        elif rank == 1:
            l1_weight = torch.empty((1024, 512), device=device, dtype=DTYPE)
            l1_bias = torch.empty(1024, device=device, dtype=DTYPE)

            dist.recv(l1_weight, src=0, group=tensor_group)
            dist.recv(l1_bias, src=0, group=tensor_group)

            model.layers[0].linear.weight.data.copy_(l1_weight)
            model.layers[0].linear.bias.data.copy_(l1_bias)

        elif rank == 2:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, group_src=0, group=pipe_group)
            dist.recv(l2_bias, group_src=0, group=pipe_group)

            model.layers[1].linear.weight.data.copy_(l2_weight)
            model.layers[1].linear.bias.data.copy_(l2_bias)

        elif rank == 3:
            l2_weight = torch.empty((256, 1024), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, src=0)
            dist.recv(l2_bias, src=0)

            model.layers[1].linear.weight.data.copy_(l2_weight)
            model.layers[1].linear.bias.data.copy_(l2_bias)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)

            # Use pipe group for rank 0 -> rank 2 communication
            dist.send(output_tp_pp, group_dst=1, group=pipe_group)
            dist.send(output_seq, group_dst=1, group=pipe_group)
        elif rank == 1:
            data_dummy = torch.empty((64, 512), device=device, dtype=DTYPE)

            output_tp_pp = model(data_dummy)

            # Use pipe group for rank 1 -> rank 3 communication
            dist.send(output_tp_pp, group_dst=1, group=pipe_group)
        elif rank == 2:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_tp_pp_recv, group_src=0, group=pipe_group)

            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)
        elif rank == 3:
            output_tp_pp_recv = torch.empty((64, 1024), device=device, dtype=DTYPE)

            # Use pipe group for rank 3 <- rank 1 communication
            dist.recv(output_tp_pp_recv, group_src=0, group=pipe_group)

            input_from_stage1 = output_tp_pp_recv
            output_tp_pp = model(input_from_stage1)

        if rank == 2:
            output_seq = torch.empty((64, 256), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_seq, group_src=0, group=pipe_group)

            # Move to CPU for comparison to avoid RBLN compilation of allclose
            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()

            # Check with different tolerance levels
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_two_linear_bias_allgather_subgroup_send_recv_test(rank: int, world_size: int, backend: str) -> None:
    """Test two linear layers with bias, ReLU, allgather, subgroup, send/recv."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:

        def tensor_to_view_list(tensor):
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            return [tensor[i] for i in range(tensor.shape[0])]

        DTYPE = torch.float16

        class TensorParallelLinear(nn.Module):
            def __init__(self, in_features, out_features, world_size, rank, group, contiguous_output=False):
                super().__init__()
                self.world_size = world_size
                self.rank = rank
                self.group = group
                self.contiguous_output = contiguous_output

                # Column-wise splitting: split output features
                self.out_features_per_partition = out_features // world_size
                self.linear = nn.Linear(in_features, self.out_features_per_partition, dtype=DTYPE)

            def forward(self, x):
                # Broadcast input to all ranks in the group
                src_rank_in_group = dist.get_global_rank(self.group, 0)
                dist.broadcast(x, src=src_rank_in_group, group=self.group)

                # Compute linear transformation using self.linear
                partial_output = self.linear(x)

                # Ensure partial_output is contiguous
                if not partial_output.is_contiguous():
                    partial_output = partial_output.contiguous()

                # All-gather the partial outputs from all ranks
                if self.contiguous_output:
                    gathered_output_shape = (self.world_size,) + partial_output.shape
                    t = torch.empty(gathered_output_shape, device=device, dtype=DTYPE)
                    gathered_outputs = tensor_to_view_list(t)
                else:
                    gathered_outputs = [
                        torch.empty(partial_output.shape, device=device, dtype=DTYPE) for _ in range(self.world_size)
                    ]

                dist.all_gather(gathered_outputs, partial_output, group=self.group)

                # Concatenate the gathered outputs
                all_outputs = torch.cat(gathered_outputs, dim=1)
                return all_outputs

        class PipelineStage(nn.Module):
            def __init__(self, layers, rank):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.rank = rank

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class SequentialModel(nn.Module):
            def __init__(self, hidden_dim_1, hidden_dim_2, hidden_dim_3):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(hidden_dim_1, hidden_dim_2, dtype=DTYPE),
                    nn.Linear(hidden_dim_2, hidden_dim_3, dtype=DTYPE),
                )

            def forward(self, x):
                return self.layers(x)

        assert world_size == 4, f"This test requires exactly 4 ranks, got {world_size}"

        set_deterministic_seeds(42)
        device = torch.device("rbln", rank)

        tensor_world_size = 2
        tensor_groups = [
            dist.new_group(list(range(i * tensor_world_size, (i + 1) * tensor_world_size)))
            for i in range(world_size // tensor_world_size)
        ]
        tensor_group = tensor_groups[rank // tensor_world_size]

        pipe_groups = [dist.new_group(ranks=[i, i + tensor_world_size]) for i in range(tensor_world_size)]
        pipe_group = pipe_groups[rank % tensor_world_size]

        model = None
        sequential_model = None
        data = None
        if rank == 0:
            sequential_model = SequentialModel(512, 2048, 256).to(device=device)
            data = torch.randn((64, 512), dtype=DTYPE, device=device)
            output_seq = sequential_model(data)

        if rank < tensor_world_size:
            layers = [TensorParallelLinear(512, 2048, tensor_world_size, rank, tensor_group)]
            model = PipelineStage(layers, rank).to(device=device)
        else:
            layers = [nn.Linear(2048, 256, dtype=DTYPE)]
            model = PipelineStage(layers, rank).to(device=device)

        dist.barrier()

        if rank == 0:
            l1_weight = sequential_model.layers[0].weight.data.clone()
            l1_bias = sequential_model.layers[0].bias.data.clone()

            # Use tensor slicing for manual weight splitting
            split_size = l1_weight.size(0) // tensor_world_size

            # Use tensor slicing for manual weight splitting
            l1_weight_split_0 = l1_weight[0:split_size, :]
            assert l1_weight_split_0.is_contiguous()

            l1_bias_split_0 = l1_bias[0:split_size]
            assert l1_bias_split_0.is_contiguous()

            model.layers[0].linear.weight.data.copy_(l1_weight_split_0)
            model.layers[0].linear.bias.data.copy_(l1_bias_split_0)

            # Slice for rank 1 (second half)
            l1_weight_split_1 = l1_weight[split_size : 2 * split_size, :]
            assert l1_weight_split_1.is_contiguous()

            l1_bias_split_1 = l1_bias[split_size : 2 * split_size]
            assert l1_bias_split_1.is_contiguous()

            dist.send(l1_weight_split_1, group_dst=1, group=tensor_group)
            dist.send(l1_bias_split_1, group_dst=1, group=tensor_group)

            l2_weight = sequential_model.layers[1].weight.data.clone()
            l2_bias = sequential_model.layers[1].bias.data.clone()

            # Use tensor slicing for manual weight splitting (dim=1 for column-wise split)
            split_size = l2_weight.size(0) // tensor_world_size

            # Slice for rank 2 (first half of columns)
            l2_weight_split_0 = l2_weight[0:split_size, :]
            assert l2_weight_split_0.is_contiguous()

            l2_bias_split_0 = l2_bias[0:split_size]
            assert l2_bias_split_0.is_contiguous()

            dist.send(l2_weight, group_dst=1, group=pipe_group)
            dist.send(l2_bias, group_dst=1, group=pipe_group)

            # Use default group for rank 0 -> rank 3 communication (cross-pipeline communication)
            dist.send(l2_weight, dst=3)
            dist.send(l2_bias, dst=3)

        elif rank == 1:
            l1_weight = torch.empty((1024, 512), device=device, dtype=DTYPE)
            l1_bias = torch.empty(1024, device=device, dtype=DTYPE)

            dist.recv(l1_weight, group_src=0, group=tensor_group)
            dist.recv(l1_bias, group_src=0, group=tensor_group)

            model.layers[0].linear.weight.data.copy_(l1_weight)
            model.layers[0].linear.bias.data.copy_(l1_bias)

        elif rank == 2:
            l2_weight = torch.empty((256, 2048), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, group_src=0, group=pipe_group)
            dist.recv(l2_bias, group_src=0, group=pipe_group)

            model.layers[0].weight.data.copy_(l2_weight)
            model.layers[0].bias.data.copy_(l2_bias)

        elif rank == 3:
            l2_weight = torch.empty((256, 2048), device=device, dtype=DTYPE)
            l2_bias = torch.empty(256, device=device, dtype=DTYPE)

            dist.recv(l2_weight, src=0)
            dist.recv(l2_bias, src=0)

            model.layers[0].weight.data.copy_(l2_weight)
            model.layers[0].bias.data.copy_(l2_bias)

        dist.barrier()

        output_tp_pp = None
        if rank == 0:
            output_tp_pp = model(data)

            # Use pipe group for rank 0 -> rank 2 communication
            dist.send(output_tp_pp, group_dst=1, group=pipe_group)
            dist.send(output_seq, group_dst=1, group=pipe_group)
        elif rank == 1:
            data_dummy = torch.empty((64, 512), device=device, dtype=DTYPE)

            output_tp_pp = model(data_dummy)
            # Rank 1 only runs model, no send/recv
        elif rank == 2:
            output_tp_pp_recv = torch.empty((64, 2048), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_tp_pp_recv, group_src=0, group=pipe_group)

            output_tp_pp = model(output_tp_pp_recv)
        elif rank == 3:
            # Rank 3 receives input through broadcast in TensorParallelLinear forward
            # The input will be broadcasted from rank 2 in the TensorParallelLinear forward
            output_tp_pp_recv = torch.empty((64, 2048), device=device, dtype=DTYPE)

            output_tp_pp = model(output_tp_pp_recv)

        if rank == 2:
            output_seq = torch.empty((64, 256), device=device, dtype=DTYPE)

            # Use pipe group for rank 2 <- rank 0 communication
            dist.recv(output_seq, group_src=0, group=pipe_group)

            # Move to CPU for comparison to avoid RBLN compilation of allclose
            final_tp_pp_cpu = output_tp_pp.cpu()
            output_seq_cpu = output_seq.cpu()

            # Check with different tolerance levels
            is_same_relaxed = torch.allclose(final_tp_pp_cpu, output_seq_cpu, atol=1e-2)
            assert is_same_relaxed, "TP+PP result does not match Sequential result"

        dist.barrier()

    finally:
        dist.destroy_process_group()


class TestTPPPRBLNBase(TestCase):
    """Base test class for RBLN TP/PP tests."""

    # CI tests run only sync mode; release tests cover both sync and async modes.
    c10d_async_envs = [
        subtest("0", decorators=[pytest.mark.test_set_ci]),  # sync mode (TORCH_RBLN_C10D_ASYNC=0)
        "1",  # async mode (TORCH_RBLN_C10D_ASYNC=1)
    ]

    def setUp(self):
        os.environ["RCCL_FORCE_EXPORT_MEM"] = "1"
        os.environ["RBLN_ROOT_IP"] = "127.0.0.1"
        os.environ["RBLN_LOCAL_IP"] = "127.0.0.1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        configure_master_port_for_rccl_tests()
        self.backend = "rbln-ccl"
        self.device_count = torch.rbln.device_count()
        self.world_size_2 = min(self.device_count, 2)
        self.world_size_4 = min(self.device_count, 4)

    def should_skip_multi_rank_tests(self):
        """Check if multi-rank tests should be skipped."""
        return self.world_size_2 <= 1

    def should_skip_4rank_tests(self):
        """Check if 4-rank tests should be skipped."""
        return self.world_size_4 < 4

    def run_c10d_test(self, c10d_async_env, nprocs, test_func, args):
        """Helper to run a test function with mp.spawn."""
        with pytest.MonkeyPatch.context() as ctx:
            ctx.setenv("TORCH_RBLN_C10D_ASYNC", c10d_async_env)

            mp.spawn(test_func, args=args, nprocs=nprocs, join=True)


@pytest.mark.single_worker
class TestDefaultGroupAllReduceRBLN(TestTPPPRBLNBase):
    """Test cases for default group allreduce operations."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_default_group_allreduce_2ranks(self, c10d_async_env):
        """Test default group allreduce with 2 ranks."""

        self.run_c10d_test(
            c10d_async_env, self.world_size_2, run_default_group_allreduce_test, (self.world_size_2, self.backend)
        )


@pytest.mark.single_worker
class TestSingleLinearAllgatherDefaultGroupRBLN(TestTPPPRBLNBase):
    """Test cases for single linear layer with allgather, default group."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_single_linear_allgather_default_group(self, c10d_async_env):
        """Test single linear layer with allgather, default group."""
        if self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")

        self.run_c10d_test(
            c10d_async_env,
            self.world_size_2,
            run_single_linear_allgather_default_group_test,
            (self.world_size_2, self.backend),
        )


@pytest.mark.single_worker
class TestPP2RBLN(TestTPPPRBLNBase):
    """Test cases for PP=2 pipeline parallel operations."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_pp2_pipeline_parallel(self, c10d_async_env):
        """Test PP=2 pipeline parallel."""
        if self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")

        self.run_c10d_test(c10d_async_env, self.world_size_2, run_pp2_test, (self.world_size_2, self.backend))


@pytest.mark.single_worker
class TestTwoLinearBiasDefaultGroupRBLN(TestTPPPRBLNBase):
    """Test cases for two linear layers with bias, default group, send/recv."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_two_linear_bias_default_group_send_recv(self, c10d_async_env):
        """Test two linear layers with bias, default group, send/recv."""
        if self.should_skip_4rank_tests():
            self.skipTest("Requires at least 4 devices")

        self.run_c10d_test(
            c10d_async_env,
            self.world_size_4,
            run_two_linear_bias_default_group_send_recv_test,
            (self.world_size_4, self.backend),
        )


@pytest.mark.single_worker
class TestTwoLinearNoBiasDefaultGroupRBLN(TestTPPPRBLNBase):
    """Test cases for two linear layers without bias, default group, send/recv."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_two_linear_no_bias_default_group_send_recv(self, c10d_async_env):
        """Test two linear layers without bias, default group, send/recv."""
        if self.should_skip_4rank_tests():
            self.skipTest("Requires at least 4 devices")
        self.run_c10d_test(
            c10d_async_env,
            self.world_size_4,
            run_two_linear_no_bias_default_group_send_recv_test,
            (self.world_size_4, self.backend),
        )


@pytest.mark.single_worker
class TestTwoLinearBiasSubgroupRBLN(TestTPPPRBLNBase):
    """Test cases for two linear layers with bias, subgroup, send/recv."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_two_linear_bias_subgroup_send_recv(self, c10d_async_env):
        """Test two linear layers with bias, subgroup, send/recv."""
        if self.should_skip_4rank_tests():
            self.skipTest("Requires at least 4 devices")

        self.run_c10d_test(
            c10d_async_env,
            self.world_size_4,
            run_two_linear_bias_subgroup_send_recv_test,
            (self.world_size_4, self.backend),
        )


@pytest.mark.single_worker
class TestTwoLinearBiasReluSubgroupRBLN(TestTPPPRBLNBase):
    """Test cases for two linear layers with bias and ReLU, subgroup, send/recv."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_two_linear_bias_relu_subgroup_send_recv(self, c10d_async_env):
        """Test two linear layers with bias and ReLU, subgroup, send/recv."""
        if self.should_skip_4rank_tests():
            self.skipTest("Requires at least 4 devices")

        self.run_c10d_test(
            c10d_async_env,
            self.world_size_4,
            run_two_linear_bias_relu_subgroup_send_recv_test,
            (self.world_size_4, self.backend),
        )


@pytest.mark.single_worker
class TestTwoLinearBiasAllgatherSubgroupRBLN(TestTPPPRBLNBase):
    """Test cases for two linear layers with bias, allgather, subgroup, send/recv."""

    @parametrize("c10d_async_env", TestTPPPRBLNBase.c10d_async_envs)
    def test_two_linear_bias_allgather_subgroup_send_recv(self, c10d_async_env):
        """Test two linear layers with bias, allgather, subgroup, send/recv."""
        if self.should_skip_4rank_tests():
            self.skipTest("Requires at least 4 devices")

        self.run_c10d_test(
            c10d_async_env,
            self.world_size_4,
            run_two_linear_bias_allgather_subgroup_send_recv_test,
            (self.world_size_4, self.backend),
        )


# Instantiate device type tests for all test classes
instantiate_device_type_tests(TestDefaultGroupAllReduceRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestSingleLinearAllgatherDefaultGroupRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestPP2RBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTwoLinearBiasDefaultGroupRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTwoLinearNoBiasDefaultGroupRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTwoLinearBiasSubgroupRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTwoLinearBiasReluSubgroupRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTwoLinearBiasAllgatherSubgroupRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
