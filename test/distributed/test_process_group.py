# Owner(s): ["module: PrivateUse1"]

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, subtest, TestCase

from test.utils import configure_master_port_for_rccl_tests, setup_distributed_environment, SUPPORTED_DTYPES


KiB = 1024
MiB = 1024 * 1024

TEST_DTYPES = SUPPORTED_DTYPES + [torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64]


def setup_environment(rank: int, world_size: int) -> None:
    """Setup environment variables for distributed testing."""
    setup_distributed_environment(rank, world_size)


def run_allreduce_test(rank: int, world_size: int, backend: str, op: dist.ReduceOp) -> None:
    """Test allreduce operation with specific reduce op."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Create test tensor with rank-specific values
        base_value = rank + 1.0
        tensor = torch.full([64], base_value, dtype=torch.float16, device=torch.device(f"rbln:{rank}"))

        # Perform allreduce
        dist.all_reduce(tensor, op=op)

        # Verify result based on operation
        expected_value = _get_expected_allreduce_value(base_value, world_size, op)
        assert tensor[0] == expected_value, (
            f"allreduce {op} failed on rank {rank}: expected={expected_value}, actual={tensor[0]}"
        )

    finally:
        dist.destroy_process_group()


def run_broadcast_test(rank: int, world_size: int, backend: str, src: int, dtype: torch.dtype) -> None:
    """Test broadcast operation from specific source rank with specified dtype."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        # Create tensor with different values based on rank
        # Use appropriate value based on dtype
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            broadcast_value = 42
        else:
            broadcast_value = 42.0

        if rank == src:
            tensor = torch.full([64], broadcast_value, dtype=dtype, device=device)
        else:
            tensor = torch.zeros(64, dtype=dtype, device=device)

        # Perform broadcast
        dist.broadcast(tensor, src)

        # Verify all ranks have the broadcasted value
        expected_tensor = torch.full_like(tensor, broadcast_value, device=device)
        torch.testing.assert_close(
            tensor.cpu(),
            expected_tensor.cpu(),
            msg=(
                f"broadcast failed on rank {rank} with dtype {dtype}: "
                f"expected all elements to be {broadcast_value}, "
                f"but got tensor with values: {tensor}"
                f"expected tensor with values: {expected_tensor}"
            ),
        )

    finally:
        dist.destroy_process_group()


def run_broadcast_zero_sized_test(rank: int, world_size: int, backend: str, src: int, dtype: torch.dtype) -> None:
    """Test broadcast operation with zero-sized tensor."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Create zero-sized tensor (shape [0])
        tensor = torch.empty([0], dtype=dtype, device=torch.device(f"rbln:{rank}"))

        # Perform broadcast - should not raise an error
        dist.broadcast(tensor, src)

        # Verify tensor is still zero-sized after broadcast
        assert tensor.shape == torch.Size([0]), (
            f"broadcast zero-sized tensor failed on rank {rank}: expected shape [0], but got shape {tensor.shape}"
        )
        assert tensor.numel() == 0, (
            f"broadcast zero-sized tensor failed on rank {rank}: expected numel=0, but got numel={tensor.numel()}"
        )

    finally:
        dist.destroy_process_group()


def run_scatter_test(rank: int, world_size: int, backend: str, root: int, dtype: torch.dtype, dim0: int = 32) -> None:
    """Test scatter operation from specific root rank with specified dtype and tensor size [dim0]."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        # Create input data for scatter
        # Use appropriate value based on dtype
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            use_int = True
        else:
            use_int = False

        if rank == root:
            # Root rank has data for all ranks
            input_list = []
            for i in range(world_size):
                value = i + 1 if use_int else float(i + 1)
                tensor = torch.full([dim0], value, dtype=dtype, device=device)
                input_list.append(tensor)
        else:
            # Non-root ranks should pass None for scatter_list
            input_list = None

        # Create output tensor
        output = torch.zeros(dim0, dtype=dtype, device=device)

        # Perform scatter
        dist.scatter(output, input_list, src=root)

        # Verify result - each rank should receive its own data
        expected_value = rank + 1 if use_int else float(rank + 1)
        expected_tensor = torch.full_like(output, expected_value, device=device)
        torch.testing.assert_close(
            output.cpu(),
            expected_tensor.cpu(),
            msg=(
                f"scatter failed on rank {rank} with dtype {dtype} dim0={dim0}: "
                f"expected all elements to be {expected_value}, "
                f"but got tensor with values: {output}"
                f"expected tensor with values: {expected_tensor}"
            ),
        )

    finally:
        dist.destroy_process_group()


def run_send_recv_test(rank: int, world_size: int, backend: str, src: int, dst: int, dtype: torch.dtype) -> None:
    """Test send/recv operation between source and destination ranks with specified dtype."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        if world_size < 2:
            return  # Skip if not enough ranks

        # Use appropriate value based on dtype
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            send_value = 42
        else:
            send_value = 42.0

        if rank == src:
            # Source rank sends data to destination rank
            tensor = torch.full([64], send_value, dtype=dtype, device=device)
            dist.send(tensor, dst=dst)
        elif rank == dst:
            # Destination rank receives data from source rank
            tensor = torch.zeros(64, dtype=dtype, device=device)
            dist.recv(tensor, src=src)
            # Verify received data
            expected_tensor = torch.full_like(tensor, send_value, device=device)
            torch.testing.assert_close(
                tensor.cpu(),
                expected_tensor.cpu(),
                msg=(
                    f"send/recv failed on rank {rank} with dtype {dtype}: "
                    f"expected all elements to be {send_value}, "
                    f"but got tensor with values: {tensor}"
                    f"expected tensor with values: {expected_tensor}"
                ),
            )

    finally:
        dist.destroy_process_group()


def run_allgather_test(rank: int, world_size: int, backend: str, dtype: torch.dtype, size: int) -> None:
    """Test allgather operation."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        # Use appropriate value based on dtype
        use_int = dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

        # Create input tensor with rank-specific values
        input_value = rank + 1 if use_int else float(rank + 1)
        input_tensor = torch.full([size], input_value, dtype=dtype, device=device)

        # Create output list for gathering results from all ranks
        output_list = []
        for _ in range(world_size):
            output_tensor = torch.zeros(size, dtype=dtype, device=device)
            output_list.append(output_tensor)

        # Perform allgather
        dist.all_gather(output_list, input_tensor)

        # Verify results - each position should contain the corresponding rank's value
        for i, output_tensor in enumerate(output_list):
            expected_value = i + 1 if use_int else float(i + 1)
            expected_tensor = torch.full_like(output_tensor, expected_value, device=device)
            torch.testing.assert_close(
                output_tensor.cpu(),
                expected_tensor.cpu(),
                msg=(
                    f"allgather failed on rank {rank} with dtype {dtype}: "
                    f"output_list[{i}] expected all elements to be {expected_value}, "
                    f"but got tensor with values: {output_tensor}"
                    f"expected tensor with values: {expected_tensor}"
                ),
            )

    finally:
        dist.destroy_process_group()


def run_allgather_into_tensor_coalesced_test(
    rank: int,
    world_size: int,
    backend: str,
    dtype: torch.dtype,
    num_tensors: int,
    size_per_tensor: int,
) -> None:
    """Test allgather_into_tensor_coalesced operation."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        from torch.distributed import distributed_c10d as c10d

        use_int = dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
        input_value = rank + 1 if use_int else float(rank + 1)

        input_tensors = [
            torch.full([size_per_tensor], input_value, dtype=dtype, device=device) for _ in range(num_tensors)
        ]
        output_tensors = [
            torch.zeros(world_size * size_per_tensor, dtype=dtype, device=device) for _ in range(num_tensors)
        ]

        group = c10d._get_default_group()
        work = group.allgather_into_tensor_coalesced(output_tensors, input_tensors)
        work.wait()

        for t in range(num_tensors):
            for r in range(world_size):
                expected_val = r + 1 if use_int else float(r + 1)
                start = r * size_per_tensor
                end = start + size_per_tensor
                chunk = output_tensors[t][start:end]
                expected = torch.full_like(chunk, expected_val, device=device)
                torch.testing.assert_close(
                    chunk.cpu(),
                    expected.cpu(),
                    msg=(
                        f"allgather_into_tensor_coalesced failed rank={rank} tensor_idx={t} chunk r={r}: "
                        f"expected {expected_val}, got {chunk}"
                    ),
                )
    finally:
        dist.destroy_process_group()


def run_reduce_scatter_test(rank: int, world_size: int, backend: str, op: dist.ReduceOp, size: int) -> None:
    """Test reduce_scatter operation with specific reduce op."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        # Create input list: each rank has world_size tensors
        # For rank i, input_list[j] will be reduced across all ranks
        # and rank i will receive the j-th chunk of the reduced result
        input_list = []
        output_size = size
        for j in range(world_size):
            # Each tensor has values based on rank and j
            base_value = float((rank + 1) * 10 + j + 1)
            tensor = torch.full([output_size], base_value, dtype=torch.float16, device=device)
            input_list.append(tensor)

        # Create output tensor
        output = torch.zeros(output_size, dtype=torch.float16, device=device)

        # Perform reduce_scatter
        dist.reduce_scatter(output, input_list, op=op)

        # Verify result
        # Each rank i should receive the reduced result of input_list[i] from all ranks
        # For rank i, we reduce input_list[i] across all ranks
        expected_value = _get_expected_reduce_scatter_value(rank, world_size, op)
        expected_tensor = torch.full_like(output, expected_value, device=device)
        torch.testing.assert_close(
            output.cpu(),
            expected_tensor.cpu(),
            msg=(
                f"reduce_scatter {op} failed on rank {rank}: "
                f"expected all elements to be {expected_value}, "
                f"but got tensor with values: {output[:5]}"
                f"expected tensor with values: {expected_tensor[:5]}"
            ),
        )

    finally:
        dist.destroy_process_group()


def run_barrier_test(rank: int, world_size: int, backend: str) -> None:
    """Test barrier synchronization."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Test barrier
        dist.barrier()

    finally:
        dist.destroy_process_group()


def run_allreduce_size_test(
    rank: int, world_size: int, backend: str, size_bytes: int, dtype: torch.dtype = torch.float16
) -> None:
    """Test allreduce operation with specific data size."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device("rbln", rank)

    try:
        # Calculate number of elements based on size and dtype
        element_size = dtype.itemsize  # bytes per element
        num_elements = size_bytes // element_size

        # Create test tensor with rank-specific values
        base_value = rank + 1.0
        tensor = torch.full([num_elements], base_value, dtype=dtype, device=device)

        # Perform allreduce with SUM operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Verify result - sum of all rank values
        expected_value = sum(range(1, world_size + 1))  # 1+2+...+world_size

        # Check first and last elements to verify correctness
        assert tensor[0] == expected_value, (
            f"allreduce size test failed on rank {rank} for size {size_bytes} bytes: "
            f"expected={expected_value}, actual={tensor[0]}"
        )
        assert tensor[-1] == expected_value, (
            f"allreduce size test failed on rank {rank} for size {size_bytes} bytes (last element): "
            f"expected={expected_value}, actual={tensor[-1]}"
        )

    finally:
        dist.destroy_process_group()


def run_unsupported_op_test(rank: int, world_size: int, backend: str) -> None:
    """Test that unsupported operations raise appropriate errors."""
    setup_environment(rank, world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # This test can be used for other unsupported operations
        # reduce_scatter is now supported, so it's removed from here
        pass

    finally:
        dist.destroy_process_group()


def _get_expected_allreduce_value(base_value: float, world_size: int, op: dist.ReduceOp) -> float:
    """Calculate expected value after allreduce operation."""
    if op == dist.ReduceOp.SUM:
        return sum(range(1, world_size + 1))  # 1+2+...+world_size
    elif op == dist.ReduceOp.AVG:
        return sum(range(1, world_size + 1)) / world_size
    elif op == dist.ReduceOp.MIN:
        return 1.0  # minimum of 1, 2, ..., world_size
    elif op == dist.ReduceOp.MAX:
        return float(world_size)  # maximum of 1, 2, ..., world_size
    elif op == dist.ReduceOp.PRODUCT:
        result = 1.0
        for i in range(1, world_size + 1):
            result *= i
        return result
    else:
        raise ValueError(f"Unsupported reduce operation: {op}")


def _get_expected_reduce_scatter_value(rank: int, world_size: int, op: dist.ReduceOp) -> float:
    """Calculate expected value after reduce_scatter operation.

    For rank i, reduce_scatter reduces input_list[i] across all ranks.
    Each rank i has input_list[i] with value (rank+1)*10 + (i+1).
    """
    if op == dist.ReduceOp.SUM:
        # Sum of (rank+1)*10 + (i+1) across all ranks for input_list[i]
        # For rank i, we reduce input_list[i] = [(r+1)*10 + (i+1)] for r in range(world_size)
        return sum((r + 1) * 10 + (rank + 1) for r in range(world_size))
    elif op == dist.ReduceOp.AVG:
        return sum((r + 1) * 10 + (rank + 1) for r in range(world_size)) / world_size
    elif op == dist.ReduceOp.MIN:
        # Minimum of (r+1)*10 + (i+1) across all ranks
        return min((r + 1) * 10 + (rank + 1) for r in range(world_size))
    elif op == dist.ReduceOp.MAX:
        # Maximum of (r+1)*10 + (i+1) across all ranks
        return max((r + 1) * 10 + (rank + 1) for r in range(world_size))
    elif op == dist.ReduceOp.PRODUCT:
        result = 1.0
        for r in range(world_size):
            result *= (r + 1) * 10 + (rank + 1)
        return result
    else:
        raise ValueError(f"Unsupported reduce operation: {op}")


class TestProcessGroupRBLNBase(TestCase):
    """Base test class for RBLN process group tests."""

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
        self.world_size = min(torch.rbln.device_count(), 2)

    def should_skip_multi_rank_tests(self):
        """Check if multi-rank tests should be skipped."""
        return self.world_size <= 1

    def run_c10d_test(self, c10d_async_env, nprocs, test_func, args):
        """Helper to run a test function with mp.spawn."""
        with pytest.MonkeyPatch.context() as ctx:
            ctx.setenv("TORCH_RBLN_C10D_ASYNC", c10d_async_env)

            mp.spawn(test_func, args=args, nprocs=nprocs, join=True)


@pytest.mark.single_worker
class TestBroadcastRBLN(TestProcessGroupRBLNBase):
    """Test cases for broadcast operations."""

    @dtypes(*TEST_DTYPES)
    @parametrize("rank", [0, 1])
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_broadcast(self, dtype, rank, c10d_async_env):
        """Test broadcast on specified rank."""
        if (rank >= 1) and self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")

        self.run_c10d_test(
            c10d_async_env, self.world_size, run_broadcast_test, (self.world_size, self.backend, rank, dtype)
        )

    @dtypes(*TEST_DTYPES)
    @parametrize("rank", [0, 1])
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_broadcast_zero_sized_tensor(self, dtype, rank, c10d_async_env):
        """Test broadcast with zero-sized tensor on specified rank."""
        if (rank >= 1) and self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")

        self.run_c10d_test(
            c10d_async_env, self.world_size, run_broadcast_zero_sized_test, (self.world_size, self.backend, rank, dtype)
        )


@pytest.mark.single_worker
class TestAllReduceRBLN(TestProcessGroupRBLNBase):
    """Test cases for allreduce operations."""

    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allreduce_sum(self, c10d_async_env):
        """Test allreduce with SUM operation."""
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_allreduce_test, (self.world_size, self.backend, dist.ReduceOp.SUM)
        )

    # NOTE: RCCL support only SUM operation
    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_allreduce_avg(self, c10d_async_env):
    #     """Test allreduce with AVG operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_allreduce_test, (self.world_size, self.backend, dist.ReduceOp.AVG)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_allreduce_min(self, c10d_async_env):
    #     """Test allreduce with MIN operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_allreduce_test, (self.world_size, self.backend, dist.ReduceOp.MIN)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_allreduce_max(self, c10d_async_env):
    #     """Test allreduce with MAX operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_allreduce_test, (self.world_size, self.backend, dist.ReduceOp.MAX)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_allreduce_product(self, c10d_async_env):
    #     """Test allreduce with PRODUCT operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_allreduce_test, (self.world_size, self.backend, dist.ReduceOp.PRODUCT)
    #     )


@pytest.mark.single_worker
class TestScatterRBLN(TestProcessGroupRBLNBase):
    """Test cases for scatter operations. Each test runs with dim0 in [32, 256]."""

    @dtypes(*TEST_DTYPES)
    @parametrize("rank", [0, 1])
    @parametrize("dim0", [32, 256])
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_scatter(self, dtype, rank, dim0, c10d_async_env):
        """Test scatter on specified rank."""
        if (rank >= 1) and self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")

        self.run_c10d_test(
            c10d_async_env, self.world_size, run_scatter_test, (self.world_size, self.backend, rank, dtype, dim0)
        )


@pytest.mark.single_worker
class TestAllGatherRBLN(TestProcessGroupRBLNBase):
    """Test cases for allgather operations."""

    @dtypes(*TEST_DTYPES)
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allgather_unaligned(self, dtype, c10d_async_env):
        """Test allgather operation with various dtypes."""
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_allgather_test, (self.world_size, self.backend, dtype, 10)
        )

    @dtypes(*TEST_DTYPES)
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allgather_aligned(self, dtype, c10d_async_env):
        """Test allgather operation with various dtypes."""
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_allgather_test, (self.world_size, self.backend, dtype, 64)
        )

    @parametrize(
        "size", [256, 512, (512 * KiB), MiB, (2 * MiB), (4 * MiB), (8 * MiB), (16 * MiB), (32 * MiB), (64 * MiB)]
    )
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allgather_float16_aligned(self, size, c10d_async_env):
        """Test allgather with various sizes."""
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_allgather_test, (self.world_size, self.backend, torch.float16, size)
        )

    @parametrize(
        "size",
        [
            (MiB + 10),  # unaligned 1 MiB
            (2 * MiB + 22),  # unaligned 2 MiB
            (4 * MiB + 44),  # unaligned 4 MiB
            (8 * MiB + 88),  # unaligned 8 MiB
            (16 * MiB + 176),  # unaligned 16 MiB
            (32 * MiB + 352),  # unaligned 32 MiB
            (64 * MiB + 704),  # unaligned 64 MiB
        ],
    )
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allgather_float16_unaligned(self, size, c10d_async_env):
        """Test allgather with various sizes."""
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_allgather_test, (self.world_size, self.backend, torch.float16, size)
        )

    @dtypes(*TEST_DTYPES)
    @parametrize("num_tensors", [1, 2])
    @parametrize("size_per_tensor", [32, 64, 128])
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allgather_into_tensor_coalesced(self, dtype, num_tensors, size_per_tensor, c10d_async_env):
        """Test allgather_into_tensor_coalesced with various configurations."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_allgather_into_tensor_coalesced_test,
            (self.world_size, self.backend, dtype, num_tensors, size_per_tensor),
        )


@pytest.mark.single_worker
class TestReduceScatterRBLN(TestProcessGroupRBLNBase):
    """Test cases for reduce_scatter operations."""

    @parametrize(
        "size",
        [1, 63, 64, 256, 512, (512 * KiB), MiB, (2 * MiB), (4 * MiB), (8 * MiB), (16 * MiB), (32 * MiB), (64 * MiB)],
    )
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_reduce_scatter_sum_aligned(self, size, c10d_async_env):
        """Test reduce_scatter with SUM operation."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_reduce_scatter_test,
            (self.world_size, self.backend, dist.ReduceOp.SUM, size),
        )

    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_reduce_scatter_sum_chunk_limit_no_chunking(self, c10d_async_env):
        """Test reduce_scatter with SUM operation."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_reduce_scatter_test,
            (self.world_size, self.backend, dist.ReduceOp.SUM, 1024 * 1024 * 15 // 4),
        )

    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_reduce_scatter_sum_chunk_limit_chunking(self, c10d_async_env):
        """Test reduce_scatter with SUM operation."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_reduce_scatter_test,
            (self.world_size, self.backend, dist.ReduceOp.SUM, 1024 * 1024 * 15 // 4 + 1),
        )

    @parametrize(
        "size",
        [
            (MiB + 10),  # unaligned 1 MiB
            (2 * MiB + 22),  # unaligned 2 MiB
            (4 * MiB + 44),  # unaligned 4 MiB
            (8 * MiB + 88),  # unaligned 8 MiB
            (16 * MiB + 100),  # unaligned 16 MiB
            (32 * MiB + 200),  # unaligned 32 MiB
            (64 * MiB + 300),  # unaligned 64 MiB
        ],
    )
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_reduce_scatter_sum_unaligned(self, size, c10d_async_env):
        """Test reduce_scatter with SUM operation."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_reduce_scatter_test,
            (self.world_size, self.backend, dist.ReduceOp.SUM, size),
        )

    # NOTE: RCCL support only SUM operation
    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_reduce_scatter_avg(self, c10d_async_env):
    #     """Test reduce_scatter with AVG operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_reduce_scatter_test, (self.world_size, self.backend, dist.ReduceOp.AVG)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_reduce_scatter_min(self, c10d_async_env):
    #     """Test reduce_scatter with MIN operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_reduce_scatter_test, (self.world_size, self.backend, dist.ReduceOp.MIN)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_reduce_scatter_max(self, c10d_async_env):
    #     """Test reduce_scatter with MAX operation."""
    #     self.run_c10d_test(
    #         c10d_async_env, self.world_size, run_reduce_scatter_test, (self.world_size, self.backend, dist.ReduceOp.MAX)
    #     )

    # @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    # def test_reduce_scatter_product(self, c10d_async_env):
    #     """Test reduce_scatter with PRODUCT operation."""
    #     self.run_c10d_test(
    #         c10d_async_env,
    #         self.world_size,
    #         run_reduce_scatter_test,
    #         (self.world_size, self.backend, dist.ReduceOp.PRODUCT),
    #     )


@pytest.mark.single_worker
class TestBarrierRBLN(TestProcessGroupRBLNBase):
    """Test cases for barrier synchronization."""

    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_barrier(self, c10d_async_env):
        """Test barrier synchronization."""
        self.run_c10d_test(c10d_async_env, self.world_size, run_barrier_test, (self.world_size, self.backend))


@pytest.mark.single_worker
class TestSendRecvRBLN(TestProcessGroupRBLNBase):
    """Test cases for send/recv operations."""

    @dtypes(*TEST_DTYPES)
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_send_recv(self, dtype, c10d_async_env):
        """Test send/recv with specified dtype from rank 0 to rank 1."""
        if self.should_skip_multi_rank_tests():
            self.skipTest("Requires world_size > 1")
        self.run_c10d_test(
            c10d_async_env, self.world_size, run_send_recv_test, (self.world_size, self.backend, 0, 1, dtype)
        )


@pytest.mark.single_worker
class TestUnsupportedOpsRBLN(TestProcessGroupRBLNBase):
    """Test cases for unsupported operations."""

    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_unsupported_operations(self, c10d_async_env):
        """Test that unsupported operations raise appropriate errors."""
        self.run_c10d_test(c10d_async_env, self.world_size, run_unsupported_op_test, (self.world_size, self.backend))


@pytest.mark.single_worker
class TestAllReduceSizesRBLN(TestProcessGroupRBLNBase):
    """Test cases for allreduce operations with various data sizes."""

    @parametrize(
        "size",
        [
            512,
            KiB,
            (10 * KiB),
            (100 * KiB),
            (512 * KiB),
            MiB,
            (2 * MiB),
            (3 * MiB),
            (4 * MiB),
            (5 * MiB),
            (8 * MiB),
            (10 * MiB),
            (16 * MiB),
            (32 * MiB),
            3407872,  # previously problematic size
            6815744,  # previously problematic size
            10223616,  # previously problematic size
            13631488,  # previously problematic size
        ],
    )
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allreduce_float16(self, size, c10d_async_env):
        """Test allreduce with specified bytes using torch.float16."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_allreduce_size_test,
            (self.world_size, self.backend, size, torch.float16),
        )

    @parametrize("size", [MiB, (4 * MiB)])
    @parametrize("c10d_async_env", TestProcessGroupRBLNBase.c10d_async_envs)
    def test_allreduce_float32(self, size, c10d_async_env):
        """Test allreduce with specified size using torch.float32."""
        self.run_c10d_test(
            c10d_async_env,
            self.world_size,
            run_allreduce_size_test,
            (self.world_size, self.backend, size, torch.float32),
        )


# Instantiate device type tests for all test classes
instantiate_device_type_tests(TestBroadcastRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAllReduceRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestScatterRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAllGatherRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestReduceScatterRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestBarrierRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestSendRecvRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestUnsupportedOpsRBLN, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestAllReduceSizesRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
