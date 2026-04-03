#!/usr/bin/env python3
"""
Benchmark script for collective communication operations in ProcessGroupRBLN.

This script measures the performance of various collective communication operations
(allreduce, broadcast, scatter, allgather, reduce_scatter, barrier, send, recv) with different data sizes.
"""

import argparse
import contextlib
import datetime
import os
import platform
import re
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple  # noqa: UP035

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase


def setup_environment(rank: int, world_size: int, enable_rccl_perf: bool = False) -> None:
    """Setup environment variables for distributed testing."""
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RBLN_ROOT_IP"] = "127.0.0.1"
    os.environ["RBLN_LOCAL_IP"] = "127.0.0.1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29604"
    if enable_rccl_perf:
        os.environ["RCCL_PERF"] = "2"
    elif "RCCL_PERF" in os.environ:
        del os.environ["RCCL_PERF"]
    torch.rbln.set_device(rank)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given dtype."""
    dtype_sizes = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
    }
    return dtype_sizes.get(dtype, 2)  # Default to 2 bytes (float16)


def get_system_info() -> Dict[str, str]:
    """Collect system information including CPU, RAM, and NPU details.

    Returns:
        Dictionary containing system information:
        - cpu_model: CPU model name
        - cpu_count: Number of CPU cores
        - ram_size_gb: Total RAM size in GB
        - npu_count: Number of NPU devices
    """
    info = {}

    # Get CPU information
    try:
        if platform.system() == "Linux":
            # Read CPU model from /proc/cpuinfo
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower():
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
        else:
            info["cpu_model"] = platform.processor() or "Unknown"
    except Exception:
        info["cpu_model"] = "Unknown"

    # Get CPU count
    try:
        info["cpu_count"] = str(os.cpu_count() or mp.cpu_count())
    except Exception:
        info["cpu_count"] = "Unknown"

    # Get RAM size
    try:
        if platform.system() == "Linux":
            # Read total RAM from /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        info["ram_size_gb"] = f"{mem_gb:.2f}"
                        break
        else:
            info["ram_size_gb"] = "Unknown"
    except Exception:
        info["ram_size_gb"] = "Unknown"

    # Get NPU count
    try:
        npu_count = torch.rbln.physical_device_count()
        info["npu_count"] = str(npu_count)
    except Exception:
        info["npu_count"] = "Unknown"

    return info


def parse_rccl_logs(stdout_content: str, op_type: str, data_size_bytes: int = 0) -> List[float]:
    """
    Parse RCCL performance logs from stdout.

    Expected log formats:
    - RCCL_SEND-2097152 byte processed 1265(us)
    - RCCL_RECV-2097152 byte processed 1227(us)
    - RCCL_ALLREDUCE-2097152 byte processed 780(us)

    Args:
        stdout_content: Content captured from stdout
        op_type: Operation type to match (e.g., 'allreduce', 'broadcast', 'scatter', 'allgather', 'send', 'recv')
        data_size_bytes: Data size in bytes (used to match RCCL_SEND-{data_size_bytes} format)

    Returns:
        List of times in seconds
    """
    times = []

    if not stdout_content:
        return times

    # Map operation types to RCCL log patterns
    op_patterns = {
        "allreduce": r"RCCL_ALLREDUCE",
        "broadcast": r"RCCL_BROADCAST",
        "scatter": r"RCCL_SCATTER",
        "allgather": r"RCCL_ALLGATHER",
        "reduce_scatter": r"RCCL_REDUCESCATTER",
        "send": r"RCCL_SEND",
        "recv": r"RCCL_RECV",
        "barrier": r"RCCL_BROADCAST",
    }

    pattern_key = op_type.lower()
    if pattern_key not in op_patterns:
        return times

    rccl_op_pattern = op_patterns[pattern_key]

    # Use data_size_bytes directly (it's already in bytes)
    expected_bytes = max(0, data_size_bytes)

    # Primary pattern: RCCL_{OP}-{bytes} byte processed {time}(us)
    # Try to match any size first, then filter by expected_bytes
    pattern_any_size = rf"{rccl_op_pattern}-(\d+)\s+byte\s+processed\s+([\d.]+)\(us\)"
    matches_any = re.findall(pattern_any_size, stdout_content, re.IGNORECASE)

    if matches_any:
        # Debug: print all matches found
        # print(f"DEBUG parse_rccl_logs: Found {len(matches_any)} matches for {rccl_op_pattern}, expected_bytes={expected_bytes}")
        for matched_bytes_str, matched_time_str in matches_any:
            matched_bytes = int(matched_bytes_str)
            # If we have expected_bytes, only match if it matches
            if expected_bytes > 0:
                if matched_bytes == expected_bytes:
                    times.append(float(matched_time_str) / 1_000_000)
                # Debug: print mismatches
                # else:
                #     print(f"DEBUG parse_rccl_logs: Mismatch - matched_bytes={matched_bytes}, expected_bytes={expected_bytes}")
            else:
                # If no expected_bytes (e.g., barrier), accept all matches
                times.append(float(matched_time_str) / 1_000_000)

        if times:
            return times

    # Fallback: try exact match with expected_bytes
    if expected_bytes > 0:
        pattern_exact = rf"{rccl_op_pattern}-{expected_bytes}\s+byte\s+processed\s+([\d.]+)\(us\)"
        matches_exact = re.findall(pattern_exact, stdout_content, re.IGNORECASE)
        if matches_exact:
            times = [float(match) / 1_000_000 for match in matches_exact]
            return times

    # Additional fallback patterns without data size matching
    fallback_patterns = [
        # Format: RCCL_SEND-2097152 byte processed 1265 us (without parentheses)
        rf"{rccl_op_pattern}-(\d+)\s+byte\s+processed\s+([\d.]+)\s*us",
        # Format: RCCL_SEND: 123.45 us
        rf"{rccl_op_pattern}:\s*([\d.]+)\s*us",
        # Format: [RCCL_SEND] time=123.45 us
        rf"\[{rccl_op_pattern}\]\s*time\s*=\s*([\d.]+)\s*us",
        # Format: RCCL_SEND time: 123.45 us
        rf"{rccl_op_pattern}\s+time\s*:\s*([\d.]+)\s*us",
        # Format: RCCL_SEND 123.45 us
        rf"{rccl_op_pattern}\s+([\d.]+)\s*us",
    ]

    for pattern in fallback_patterns:
        matches = re.findall(pattern, stdout_content, re.IGNORECASE)
        if matches:
            # For patterns with bytes, filter by expected_bytes if available
            if len(matches[0]) == 2:  # Pattern with bytes and time
                for matched_bytes_str, matched_time_str in matches:
                    matched_bytes = int(matched_bytes_str)
                    if expected_bytes == 0 or matched_bytes == expected_bytes:
                        times.append(float(matched_time_str) / 1_000_000)
            else:  # Pattern with only time
                times = [float(match) / 1_000_000 for match in matches]

            if times:
                break

    return times


@contextlib.contextmanager
def stdout_stderr_redirect(rank: int):
    """Context manager for redirecting stdout and stderr to temporary files."""
    stdout_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=f"_rank{rank}_stdout.txt")
    stderr_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=f"_rank{rank}_stderr.txt")
    stdout_file.close()
    stderr_file.close()

    stdout_fd = os.open(stdout_file.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    stderr_fd = os.open(stderr_file.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    try:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        yield stdout_file.name, stderr_file.name
    finally:
        # Restore file descriptors
        try:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
        except (OSError, ValueError):
            # If restoration fails, try to restore to /dev/null to prevent crashes
            try:
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                os.close(devnull_fd)
            except Exception:
                pass

        # Close file descriptors
        try:
            os.close(stdout_fd)
            os.close(stderr_fd)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        except (OSError, ValueError):
            pass


def run_benchmark_with_cleanup(
    rank: int,
    world_size: int,
    backend: str,
    measure_rccl: bool,
    op_func: Callable[[], None],
    op_name: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
) -> Tuple[List[float], List[float]]:
    """Common benchmark execution logic with cleanup.

    Args:
        rank: Process rank
        world_size: World size
        backend: Distributed backend
        measure_rccl: Whether to measure RCCL API time
        op_func: Function that executes the operation (called in warmup and measurement loops)
        op_name: Operation name for logging
        data_size_bytes: Data size in bytes for RCCL log parsing
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        Tuple of (iteration_times, rccl_times)
    """
    setup_environment(rank, world_size, enable_rccl_perf=measure_rccl)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        if measure_rccl:
            # Phase: RCCL API measurement (with RCCL_PERF=2 and log capture)
            with stdout_stderr_redirect(rank) as (stdout_file, stderr_file):
                # Warmup
                for _ in range(num_warmup):
                    op_func()

            dist.barrier()

            # Clear files and measure
            with stdout_stderr_redirect(rank) as (stdout_file, stderr_file):
                for _ in range(num_iterations):
                    op_func()

            # Read logs
            with open(stdout_file) as f:
                stdout_content = f.read()
            with open(stderr_file) as f:
                stderr_content = f.read()

            # Debug: print actual RCCL logs for debugging
            if rank == 0 and stdout_content.strip():
                print(f"DEBUG [{op_name}] stdout content (first 500 chars): {stdout_content[:500]}")
            if rank == 0 and stderr_content.strip():
                print(f"DEBUG [{op_name}] stderr content (first 500 chars): {stderr_content[:500]}")

            # Parse RCCL logs
            rccl_times_stdout = parse_rccl_logs(stdout_content, op_name, data_size_bytes)
            rccl_times_stderr = parse_rccl_logs(stderr_content, op_name, data_size_bytes)

            if not rccl_times_stdout and rccl_times_stderr:
                rccl_times = rccl_times_stderr
            else:
                rccl_times = rccl_times_stdout

            # Cleanup temp files
            try:
                os.unlink(stdout_file)
                os.unlink(stderr_file)
            except Exception:
                pass

            iteration_times = []
        else:
            # Phase: Python API measurement (no RCCL_PERF, no log capture)
            for _ in range(num_warmup):
                op_func()

            dist.barrier()

            iteration_times = []
            for _ in range(num_iterations):
                iter_start = time.perf_counter()
                op_func()
                iter_end = time.perf_counter()
                iteration_times.append(iter_end - iter_start)

            rccl_times = []
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return iteration_times, rccl_times


def benchmark_allreduce(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark allreduce operation."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements = data_size_bytes // dtype_size

    def op_func():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    tensor = torch.full([num_elements], float(rank + 1), dtype=dtype, device=device)

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, "allreduce", data_size_bytes, num_warmup, num_iterations
    )


def benchmark_broadcast(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    root: int = 0,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark broadcast operation."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements = data_size_bytes // dtype_size

    def op_func():
        dist.broadcast(tensor, src=root)

    if rank == root:
        tensor = torch.full([num_elements], 42.0, dtype=dtype, device=device)
    else:
        tensor = torch.zeros([num_elements], dtype=dtype, device=device)

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, "broadcast", data_size_bytes, num_warmup, num_iterations
    )


def benchmark_scatter(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    root: int = 0,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark scatter operation."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements = data_size_bytes // dtype_size

    def op_func():
        dist.scatter(output, input_list, src=root)

    # Create input data for scatter
    if rank == root:
        input_list = []
        for i in range(world_size):
            tensor = torch.full([num_elements], float(i + 1), dtype=dtype, device=device)
            input_list.append(tensor)
    else:
        input_list = None

    # Create output tensor
    output = torch.zeros([num_elements], dtype=dtype, device=device)

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, "scatter", data_size_bytes, num_warmup, num_iterations
    )


def benchmark_allgather(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark allgather operation."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements = data_size_bytes // dtype_size

    def op_func():
        dist.all_gather(output_list, input_tensor)

    # Create input tensor
    input_tensor = torch.full([num_elements], float(rank + 1), dtype=dtype, device=device)

    # Create output list
    output_list = []
    for _ in range(world_size):
        output_tensor = torch.zeros([num_elements], dtype=dtype, device=device)
        output_list.append(output_tensor)

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, "allgather", data_size_bytes, num_warmup, num_iterations
    )


def benchmark_reduce_scatter(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark reduce_scatter operation."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements_per_rank = data_size_bytes // dtype_size // world_size

    def op_func():
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)

    # Create input list (each rank sends a list of tensors of size num_elements_per_rank)
    input_list = []
    for _ in range(world_size):
        input_tensor = torch.full([num_elements_per_rank], float(rank + 1), dtype=dtype, device=device)
        input_list.append(input_tensor)

    # Create output tensor (each rank receives a tensor of size num_elements_per_rank)
    output_tensor = torch.zeros([num_elements_per_rank], dtype=dtype, device=device)

    # For reduce_scatter, RCCL logs show the per-rank data size
    per_rank_data_size_bytes = num_elements_per_rank * dtype_size

    iteration_times, rccl_times = run_benchmark_with_cleanup(
        rank,
        world_size,
        backend,
        measure_rccl,
        op_func,
        "reduce_scatter",
        per_rank_data_size_bytes,
        num_warmup,
        num_iterations,
    )

    return iteration_times, rccl_times


def benchmark_barrier(
    rank: int, world_size: int, backend: str, num_warmup: int, num_iterations: int, measure_rccl: bool = False
) -> Tuple[List[float], List[float]]:
    """Benchmark barrier operation."""

    def op_func():
        dist.barrier()

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, "barrier", 0, num_warmup, num_iterations
    )


def benchmark_sendrecv(
    rank: int,
    world_size: int,
    backend: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    dtype: torch.dtype = torch.float16,
    measure_rccl: bool = False,
) -> Tuple[List[float], List[float]]:
    """Benchmark send/recv operation. Rank 0 performs send, rank 1 performs recv."""
    device = torch.device("rbln", rank)
    dtype_size = get_dtype_size(dtype)
    num_elements = data_size_bytes // dtype_size

    if rank == 0:

        def op_func():
            dist.send(tensor, dst=1)

        tensor = torch.full([num_elements], float(rank + 1), dtype=dtype, device=device)
        op_name = "send"
    elif rank == 1:

        def op_func():
            dist.recv(tensor, src=0)

        tensor = torch.zeros([num_elements], dtype=dtype, device=device)
        op_name = "recv"
    else:
        # Other ranks don't participate
        return [], []

    return run_benchmark_with_cleanup(
        rank, world_size, backend, measure_rccl, op_func, op_name, data_size_bytes, num_warmup, num_iterations
    )


def run_benchmark(
    rank: int,
    world_size: int,
    backend: str,
    op_name: str,
    data_size_bytes: int,
    num_warmup: int,
    num_iterations: int,
    results: List,
    dtype: torch.dtype = torch.float16,
    peer_rank: Optional[int] = None,
    measure_rccl: bool = False,
) -> None:
    """Run benchmark for a specific operation."""
    try:
        if op_name == "allreduce":
            iteration_times, rccl_times = benchmark_allreduce(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, measure_rccl
            )
        elif op_name == "broadcast":
            iteration_times, rccl_times = benchmark_broadcast(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, 0, measure_rccl
            )
        elif op_name == "scatter":
            iteration_times, rccl_times = benchmark_scatter(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, 0, measure_rccl
            )
        elif op_name == "allgather":
            iteration_times, rccl_times = benchmark_allgather(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, measure_rccl
            )
        elif op_name == "reduce_scatter":
            iteration_times, rccl_times = benchmark_reduce_scatter(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, measure_rccl
            )
        elif op_name == "barrier":
            iteration_times, rccl_times = benchmark_barrier(
                rank, world_size, backend, num_warmup, num_iterations, measure_rccl
            )
        elif op_name == "sendrecv":
            iteration_times, rccl_times = benchmark_sendrecv(
                rank, world_size, backend, data_size_bytes, num_warmup, num_iterations, dtype, measure_rccl
            )
        else:
            raise ValueError(f"Unknown operation: {op_name}")

        # Calculate avg/min/max from iteration times
        if iteration_times:
            avg_time = sum(iteration_times) / len(iteration_times)
            min_time = min(iteration_times)
            max_time = max(iteration_times)
        else:
            avg_time = min_time = max_time = 0.0

        # Calculate avg/min/max from RCCL times
        if rccl_times:
            rccl_avg_time = sum(rccl_times) / len(rccl_times)
            rccl_min_time = min(rccl_times)
            rccl_max_time = max(rccl_times)
        else:
            rccl_avg_time = rccl_min_time = rccl_max_time = 0.0

        # Store result for this rank (including dtype information)
        # For sendrecv: rank 0 measures send, rank 1 measures recv
        # Store rank 0's result as "send", rank 1's result as "recv"
        if op_name == "sendrecv":
            if (iteration_times or rccl_times) and rank == 0:
                # Rank 0: store as "send"
                results.append(
                    (
                        "send",
                        data_size_bytes,
                        rank,
                        avg_time,
                        min_time,
                        max_time,
                        rccl_avg_time,
                        rccl_min_time,
                        rccl_max_time,
                        dtype,
                    )
                )
                dtype_size = get_dtype_size(dtype)
                num_elements = data_size_bytes // dtype_size
                mode_str = "RCCL API" if measure_rccl else "Python API"
                if measure_rccl:
                    # When measuring RCCL API, show RCCL times
                    print(
                        f"{'send':12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"rank={rank} | avg={rccl_avg_time * 1000000:.4f} us | "
                        f"min={rccl_min_time * 1000000:.4f} us | max={rccl_max_time * 1000000:.4f} us"
                    )
                else:
                    # When measuring Python API, show Python times
                    print(
                        f"{'send':12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"rank={rank} | avg={avg_time * 1000000:.4f} us | "
                        f"min={min_time * 1000000:.4f} us | max={max_time * 1000000:.4f} us"
                    )
            elif (iteration_times or rccl_times) and rank == 1:
                # Rank 1: store as "recv"
                results.append(
                    (
                        "recv",
                        data_size_bytes,
                        rank,
                        avg_time,
                        min_time,
                        max_time,
                        rccl_avg_time,
                        rccl_min_time,
                        rccl_max_time,
                        dtype,
                    )
                )
                dtype_size = get_dtype_size(dtype)
                num_elements = data_size_bytes // dtype_size
                mode_str = "RCCL API" if measure_rccl else "Python API"
                if measure_rccl:
                    # When measuring RCCL API, show RCCL times
                    print(
                        f"{'recv':12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"rank={rank} | avg={rccl_avg_time * 1000000:.4f} us | "
                        f"min={rccl_min_time * 1000000:.4f} us | max={rccl_max_time * 1000000:.4f} us"
                    )
                else:
                    # When measuring Python API, show Python times
                    print(
                        f"{'recv':12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"rank={rank} | avg={avg_time * 1000000:.4f} us | "
                        f"min={min_time * 1000000:.4f} us | max={max_time * 1000000:.4f} us"
                    )
        else:
            results.append(
                (
                    op_name,
                    data_size_bytes,
                    rank,
                    avg_time,
                    min_time,
                    max_time,
                    rccl_avg_time,
                    rccl_min_time,
                    rccl_max_time,
                    dtype,
                )
            )
            if rank == 0:
                dtype_size = get_dtype_size(dtype)
                num_elements = data_size_bytes // dtype_size
                mode_str = "RCCL API" if measure_rccl else "Python API"
                if measure_rccl:
                    # When measuring RCCL API, show RCCL times
                    print(
                        f"{op_name:12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"avg={rccl_avg_time * 1000000:.4f} us | min={rccl_min_time * 1000000:.4f} us | "
                        f"max={rccl_max_time * 1000000:.4f} us"
                    )
                else:
                    # When measuring Python API, show Python times
                    print(
                        f"{op_name:12s} | [{mode_str}] | size={data_size_bytes:6d} bytes ({num_elements} elements) | "
                        f"avg={avg_time * 1000000:.4f} us | min={min_time * 1000000:.4f} us | "
                        f"max={max_time * 1000000:.4f} us"
                    )

    except Exception as e:
        print(f"Error benchmarking {op_name} with size {data_size_bytes} bytes on rank {rank}: {e}")


def save_results_to_file(
    results: List[Tuple[str, int, int, float, float, float, float, float, float, torch.dtype]],
    ops: List[str],
    sizes: List[int],
    world_size: int,
    size_map: Dict[int, str],
    num_warmup: int,
    num_iterations: int,
    verbose: bool = False,
) -> Tuple[str, str]:
    """Save benchmark results to txt and md files.

    Results tuple format: (op_name, data_size_bytes, rank, avg_time, min_time, max_time,
                           rccl_avg_time, rccl_min_time, rccl_max_time, dtype)

    Returns:
        Tuple of (txt_file_path, md_file_path)
    """
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_txt = f"benchmark_results_{timestamp}.txt"
    output_file_md = f"benchmark_results_{timestamp}.md"

    # Collect system information
    system_info = get_system_info()

    with open(output_file_txt, "w") as f:
        # Write header
        f.write("=" * 120 + "\n")
        f.write("Collective Communication Operations Benchmark Results\n")
        f.write("=" * 120 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"World Size: {world_size}\n")
        f.write("\n")
        f.write("System Information:\n")
        f.write(f"  CPU Model: {system_info.get('cpu_model', 'Unknown')}\n")
        f.write(f"  CPU Count: {system_info.get('cpu_count', 'Unknown')}\n")
        f.write(f"  RAM Size: {system_info.get('ram_size_gb', 'Unknown')} GB\n")
        f.write(f"  NPU Count: {system_info.get('npu_count', 'Unknown')}\n")
        f.write("\n")
        f.write("Test Config:\n")
        f.write(f"  Num Warmup: {num_warmup}\n")
        f.write(f"  Num Iterations: {num_iterations}\n")
        f.write(f"  Operations: {', '.join(ops)}\n")
        f.write(f"  Sizes: {', '.join([size_map.get(s, str(s)) for s in sizes])}\n")
        f.write("=" * 120 + "\n\n")

        # Group results by operation, data size, rank, and dtype
        # Merge Phase 1 (Python API) and Phase 2 (RCCL API) results into same row
        results_by_op_size_rank_dtype = {}
        for (
            op_name,
            data_size_bytes,
            rank,
            avg_time,
            min_time,
            max_time,
            rccl_avg_time,
            rccl_min_time,
            rccl_max_time,
            dtype,
        ) in results:
            key = (op_name, data_size_bytes, rank, dtype)
            if key not in results_by_op_size_rank_dtype:
                # Initialize with zeros
                results_by_op_size_rank_dtype[key] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # Merge: take non-zero values (Phase 1 has Python times, Phase 2 has RCCL times)
            (
                existing_py_avg,
                existing_py_min,
                existing_py_max,
                existing_rccl_avg,
                existing_rccl_min,
                existing_rccl_max,
            ) = results_by_op_size_rank_dtype[key]

            # Update Python times if non-zero (from Phase 1)
            if avg_time > 0:
                existing_py_avg = avg_time
                existing_py_min = min_time
                existing_py_max = max_time

            # Update RCCL times if non-zero (from Phase 2)
            if rccl_avg_time > 0:
                existing_rccl_avg = rccl_avg_time
                existing_rccl_min = rccl_min_time
                existing_rccl_max = rccl_max_time

            results_by_op_size_rank_dtype[key] = (
                existing_py_avg,
                existing_py_min,
                existing_py_max,
                existing_rccl_avg,
                existing_rccl_min,
                existing_rccl_max,
            )

        # Convert to the format expected by the rest of the code
        results_by_op_size_dtype = {}
        for (op_name, data_size_bytes, rank, dtype), (
            py_avg,
            py_min,
            py_max,
            rccl_avg,
            rccl_min,
            rccl_max,
        ) in results_by_op_size_rank_dtype.items():
            key = (op_name, data_size_bytes, dtype)
            if key not in results_by_op_size_dtype:
                results_by_op_size_dtype[key] = []
            results_by_op_size_dtype[key].append((rank, py_avg, py_min, py_max, rccl_avg, rccl_min, rccl_max))

        if verbose:
            # Write results for each operation
            for op_name in ops:
                f.write(f"\n{'=' * 120}\n")
                f.write(f"Operation: {op_name}\n")
                f.write(f"{'=' * 120}\n")

                if op_name == "barrier":
                    # Barrier doesn't use data size or dtype
                    barrier_keys = [k for k in results_by_op_size_dtype.keys() if k[0] == op_name and k[1] == 0]
                    if barrier_keys:
                        key = barrier_keys[0]  # Use first barrier key found
                        header = (
                            f"\n{'Rank':<8} | {'Python Avg (us)':<18} | {'Python Min (us)':<18} | "
                            f"{'Python Max (us)':<18} | {'RCCL Avg (us)':<18} | "
                            f"{'RCCL Min (us)':<18} | {'RCCL Max (us)':<18}\n"
                        )
                        f.write(header)
                        f.write("-" * 120 + "\n")
                        for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                            results_by_op_size_dtype[key]
                        ):
                            row = (
                                f"{rank:<8} | {avg_time * 1000000:<18.4f} | "
                                f"{min_time * 1000000:<18.4f} | {max_time * 1000000:<18.4f} | "
                                f"{rccl_avg_time * 1000000:<18.4f} | {rccl_min_time * 1000000:<18.4f} | "
                                f"{rccl_max_time * 1000000:<18.4f}\n"
                            )
                            f.write(row)
                else:
                    # Group by dtype first, then by size
                    dtypes_for_op = sorted({dtype for op, _, dtype in results_by_op_size_dtype.keys() if op == op_name})
                    for dtype in dtypes_for_op:
                        dtype_size = get_dtype_size(dtype)
                        f.write(f"\nDtype: {dtype} ({dtype_size} bytes per element)\n")
                        for data_size_bytes in sizes:
                            key = (op_name, data_size_bytes, dtype)
                            if key not in results_by_op_size_dtype:
                                continue

                            num_elements = data_size_bytes // dtype_size
                            size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                            f.write(f"\nData Size: {size_str} ({data_size_bytes} bytes, {num_elements} elements)\n")
                            header = (
                                f"{'Rank':<8} | {'Python Avg (us)':<18} | {'Python Min (us)':<18} | "
                                f"{'Python Max (us)':<18} | {'RCCL Avg (us)':<18} | "
                                f"{'RCCL Min (us)':<18} | {'RCCL Max (us)':<18}\n"
                            )
                            f.write(header)
                            f.write("-" * 120 + "\n")

                            for (
                                rank,
                                avg_time,
                                min_time,
                                max_time,
                                rccl_avg_time,
                                rccl_min_time,
                                rccl_max_time,
                            ) in sorted(results_by_op_size_dtype[key]):
                                row = (
                                    f"{rank:<8} | {avg_time * 1000000:<18.4f} | "
                                    f"{min_time * 1000000:<18.4f} | {max_time * 1000000:<18.4f} | "
                                    f"{rccl_avg_time * 1000000:<18.4f} | {rccl_min_time * 1000000:<18.4f} | "
                                    f"{rccl_max_time * 1000000:<18.4f}\n"
                                )
                                f.write(row)

        f.write("\n" + "=" * 150 + "\n")
        f.write("Summary Table\n")
        f.write("=" * 150 + "\n")
        header = (
            f"{'Operation':<12} | {'Dtype':<15} | {'Size':<20} | {'Rank':<6} | "
            f"{'Python Avg (us)':<18} | {'Python Min (us)':<18} | {'Python Max (us)':<18} | "
            f"{'RCCL Avg (us)':<18} | {'RCCL Min (us)':<18} | {'RCCL Max (us)':<18}\n"
        )
        f.write(header)
        f.write("-" * 150 + "\n")

        for op_name in ops:
            if op_name == "barrier":
                barrier_keys = [k for k in results_by_op_size_dtype.keys() if k[0] == op_name and k[1] == 0]
                if barrier_keys:
                    key = barrier_keys[0]
                    for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                        results_by_op_size_dtype[key]
                    ):
                        row = (
                            f"{op_name:<12} | {'N/A':<15} | {'N/A':<20} | {rank:<6} | "
                            f"{avg_time * 1000000:<18.4f} | {min_time * 1000000:<18.4f} | "
                            f"{max_time * 1000000:<18.4f} | {rccl_avg_time * 1000000:<18.4f} | "
                            f"{rccl_min_time * 1000000:<18.4f} | {rccl_max_time * 1000000:<18.4f}\n"
                        )
                        f.write(row)
            else:
                # Group by dtype first
                dtypes_for_op = sorted({dtype for op, _, dtype in results_by_op_size_dtype.keys() if op == op_name})
                for dtype in dtypes_for_op:
                    dtype_size = get_dtype_size(dtype)
                    for data_size_bytes in sizes:
                        key = (op_name, data_size_bytes, dtype)
                        if key not in results_by_op_size_dtype:
                            continue
                        num_elements = data_size_bytes // dtype_size
                        size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                        size_display = f"{size_str} ({num_elements} elem)"

                        for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                            results_by_op_size_dtype[key]
                        ):
                            row = (
                                f"{op_name:<12} | {str(dtype):<15} | {size_display:<20} | {rank:<6} | "
                                f"{avg_time * 1000000:<18.4f} | {min_time * 1000000:<18.4f} | "
                                f"{max_time * 1000000:<18.4f} | {rccl_avg_time * 1000000:<18.4f} | "
                                f"{rccl_min_time * 1000000:<18.4f} | {rccl_max_time * 1000000:<18.4f}\n"
                            )
                            f.write(row)

    # Write markdown file
    with open(output_file_md, "w") as f:
        # Write header
        f.write("# Collective Communication Operations Benchmark Results\n\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**World Size:** {world_size}\n\n")
        f.write("## System Information\n\n")
        f.write(f"- **CPU Model:** {system_info.get('cpu_model', 'Unknown')}\n")
        f.write(f"- **CPU Count:** {system_info.get('cpu_count', 'Unknown')}\n")
        f.write(f"- **RAM Size:** {system_info.get('ram_size_gb', 'Unknown')} GB\n")
        f.write(f"- **NPU Count:** {system_info.get('npu_count', 'Unknown')}\n\n")
        f.write("## Test Config\n\n")
        f.write(f"- **Num Warmup:** {num_warmup}\n")
        f.write(f"- **Num Iterations:** {num_iterations}\n")
        f.write(f"- **Operations:** {', '.join(ops)}\n")
        f.write(f"- **Sizes:** {', '.join([size_map.get(s, str(s)) for s in sizes])}\n\n")
        f.write("---\n\n")

        # Write summary table in markdown format
        f.write("## Summary Table\n\n")
        header = (
            "| Operation | Dtype | Size | Rank | Python Avg (us) | Python Min (us) | "
            "Python Max (us) | RCCL Avg (us) | RCCL Min (us) | RCCL Max (us) |\n"
        )
        f.write(header)
        separator = (
            "|-----------|-------|------|------|-----------------|----------------|"
            "-----------------|---------------|---------------|---------------|\n"
        )
        f.write(separator)

        for op_name in ops:
            if op_name == "barrier":
                barrier_keys = [k for k in results_by_op_size_dtype.keys() if k[0] == op_name and k[1] == 0]
                if barrier_keys:
                    key = barrier_keys[0]
                    for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                        results_by_op_size_dtype[key]
                    ):
                        row = (
                            f"| {op_name} | N/A | N/A | {rank} | {avg_time * 1000000:.4f} | "
                            f"{min_time * 1000000:.4f} | {max_time * 1000000:.4f} | "
                            f"{rccl_avg_time * 1000000:.4f} | {rccl_min_time * 1000000:.4f} | "
                            f"{rccl_max_time * 1000000:.4f} |\n"
                        )
                        f.write(row)
            else:
                # Group by dtype first
                dtypes_for_op = sorted({dtype for op, _, dtype in results_by_op_size_dtype.keys() if op == op_name})
                for dtype in dtypes_for_op:
                    dtype_size = get_dtype_size(dtype)
                    for data_size_bytes in sizes:
                        key = (op_name, data_size_bytes, dtype)
                        if key not in results_by_op_size_dtype:
                            continue
                        num_elements = data_size_bytes // dtype_size
                        size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                        size_display = f"{size_str} ({num_elements} elem)"

                        for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                            results_by_op_size_dtype[key]
                        ):
                            row = (
                                f"| {op_name} | {str(dtype)} | {size_display} | {rank} | "
                                f"{avg_time * 1000000:.4f} | {min_time * 1000000:.4f} | "
                                f"{max_time * 1000000:.4f} | {rccl_avg_time * 1000000:.4f} | "
                                f"{rccl_min_time * 1000000:.4f} | {rccl_max_time * 1000000:.4f} |\n"
                            )
                            f.write(row)

        # Write detailed results if verbose
        if verbose:
            f.write("\n---\n\n")
            f.write("## Detailed Results\n\n")
            for op_name in ops:
                f.write(f"### {op_name}\n\n")

                if op_name == "barrier":
                    barrier_keys = [k for k in results_by_op_size_dtype.keys() if k[0] == op_name and k[1] == 0]
                    if barrier_keys:
                        key = barrier_keys[0]
                        header = (
                            "| Rank | Python Avg (us) | Python Min (us) | Python Max (us) | "
                            "RCCL Avg (us) | RCCL Min (us) | RCCL Max (us) |\n"
                        )
                        f.write(header)
                        separator = (
                            "|------|-----------------|----------------|-----------------|"
                            "---------------|---------------|---------------|\n"
                        )
                        f.write(separator)
                        for rank, avg_time, min_time, max_time, rccl_avg_time, rccl_min_time, rccl_max_time in sorted(
                            results_by_op_size_dtype[key]
                        ):
                            row = (
                                f"| {rank} | {avg_time * 1000000:.4f} | {min_time * 1000000:.4f} | "
                                f"{max_time * 1000000:.4f} | {rccl_avg_time * 1000000:.4f} | "
                                f"{rccl_min_time * 1000000:.4f} | {rccl_max_time * 1000000:.4f} |\n"
                            )
                            f.write(row)
                        f.write("\n")
                else:
                    # Group by dtype first, then by size
                    dtypes_for_op = sorted({dtype for op, _, dtype in results_by_op_size_dtype.keys() if op == op_name})
                    for dtype in dtypes_for_op:
                        dtype_size = get_dtype_size(dtype)
                        f.write(f"#### Dtype: {dtype} ({dtype_size} bytes per element)\n\n")
                        for data_size_bytes in sizes:
                            key = (op_name, data_size_bytes, dtype)
                            if key not in results_by_op_size_dtype:
                                continue

                            num_elements = data_size_bytes // dtype_size
                            size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                            f.write(f"**Data Size:** {size_str} ({data_size_bytes} bytes, {num_elements} elements)\n\n")
                            header = (
                                "| Rank | Python Avg (us) | Python Min (us) | Python Max (us) | "
                                "RCCL Avg (us) | RCCL Min (us) | RCCL Max (us) |\n"
                            )
                            f.write(header)
                            separator = (
                                "|------|-----------------|----------------|-----------------|"
                                "---------------|---------------|---------------|\n"
                            )
                            f.write(separator)

                            for (
                                rank,
                                avg_time,
                                min_time,
                                max_time,
                                rccl_avg_time,
                                rccl_min_time,
                                rccl_max_time,
                            ) in sorted(results_by_op_size_dtype[key]):
                                row = (
                                    f"| {rank} | {avg_time * 1000000:.4f} | {min_time * 1000000:.4f} | "
                                    f"{max_time * 1000000:.4f} | {rccl_avg_time * 1000000:.4f} | "
                                    f"{rccl_min_time * 1000000:.4f} | {rccl_max_time * 1000000:.4f} |\n"
                                )
                                f.write(row)

                            # Calculate average across ranks (skip for point-to-point ops)
                            if op_name not in ["send", "recv"]:
                                avg_avg = sum(avg for _, avg, _, _, _, _, _ in results_by_op_size_dtype[key]) / len(
                                    results_by_op_size_dtype[key]
                                )
                                avg_min = sum(min_t for _, _, min_t, _, _, _, _ in results_by_op_size_dtype[key]) / len(
                                    results_by_op_size_dtype[key]
                                )
                                avg_max = sum(max_t for _, _, _, max_t, _, _, _ in results_by_op_size_dtype[key]) / len(
                                    results_by_op_size_dtype[key]
                                )
                                rccl_avg_avg = sum(
                                    rccl_avg for _, _, _, _, rccl_avg, _, _ in results_by_op_size_dtype[key]
                                ) / len(results_by_op_size_dtype[key])
                                rccl_avg_min = sum(
                                    rccl_min for _, _, _, _, _, rccl_min, _ in results_by_op_size_dtype[key]
                                ) / len(results_by_op_size_dtype[key])
                                rccl_avg_max = sum(
                                    rccl_max for _, _, _, _, _, _, rccl_max in results_by_op_size_dtype[key]
                                ) / len(results_by_op_size_dtype[key])
                                avg_row = (
                                    f"| **Average** | {avg_avg * 1000000:.4f} | {avg_min * 1000000:.4f} | "
                                    f"{avg_max * 1000000:.4f} | {rccl_avg_avg * 1000000:.4f} | "
                                    f"{rccl_avg_min * 1000000:.4f} | {rccl_avg_max * 1000000:.4f} |\n"
                                )
                                f.write(avg_row)

                            f.write("\n")

    return output_file_txt, output_file_md


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark collective communication operations")
    parser.add_argument(
        "--world-size", type=int, default=None, help="Number of processes (default: number of RBLN devices)"
    )
    parser.add_argument("--backend", type=str, default="rbln-ccl", help="Distributed backend")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument(
        "--ops",
        type=str,
        nargs="+",
        default=["allreduce", "broadcast", "scatter", "allgather", "reduce_scatter", "barrier", "send", "recv"],
        help="Operations to benchmark",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=["4096", "32768", "65536", "131072", "524288", "1048576"],
        help="Data sizes to benchmark (in bytes). Can be space-separated or comma-separated.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8"],
        help="Data type (default: float16)",
    )

    args = parser.parse_args(argv)

    # Parse sizes: handle comma-separated values
    sizes_list = []
    for size_arg in args.sizes:
        # Split by comma and strip whitespace
        for size_str in size_arg.split(","):
            size_str = size_str.strip()
            if size_str:
                try:
                    sizes_list.append(int(size_str))
                except ValueError:
                    parser.error(f"Invalid size value: {size_str}. Must be an integer.")
    args.sizes = sizes_list

    # Convert dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    dtype = dtype_map[args.dtype]
    dtype_size = get_dtype_size(dtype)

    def get_world_size_for_op(op_name: str, user_world_size: Optional[int]) -> int:
        """Determine world_size for a specific operation.

        Args:
            op_name: Operation name (e.g., 'sendrecv', 'allreduce', 'barrier')
            user_world_size: User-specified world_size (None if not specified)

        Returns:
            world_size to use for this operation
        """
        if user_world_size is not None:
            return user_world_size

        if op_name == "sendrecv":
            # Send/recv operations use exactly 2 ranks
            if torch.rbln.device_count() < 2:
                raise RuntimeError("Send/recv operations require at least 2 RBLN devices")
            return 2
        else:
            # Other collective operations require at least 4 ranks
            world_size = min(torch.rbln.device_count(), 4)
            if world_size < 4:
                raise RuntimeError(f"{op_name} operation requires at least 4 RBLN devices (found {world_size})")
            return world_size

    # Data size mapping for human-readable output
    size_map = {
        4096: "4k",
        32768: "32k",
        65536: "64k",
        131072: "128k",
        524288: "512k",
        1048576: "1m",
    }

    print("=" * 120)
    print("Collective Communication Operations Benchmark")
    print("=" * 120)
    print(f"Backend: {args.backend}")
    print(f"Dtype: {args.dtype} ({dtype_size} bytes per element)")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")
    print(f"Operations: {', '.join(args.ops)}")
    print(f"Data sizes (bytes): {[size_map.get(s, str(s)) for s in args.sizes]}")
    elements_list = [str(s // dtype_size) for s in args.sizes]
    print(f"Data sizes (elements): {elements_list}")
    print("Note: send/recv operations use world_size=2, other operations use world_size=4")
    print("=" * 120)
    print()

    # Shared list to collect results from all ranks
    manager = mp.Manager()
    results = manager.list()

    # Process ops: if send or recv is in args.ops, replace with sendrecv
    processed_ops = []
    has_send_or_recv = False
    for op_name in args.ops:
        if op_name in ["send", "recv"]:
            if not has_send_or_recv:
                processed_ops.append("sendrecv")
                has_send_or_recv = True
            # Skip individual send/recv, we'll use sendrecv instead
        else:
            processed_ops.append(op_name)

    # Phase 1: Python API measurement (RCCL_PERF unset)
    print("\n" + "=" * 120)
    print("Phase 1: Python API Measurement (RCCL_PERF unset)")
    print("=" * 120)

    for op_name in processed_ops:
        # Get world_size for this specific operation
        world_size = get_world_size_for_op(op_name, args.world_size)

        if op_name == "barrier":
            # Barrier doesn't use data size
            print(f"\nBenchmarking {op_name}...")
            mp.spawn(
                run_benchmark,
                args=(
                    world_size,
                    args.backend,
                    op_name,
                    0,
                    args.num_warmup,
                    args.num_iterations,
                    results,
                    dtype,
                    None,
                    False,
                ),
                nprocs=world_size,
                join=True,
            )
        elif op_name == "sendrecv":
            for data_size_bytes in args.sizes:
                num_elements = data_size_bytes // dtype_size
                size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                print(
                    f"\nBenchmarking sendrecv with size {size_str} "
                    f"({data_size_bytes} bytes, {num_elements} elements)..."
                )
                mp.spawn(
                    run_benchmark,
                    args=(
                        world_size,
                        args.backend,
                        op_name,
                        data_size_bytes,
                        args.num_warmup,
                        args.num_iterations,
                        results,
                        dtype,
                        None,
                        False,
                    ),
                    nprocs=world_size,
                    join=True,
                )
        else:
            for data_size_bytes in args.sizes:
                num_elements = data_size_bytes // dtype_size
                size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                print(
                    f"\nBenchmarking {op_name} with size {size_str} "
                    f"({data_size_bytes} bytes, {num_elements} elements)..."
                )
                mp.spawn(
                    run_benchmark,
                    args=(
                        world_size,
                        args.backend,
                        op_name,
                        data_size_bytes,
                        args.num_warmup,
                        args.num_iterations,
                        results,
                        dtype,
                        None,
                        False,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    # Phase 2: RCCL API measurement (RCCL_PERF=2)
    print("\n" + "=" * 120)
    print("Phase 2: RCCL API Measurement (RCCL_PERF=2)")
    print("=" * 120)

    for op_name in processed_ops:
        # Get world_size for this specific operation
        world_size = get_world_size_for_op(op_name, args.world_size)

        if op_name == "barrier":
            # Barrier doesn't use data size
            print(f"\nBenchmarking {op_name}...")
            mp.spawn(
                run_benchmark,
                args=(
                    world_size,
                    args.backend,
                    op_name,
                    0,
                    args.num_warmup,
                    args.num_iterations,
                    results,
                    dtype,
                    None,
                    True,
                ),
                nprocs=world_size,
                join=True,
            )
        elif op_name == "sendrecv":
            for data_size_bytes in args.sizes:
                num_elements = data_size_bytes // dtype_size
                size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                print(
                    f"\nBenchmarking sendrecv with size {size_str} "
                    f"({data_size_bytes} bytes, {num_elements} elements)..."
                )
                mp.spawn(
                    run_benchmark,
                    args=(
                        world_size,
                        args.backend,
                        op_name,
                        data_size_bytes,
                        args.num_warmup,
                        args.num_iterations,
                        results,
                        dtype,
                        None,
                        True,
                    ),
                    nprocs=world_size,
                    join=True,
                )
        else:
            for data_size_bytes in args.sizes:
                num_elements = data_size_bytes // dtype_size
                size_str = size_map.get(data_size_bytes, str(data_size_bytes))
                print(
                    f"\nBenchmarking {op_name} with size {size_str} "
                    f"({data_size_bytes} bytes, {num_elements} elements)..."
                )
                mp.spawn(
                    run_benchmark,
                    args=(
                        world_size,
                        args.backend,
                        op_name,
                        data_size_bytes,
                        args.num_warmup,
                        args.num_iterations,
                        results,
                        dtype,
                        None,
                        True,
                    ),
                    nprocs=world_size,
                    join=True,
                )

    # Save results to file
    # Calculate max world_size from results (since different operations may use different world_size)
    max_world_size = 0
    if results:
        # Results tuple format: (op_name, data_size_bytes, rank, ...)
        max_world_size = max(result[2] for result in results) + 1  # rank is 0-indexed
    else:
        # Fallback: use max world_size from all operations
        for op_name in processed_ops:
            op_world_size = get_world_size_for_op(op_name, args.world_size)
            max_world_size = max(max_world_size, op_world_size)

    output_file_txt, output_file_md = save_results_to_file(
        results, args.ops, args.sizes, max_world_size, size_map, args.num_warmup, args.num_iterations
    )
    print(f"\nResults saved to: {output_file_txt}")
    print(f"Results saved to (Markdown): {output_file_md}")


@pytest.mark.test_set_perf
@pytest.mark.single_worker
class TestBenchmarkCollectiveOps(TestCase):
    """Pytest wrapper so the benchmark can be invoked via `pytest -m "test_set_perf"`."""

    def test_benchmark(self):
        main([])


instantiate_device_type_tests(TestBenchmarkCollectiveOps, globals(), only_for="privateuse1")

if __name__ == "__main__":
    main()
