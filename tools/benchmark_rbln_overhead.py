#!/usr/bin/env python3
"""Benchmark host-side overhead for CPU vs RBLN elementwise add paths.

This tool is intentionally focused on call-path overhead, not end-to-end device
execution latency. Pair it with ``TORCH_RBLN_PROFILE=ON`` to get the internal
phase breakdown emitted by torch-rbln at process exit.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LatencyStats:
    first_ns: int
    mean_ns: int
    median_ns: int
    min_ns: int
    max_ns: int


class AddModule(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024, help="Number of rows for the benchmark tensor.")
    parser.add_argument("--cols", type=int, default=1024, help="Number of cols for the benchmark tensor.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of timed iterations per benchmark.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before the timed steady state.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Tensor dtype.")
    parser.add_argument("--device", default="rbln:0", help="Target RBLN device for eager/compiled paths.")
    parser.add_argument(
        "--skip-compiled",
        action="store_true",
        help="Skip the torch.compile benchmark and only run CPU/RBLN eager paths.",
    )
    return parser.parse_args()


def benchmark_callable(fn, *, iterations: int, warmup: int) -> LatencyStats:
    first_latency_ns = _run_once(fn)
    for _ in range(warmup):
        fn()
    latencies_ns = [_run_once(fn) for _ in range(iterations)]
    return LatencyStats(
        first_ns=first_latency_ns,
        mean_ns=int(statistics.fmean(latencies_ns)),
        median_ns=int(statistics.median(latencies_ns)),
        min_ns=min(latencies_ns),
        max_ns=max(latencies_ns),
    )


def _run_once(fn) -> int:
    start_ns = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - start_ns


def benchmark_compile_api(module: torch.nn.Module) -> tuple[torch.nn.Module, int]:
    start_ns = time.perf_counter_ns()
    compiled = torch.compile(module, backend="rbln", dynamic=False)
    return compiled, time.perf_counter_ns() - start_ns


def format_ns(duration_ns: int) -> str:
    if duration_ns >= 1_000_000_000:
        return f"{duration_ns / 1_000_000_000:.3f}s"
    if duration_ns >= 1_000_000:
        return f"{duration_ns / 1_000_000:.3f}ms"
    if duration_ns >= 1_000:
        return f"{duration_ns / 1_000:.3f}us"
    return f"{duration_ns}ns"


def print_stats(label: str, stats: LatencyStats) -> None:
    print(
        f"{label:<24} "
        f"first={format_ns(stats.first_ns):>10} "
        f"mean={format_ns(stats.mean_ns):>10} "
        f"median={format_ns(stats.median_ns):>10} "
        f"min={format_ns(stats.min_ns):>10} "
        f"max={format_ns(stats.max_ns):>10}"
    )


def main() -> None:
    args = parse_args()
    shape = (args.rows, args.cols)
    dtype = getattr(torch, args.dtype)
    profile_enabled = os.getenv("TORCH_RBLN_PROFILE", "").strip().lower() in {"1", "true", "on", "yes"}

    try:
        import torch_rbln  # noqa: F401

        from torch_rbln._internal.compile_cache import clear_rbln_compile_cache
    except Exception as error:
        raise RuntimeError(
            "Failed to import torch_rbln. Check the RBLN runtime environment first "
            "(for example: `python -m torch_rbln.diagnose`)."
        ) from error

    print("RBLN host-side overhead benchmark")
    print(f"  shape={shape} dtype={dtype} device={args.device}")
    print(
        f"  iterations={args.iterations} warmup={args.warmup} "
        f"profile_enabled={profile_enabled}"
    )
    if not profile_enabled:
        print("  hint=set TORCH_RBLN_PROFILE=ON before launch to emit the internal phase breakdown")

    lhs_cpu = torch.randn(shape, dtype=dtype, device="cpu")
    rhs_cpu = torch.randn(shape, dtype=dtype, device="cpu")
    lhs_rbln = lhs_cpu.to(args.device)
    rhs_rbln = rhs_cpu.to(args.device)

    cpu_stats = benchmark_callable(lambda: torch.add(lhs_cpu, rhs_cpu), iterations=args.iterations, warmup=args.warmup)
    print_stats("cpu torch.add", cpu_stats)

    torch._dynamo.reset()
    clear_rbln_compile_cache()
    eager_stats = benchmark_callable(
        lambda: torch.add(lhs_rbln, rhs_rbln),
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print_stats("rbln eager torch.add", eager_stats)
    print(
        f"{'rbln eager delta':<24} "
        f"first={format_ns(eager_stats.first_ns - cpu_stats.first_ns):>10} "
        f"mean={format_ns(eager_stats.mean_ns - cpu_stats.mean_ns):>10}"
    )

    if args.skip_compiled:
        return

    torch._dynamo.reset()
    clear_rbln_compile_cache()
    compiled_module, compile_api_ns = benchmark_compile_api(AddModule().eval())
    compiled_stats = benchmark_callable(
        lambda: compiled_module(lhs_rbln, rhs_rbln),
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print(f"{'rbln torch.compile()':<24} api_call={format_ns(compile_api_ns):>10}")
    print_stats("rbln compiled add", compiled_stats)
    print(
        f"{'rbln compiled delta':<24} "
        f"first={format_ns(compiled_stats.first_ns - cpu_stats.first_ns):>10} "
        f"mean={format_ns(compiled_stats.mean_ns - cpu_stats.mean_ns):>10}"
    )


if __name__ == "__main__":
    main()
