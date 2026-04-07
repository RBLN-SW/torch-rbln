"""Runtime overhead profiling utilities for torch-rbln.

This module is intentionally lightweight when profiling is disabled and only
activates the higher-overhead dispatch wrapping path when
``TORCH_RBLN_PROFILE`` is enabled before ``torch_rbln`` is imported.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping


_TRUTHY_ENV_VALUES = frozenset({"1", "true", "on", "yes"})
_PROFILE_ENV = "TORCH_RBLN_PROFILE"
_PROFILE_TOP_N_ENV = "TORCH_RBLN_PROFILE_TOPN"
_DEFAULT_TOP_N = 10


@dataclass
class _DurationStat:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0

    def add(self, duration_ns: int) -> None:
        self.count += 1
        self.total_ns += duration_ns
        self.max_ns = max(self.max_ns, duration_ns)


@dataclass(frozen=True)
class _CallContext:
    category: str
    name: str


class _ProfileStore:
    def __init__(self) -> None:
        self.call_totals: defaultdict[tuple[str, str], _DurationStat] = defaultdict(_DurationStat)
        self.phase_totals: defaultdict[str, _DurationStat] = defaultdict(_DurationStat)
        self.phase_by_context: defaultdict[tuple[str, str, str], _DurationStat] = defaultdict(_DurationStat)
        self.counter_totals: Counter[str] = Counter()
        self.counter_by_context: Counter[tuple[str, str, str]] = Counter()
        self.guard_reason_totals: Counter[str] = Counter()
        self.guard_reason_by_context: Counter[tuple[str, str, str]] = Counter()

    def reset(self) -> None:
        self.call_totals.clear()
        self.phase_totals.clear()
        self.phase_by_context.clear()
        self.counter_totals.clear()
        self.counter_by_context.clear()
        self.guard_reason_totals.clear()
        self.guard_reason_by_context.clear()


_PROFILE_STORE = _ProfileStore()
_PROFILE_LOCK = threading.Lock()
_PROFILE_TLS = threading.local()
_PROFILE_ATEXIT_REGISTERED = False
_PROFILE_SUMMARY_EMITTED = False
_PROFILE_PID = os.getpid()


def is_rbln_overhead_profiling_enabled() -> bool:
    return os.getenv(_PROFILE_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _profile_top_n() -> int:
    raw = os.getenv(_PROFILE_TOP_N_ENV)
    if raw is None:
        return _DEFAULT_TOP_N
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_TOP_N


def _ensure_atexit_reporter_registered() -> None:
    global _PROFILE_ATEXIT_REGISTERED
    _ensure_process_local_state()
    if _PROFILE_ATEXIT_REGISTERED or not is_rbln_overhead_profiling_enabled():
        return
    atexit.register(_emit_profile_summary_at_exit)
    _PROFILE_ATEXIT_REGISTERED = True


def _ensure_process_local_state() -> None:
    if os.getpid() != _PROFILE_PID:
        _reset_profiler_after_fork()


def _reset_profiler_after_fork() -> None:
    global _PROFILE_STORE, _PROFILE_LOCK, _PROFILE_TLS
    global _PROFILE_ATEXIT_REGISTERED, _PROFILE_SUMMARY_EMITTED, _PROFILE_PID

    _PROFILE_STORE = _ProfileStore()
    _PROFILE_LOCK = threading.Lock()
    _PROFILE_TLS = threading.local()
    _PROFILE_ATEXIT_REGISTERED = False
    _PROFILE_SUMMARY_EMITTED = False
    _PROFILE_PID = os.getpid()


def _get_context_stack() -> list[_CallContext]:
    stack = getattr(_PROFILE_TLS, "context_stack", None)
    if stack is None:
        stack = []
        _PROFILE_TLS.context_stack = stack
    return stack


@contextmanager
def profile_call_context(name: str, category: str, *, allow_nested: bool = True) -> Iterator[bool]:
    if not is_rbln_overhead_profiling_enabled():
        yield False
        return

    _ensure_atexit_reporter_registered()

    stack = _get_context_stack()
    if not allow_nested and any(ctx.category == category for ctx in stack):
        yield False
        return

    context = _CallContext(category=category, name=name)
    start_ns = time.perf_counter_ns()
    stack.append(context)
    try:
        yield True
    finally:
        duration_ns = time.perf_counter_ns() - start_ns
        stack.pop()
        _record_call_total(context.category, context.name, duration_ns)


@contextmanager
def profile_phase(name: str) -> Iterator[None]:
    if not is_rbln_overhead_profiling_enabled():
        yield
        return

    _ensure_atexit_reporter_registered()

    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        record_phase_duration(name, time.perf_counter_ns() - start_ns)


def _record_call_total(category: str, name: str, duration_ns: int) -> None:
    _ensure_process_local_state()
    with _PROFILE_LOCK:
        _PROFILE_STORE.call_totals[(category, name)].add(duration_ns)


def record_phase_duration(name: str, duration_ns: int) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_atexit_reporter_registered()

    contexts = tuple(_get_context_stack())
    with _PROFILE_LOCK:
        _PROFILE_STORE.phase_totals[name].add(duration_ns)
        for context in contexts:
            _PROFILE_STORE.phase_by_context[(context.category, context.name, name)].add(duration_ns)


def record_counter(name: str, delta: int = 1) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_atexit_reporter_registered()

    contexts = tuple(_get_context_stack())
    with _PROFILE_LOCK:
        _PROFILE_STORE.counter_totals[name] += delta
        for context in contexts:
            _PROFILE_STORE.counter_by_context[(context.category, context.name, name)] += delta


def record_guard_failure_reason(reason: str, delta: int = 1) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_atexit_reporter_registered()

    normalized = " ".join(str(reason).split())
    if not normalized:
        normalized = "<empty guard failure reason>"

    contexts = tuple(_get_context_stack())
    with _PROFILE_LOCK:
        _PROFILE_STORE.guard_reason_totals[normalized] += delta
        for context in contexts:
            _PROFILE_STORE.guard_reason_by_context[(context.category, context.name, normalized)] += delta


def wrap_registered_dispatch_functions(namespace: dict[str, Any], *, module_name: str) -> None:
    """Wrap locally-defined RBLN dispatch entries so phases can be attributed per op."""
    if not is_rbln_overhead_profiling_enabled():
        return

    for name, value in list(namespace.items()):
        if not _should_wrap_registered_dispatch(name, value, module_name):
            continue
        namespace[name] = _wrap_registered_dispatch(name, value)


def _should_wrap_registered_dispatch(name: str, value: Any, module_name: str) -> bool:
    if not inspect.isfunction(value):
        return False
    if value.__module__ != module_name:
        return False
    return "_rbln" in name


def _wrap_registered_dispatch(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with profile_call_context(name, "eager_op", allow_nested=False):
            return fn(*args, **kwargs)

    return wrapped


def snapshot_dynamo_state() -> dict[str, Any] | None:
    if not is_rbln_overhead_profiling_enabled():
        return None

    try:
        import torch._dynamo.utils as dynamo_utils
    except Exception:
        return None

    compile_metrics_ns = {
        name: int(sum(values) * 1_000_000_000)
        for name, values in dynamo_utils.compilation_time_metrics.items()
    }
    counter_values = dict(_flatten_mapping("", dynamo_utils.counters))
    guard_failures = {
        _normalize_guard_key(key): tuple(str(reason) for reason in reasons)
        for key, reasons in dynamo_utils.guard_failures.items()
    }
    return {
        "compile_metrics_ns": compile_metrics_ns,
        "counter_values": counter_values,
        "guard_failures": guard_failures,
    }


def record_dynamo_state_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> None:
    if not is_rbln_overhead_profiling_enabled() or before is None or after is None:
        return

    before_compile = before.get("compile_metrics_ns", {})
    after_compile = after.get("compile_metrics_ns", {})
    for name, current_ns in after_compile.items():
        delta_ns = current_ns - before_compile.get(name, 0)
        if delta_ns > 0:
            record_phase_duration(f"dynamo.metric.{name}", delta_ns)

    before_counters = before.get("counter_values", {})
    after_counters = after.get("counter_values", {})
    for name, current_value in after_counters.items():
        delta = current_value - before_counters.get(name, 0)
        if delta > 0:
            record_counter(f"dynamo.counter.{name}", delta)

    before_guards = before.get("guard_failures", {})
    after_guards = after.get("guard_failures", {})
    total_new_guard_failures = 0
    for name, current_reasons in after_guards.items():
        prior_reasons = before_guards.get(name, ())
        if len(current_reasons) <= len(prior_reasons):
            continue
        new_reasons = current_reasons[len(prior_reasons) :]
        total_new_guard_failures += len(new_reasons)
        for reason in new_reasons:
            record_guard_failure_reason(reason)
    if total_new_guard_failures > 0:
        record_counter("dynamo.guard_failures", total_new_guard_failures)


def reset_rbln_overhead_profile() -> None:
    global _PROFILE_SUMMARY_EMITTED
    _ensure_process_local_state()
    with _PROFILE_LOCK:
        _PROFILE_STORE.reset()
    _PROFILE_SUMMARY_EMITTED = False


def get_rbln_overhead_profile_snapshot() -> dict[str, Any]:
    _ensure_process_local_state()
    with _PROFILE_LOCK:
        return {
            "pid": _PROFILE_PID,
            "call_totals": {
                key: {"count": stat.count, "total_ns": stat.total_ns, "max_ns": stat.max_ns}
                for key, stat in _PROFILE_STORE.call_totals.items()
            },
            "phase_totals": {
                key: {"count": stat.count, "total_ns": stat.total_ns, "max_ns": stat.max_ns}
                for key, stat in _PROFILE_STORE.phase_totals.items()
            },
            "phase_by_context": {
                key: {"count": stat.count, "total_ns": stat.total_ns, "max_ns": stat.max_ns}
                for key, stat in _PROFILE_STORE.phase_by_context.items()
            },
            "counter_totals": dict(_PROFILE_STORE.counter_totals),
            "counter_by_context": dict(_PROFILE_STORE.counter_by_context),
            "guard_reason_totals": dict(_PROFILE_STORE.guard_reason_totals),
            "guard_reason_by_context": dict(_PROFILE_STORE.guard_reason_by_context),
        }


def format_rbln_overhead_summary(*, top_n: int | None = None) -> str:
    snapshot = get_rbln_overhead_profile_snapshot()
    limit = top_n or _profile_top_n()

    if not snapshot["call_totals"] and not snapshot["phase_totals"] and not snapshot["counter_totals"]:
        return (
            "[TORCH-RBLN][PROFILE] No samples were recorded. "
            f"Enable {_PROFILE_ENV}=ON before importing torch_rbln and execute at least one profiled path."
        )

    lines = [
        "[TORCH-RBLN][PROFILE] Runtime overhead summary",
        f"  pid={snapshot['pid']} env={_PROFILE_ENV}=ON top_n={limit}",
    ]
    lines.extend(_format_context_section("Eager dispatch calls", "eager_op", snapshot, limit))
    lines.extend(_format_context_section("Compiled function calls", "compiled_fn", snapshot, limit))
    lines.extend(_format_phase_totals(snapshot, limit))
    lines.extend(_format_counter_totals(snapshot, limit))
    lines.extend(_format_guard_reasons(snapshot, limit))
    return "\n".join(lines)


def emit_rbln_overhead_summary(*, reset: bool = False) -> str:
    global _PROFILE_SUMMARY_EMITTED
    summary = format_rbln_overhead_summary()
    sys.stderr.write(summary + "\n")
    _PROFILE_SUMMARY_EMITTED = True
    if reset:
        reset_rbln_overhead_profile()
    return summary


def _emit_profile_summary_at_exit() -> None:
    if _PROFILE_SUMMARY_EMITTED or not is_rbln_overhead_profiling_enabled():
        return
    emit_rbln_overhead_summary(reset=False)


def _flatten_mapping(prefix: str, value: Any) -> Iterator[tuple[str, int]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_mapping(child_prefix, nested)
        return

    if isinstance(value, (int, bool)):
        yield prefix, int(value)


def _normalize_guard_key(key: Any) -> str:
    if hasattr(key, "co_filename") and hasattr(key, "co_name") and hasattr(key, "co_firstlineno"):
        filename = os.path.basename(str(key.co_filename))
        return f"{filename}:{key.co_firstlineno}:{key.co_name}"
    return str(key)


def _format_context_section(title: str, category: str, snapshot: dict[str, Any], limit: int) -> list[str]:
    call_totals = snapshot["call_totals"]
    items = [
        (name, stats)
        for (item_category, name), stats in call_totals.items()
        if item_category == category
    ]
    if not items:
        return [f"{title}: none"]

    phase_by_context = snapshot["phase_by_context"]
    counter_by_context = snapshot["counter_by_context"]
    lines = [f"{title}:"]
    for name, stats in sorted(items, key=lambda item: item[1]["total_ns"], reverse=True)[:limit]:
        phase_items = [
            (phase, phase_stats["total_ns"])
            for (ctx_category, ctx_name, phase), phase_stats in phase_by_context.items()
            if ctx_category == category and ctx_name == name
        ]
        phase_items.sort(key=lambda item: item[1], reverse=True)
        counter_items = [
            (counter_name, value)
            for (ctx_category, ctx_name, counter_name), value in counter_by_context.items()
            if ctx_category == category and ctx_name == name and value
        ]
        counter_items.sort(key=lambda item: item[1], reverse=True)

        extras = []
        if phase_items:
            phase_summary = ", ".join(
                f"{phase.split('.')[-1]}={_format_duration(total_ns)}"
                for phase, total_ns in phase_items[:4]
            )
            extras.append(f"phases[{phase_summary}]")
        if counter_items:
            counter_summary = ", ".join(
                f"{counter_name.split('.')[-1]}={value}"
                for counter_name, value in counter_items[:4]
            )
            extras.append(f"counters[{counter_summary}]")

        line = (
            f"  {name}: calls={stats['count']} total={_format_duration(stats['total_ns'])} "
            f"avg={_format_duration(_safe_average(stats['total_ns'], stats['count']))} "
            f"max={_format_duration(stats['max_ns'])}"
        )
        if extras:
            line += " | " + " | ".join(extras)
        lines.append(line)
    return lines


def _format_phase_totals(snapshot: dict[str, Any], limit: int) -> list[str]:
    phase_totals = snapshot["phase_totals"]
    if not phase_totals:
        return ["Phase totals: none"]

    lines = ["Phase totals:"]
    items = sorted(phase_totals.items(), key=lambda item: item[1]["total_ns"], reverse=True)[:limit]
    for name, stats in items:
        lines.append(
            f"  {name}: calls={stats['count']} total={_format_duration(stats['total_ns'])} "
            f"avg={_format_duration(_safe_average(stats['total_ns'], stats['count']))} "
            f"max={_format_duration(stats['max_ns'])}"
        )
    return lines


def _format_counter_totals(snapshot: dict[str, Any], limit: int) -> list[str]:
    counter_totals = snapshot["counter_totals"]
    if not counter_totals:
        return ["Counter totals: none"]

    lines = ["Counter totals:"]
    for name, value in Counter(counter_totals).most_common(limit):
        lines.append(f"  {name}: {value}")
    return lines


def _format_guard_reasons(snapshot: dict[str, Any], limit: int) -> list[str]:
    guard_reason_totals = snapshot["guard_reason_totals"]
    if not guard_reason_totals:
        return ["Guard failure reasons: none"]

    lines = ["Guard failure reasons:"]
    for reason, value in Counter(guard_reason_totals).most_common(limit):
        lines.append(f"  {value}x {reason}")
    return lines


def _format_duration(duration_ns: int) -> str:
    if duration_ns >= 1_000_000_000:
        return f"{duration_ns / 1_000_000_000:.3f}s"
    if duration_ns >= 1_000_000:
        return f"{duration_ns / 1_000_000:.3f}ms"
    if duration_ns >= 1_000:
        return f"{duration_ns / 1_000:.3f}us"
    return f"{duration_ns}ns"


def _safe_average(total_ns: int, count: int) -> int:
    if count <= 0:
        return 0
    return total_ns // count


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_profiler_after_fork)
