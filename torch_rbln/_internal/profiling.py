"""Runtime overhead profiling utilities for torch-rbln.

This module is intentionally lightweight when profiling is disabled and only
activates the higher-overhead dispatch wrapping path when
``TORCH_RBLN_PROFILE`` is enabled before ``torch_rbln`` is imported.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import multiprocessing.util
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


@dataclass
class _ActivePhaseFrame:
    start_ns: int
    child_ns: int = 0


class _ProfileStore:
    def __init__(self) -> None:
        self.call_totals: defaultdict[tuple[str, str], _DurationStat] = defaultdict(_DurationStat)
        self.phase_totals: defaultdict[str, _DurationStat] = defaultdict(_DurationStat)
        self.phase_exclusive_totals: defaultdict[str, _DurationStat] = defaultdict(_DurationStat)
        self.phase_by_context: defaultdict[tuple[str, str, str], _DurationStat] = defaultdict(_DurationStat)
        self.counter_totals: Counter[str] = Counter()
        self.counter_by_context: Counter[tuple[str, str, str]] = Counter()
        self.guard_reason_totals: Counter[str] = Counter()
        self.guard_reason_by_context: Counter[tuple[str, str, str]] = Counter()

    def reset(self) -> None:
        self.call_totals.clear()
        self.phase_totals.clear()
        self.phase_exclusive_totals.clear()
        self.phase_by_context.clear()
        self.counter_totals.clear()
        self.counter_by_context.clear()
        self.guard_reason_totals.clear()
        self.guard_reason_by_context.clear()


_PROFILE_STORE = _ProfileStore()
_PROFILE_LOCK = threading.Lock()
_PROFILE_TLS = threading.local()
_PROFILE_ATEXIT_REGISTERED = False
_PROFILE_MP_FINALIZER_REGISTERED = False
_PROFILE_MP_FINALIZER = None
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


def _ensure_multiprocessing_reporter_registered() -> None:
    global _PROFILE_MP_FINALIZER, _PROFILE_MP_FINALIZER_REGISTERED
    _ensure_process_local_state()
    if _PROFILE_MP_FINALIZER_REGISTERED or not is_rbln_overhead_profiling_enabled():
        return
    _PROFILE_MP_FINALIZER = multiprocessing.util.Finalize(
        None,
        _emit_profile_summary_at_exit,
        exitpriority=1,
    )
    _PROFILE_MP_FINALIZER_REGISTERED = True


def _ensure_process_exit_reporters_registered() -> None:
    _ensure_atexit_reporter_registered()
    _ensure_multiprocessing_reporter_registered()


def _ensure_process_local_state() -> None:
    if os.getpid() != _PROFILE_PID:
        _reset_profiler_after_fork()


def _reset_profiler_after_fork() -> None:
    global _PROFILE_STORE, _PROFILE_LOCK, _PROFILE_TLS
    global _PROFILE_SUMMARY_EMITTED, _PROFILE_PID

    _PROFILE_STORE = _ProfileStore()
    _PROFILE_LOCK = threading.Lock()
    _PROFILE_TLS = threading.local()
    _PROFILE_SUMMARY_EMITTED = False
    _PROFILE_PID = os.getpid()


def _get_context_stack() -> list[_CallContext]:
    stack = getattr(_PROFILE_TLS, "context_stack", None)
    if stack is None:
        stack = []
        _PROFILE_TLS.context_stack = stack
    return stack


def _get_phase_stack() -> list[_ActivePhaseFrame]:
    stack = getattr(_PROFILE_TLS, "phase_stack", None)
    if stack is None:
        stack = []
        _PROFILE_TLS.phase_stack = stack
    return stack


@contextmanager
def profile_call_context(name: str, category: str, *, allow_nested: bool = True) -> Iterator[bool]:
    if not is_rbln_overhead_profiling_enabled():
        yield False
        return

    _ensure_process_exit_reporters_registered()

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

    _ensure_process_exit_reporters_registered()

    start_ns = time.perf_counter_ns()
    phase_stack = _get_phase_stack()
    phase_frame = _ActivePhaseFrame(start_ns=start_ns)
    phase_stack.append(phase_frame)
    try:
        yield
    finally:
        duration_ns = time.perf_counter_ns() - start_ns
        phase_stack.pop()
        exclusive_ns = max(0, duration_ns - phase_frame.child_ns)
        if phase_stack:
            phase_stack[-1].child_ns += duration_ns
        _record_phase_stats(name, duration_ns, exclusive_ns)


def _record_call_total(category: str, name: str, duration_ns: int) -> None:
    _ensure_process_local_state()
    with _PROFILE_LOCK:
        _PROFILE_STORE.call_totals[(category, name)].add(duration_ns)


def record_phase_duration(name: str, duration_ns: int) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_process_exit_reporters_registered()

    _record_phase_stats(name, duration_ns, duration_ns)


def _record_phase_stats(name: str, duration_ns: int, exclusive_ns: int) -> None:
    _ensure_process_local_state()
    contexts = tuple(_get_context_stack())
    with _PROFILE_LOCK:
        _PROFILE_STORE.phase_totals[name].add(duration_ns)
        _PROFILE_STORE.phase_exclusive_totals[name].add(exclusive_ns)
        for context in contexts:
            _PROFILE_STORE.phase_by_context[(context.category, context.name, name)].add(duration_ns)


def get_rbln_overhead_phase_exclusive_snapshot() -> dict[str, dict[str, int]]:
    _ensure_process_local_state()
    with _PROFILE_LOCK:
        return {
            key: {"count": stat.count, "total_ns": stat.total_ns, "max_ns": stat.max_ns}
            for key, stat in _PROFILE_STORE.phase_exclusive_totals.items()
        }


def record_counter(name: str, delta: int = 1) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_process_exit_reporters_registered()

    contexts = tuple(_get_context_stack())
    with _PROFILE_LOCK:
        _PROFILE_STORE.counter_totals[name] += delta
        for context in contexts:
            _PROFILE_STORE.counter_by_context[(context.category, context.name, name)] += delta


def record_guard_failure_reason(reason: str, delta: int = 1) -> None:
    if not is_rbln_overhead_profiling_enabled():
        return

    _ensure_process_exit_reporters_registered()

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
            "phase_exclusive_totals": {
                key: {"count": stat.count, "total_ns": stat.total_ns, "max_ns": stat.max_ns}
                for key, stat in _PROFILE_STORE.phase_exclusive_totals.items()
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


def has_rbln_overhead_profile_samples() -> bool:
    snapshot = get_rbln_overhead_profile_snapshot()
    return any(
        (
            snapshot["call_totals"],
            snapshot["phase_totals"],
            snapshot["counter_totals"],
            snapshot["guard_reason_totals"],
        )
    )


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
    lines.extend(_format_overview(snapshot))
    lines.extend(_format_warmup_adjusted_view(snapshot))
    lines.extend(_format_added_overhead_estimate(snapshot, limit))
    lines.extend(_format_context_section("Eager dispatch calls", "eager_op", snapshot, limit))
    lines.extend(_format_context_section("Compiled function calls", "compiled_fn", snapshot, limit))
    lines.extend(_format_phase_totals(snapshot, limit))
    lines.extend(_format_counter_totals(snapshot, limit))
    lines.extend(_format_guard_reasons(snapshot, limit))
    return "\n".join(lines)


def emit_rbln_overhead_summary(
    *,
    reset: bool = False,
    top_n: int | None = None,
    writer: Callable[[str], Any] | None = None,
) -> str:
    global _PROFILE_SUMMARY_EMITTED
    summary = format_rbln_overhead_summary(top_n=top_n)
    _write_summary(summary, writer=writer)
    _PROFILE_SUMMARY_EMITTED = True
    if reset:
        reset_rbln_overhead_profile()
    return summary


def maybe_emit_rbln_overhead_summary(
    *,
    reset: bool = False,
    top_n: int | None = None,
    writer: Callable[[str], Any] | None = None,
) -> str | None:
    if not is_rbln_overhead_profiling_enabled() or not has_rbln_overhead_profile_samples():
        return None
    return emit_rbln_overhead_summary(reset=reset, top_n=top_n, writer=writer)


def log_rbln_overhead_summary(
    log_fn: Callable[[str], Any],
    *,
    reset: bool = False,
    top_n: int | None = None,
) -> str | None:
    return maybe_emit_rbln_overhead_summary(reset=reset, top_n=top_n, writer=lambda summary: _log_summary(summary, log_fn))


def _emit_profile_summary_at_exit() -> None:
    if _PROFILE_SUMMARY_EMITTED or not is_rbln_overhead_profiling_enabled():
        return
    maybe_emit_rbln_overhead_summary(reset=False)


def _write_summary(summary: str, *, writer: Callable[[str], Any] | None = None) -> None:
    if writer is None:
        sys.stderr.write(summary + "\n")
        sys.stderr.flush()
        return
    writer(summary)


def _log_summary(summary: str, log_fn: Callable[[str], Any]) -> None:
    for line in summary.splitlines():
        log_fn(line)


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


def _format_overview(snapshot: dict[str, Any]) -> list[str]:
    eager_ops = sum(1 for category, _ in snapshot["call_totals"] if category == "eager_op")
    compiled_fns = sum(1 for category, _ in snapshot["call_totals"] if category == "compiled_fn")
    return [
        "Overview:",
        "  "
        f"eager_ops={eager_ops} "
        f"compiled_fns={compiled_fns} "
        f"phases={len(snapshot['phase_totals'])} "
        f"counters={len(snapshot['counter_totals'])} "
        f"guard_reasons={len(snapshot['guard_reason_totals'])}",
    ]


def _summarize_added_overhead_buckets(snapshot: dict[str, Any]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    phase_exclusive_totals = snapshot.get("phase_exclusive_totals", {})
    known_buckets: Counter[str] = Counter()
    excluded_buckets: Counter[str] = Counter()
    unclassified_buckets: Counter[str] = Counter()

    for phase_name, stats in phase_exclusive_totals.items():
        phase_total = stats["total_ns"]
        bucket_name, bucket_group = _classify_phase_for_overhead_estimate(phase_name)
        if bucket_group == "known":
            known_buckets[bucket_name] += phase_total
        elif bucket_group == "excluded":
            excluded_buckets[bucket_name] += phase_total
        else:
            unclassified_buckets[bucket_name] += phase_total

    return known_buckets, excluded_buckets, unclassified_buckets


def _format_warmup_adjusted_view(snapshot: dict[str, Any]) -> list[str]:
    known_buckets, _, _ = _summarize_added_overhead_buckets(snapshot)
    if not known_buckets:
        return ["Warm-up adjusted view: none"]

    warmup_total = known_buckets.get("compile_stack", 0)
    steady_state_total = sum(total_ns for bucket_name, total_ns in known_buckets.items() if bucket_name != "compile_stack")
    known_total = warmup_total + steady_state_total
    compile_events = _estimate_compile_event_count(snapshot)
    profiled_calls = _estimate_profiled_call_count(snapshot)

    lines = ["Warm-up adjusted view:"]
    lines.append(
        "  "
        f"compile_events~={compile_events} "
        f"profiled_calls={profiled_calls} "
        f"guard_failures={snapshot['counter_totals'].get('dynamo.guard_failures', 0)}"
    )
    lines.extend(
        _format_table_header(
            ("Scope", 24, "<"),
            ("Share", 7, ">"),
            ("Total", 12, ">"),
            ("Normalized", 18, ">"),
            ("Interpretation", 0, "<"),
        )
    )

    if warmup_total > 0:
        lines.append(
            _format_table_row(
                ("one_time_warmup", 24, "<"),
                (_format_percentage(warmup_total, known_total), 7, ">"),
                (_format_duration(warmup_total), 12, ">"),
                (_format_normalized_duration(warmup_total, compile_events, "compile"), 18, ">"),
                ("mostly amortized after cache reuse; may return on recompiles", 0, "<"),
            )
        )

    if steady_state_total > 0:
        lines.append(
            _format_table_row(
                ("recurring_steady_state", 24, "<"),
                (_format_percentage(steady_state_total, known_total), 7, ">"),
                (_format_duration(steady_state_total), 12, ">"),
                (_format_normalized_duration(steady_state_total, profiled_calls, "call"), 18, ">"),
                ("persists after warm-up and represents ongoing host-side overhead", 0, "<"),
            )
        )

    lines.append(
        _format_table_row(
            ("known_overhead_total", 24, "<"),
            ("100.0%", 7, ">"),
            (_format_duration(known_total), 12, ">"),
            (_format_normalized_duration(known_total, profiled_calls, "call"), 18, ">"),
            ("raw total of known extra overhead in this run", 0, "<"),
        )
    )
    return lines


def _format_added_overhead_estimate(snapshot: dict[str, Any], limit: int) -> list[str]:
    phase_exclusive_totals = snapshot.get("phase_exclusive_totals", {})
    if not phase_exclusive_totals:
        return ["Added overhead estimate: none"]

    known_buckets, excluded_buckets, unclassified_buckets = _summarize_added_overhead_buckets(snapshot)

    lines = ["Added overhead estimate (exclusive, host-side):"]

    if known_buckets:
        known_total = sum(known_buckets.values())
        known_items = known_buckets.most_common()
        name_width = max(len("Bucket"), max(len(name) for name, _ in known_items + [("known_overhead_total", 0)]))
        name_width = min(name_width, 24)
        lines.extend(
            _format_table_header(
                ("Bucket", name_width, "<"),
                ("Share", 7, ">"),
                ("Total", 12, ">"),
                ("What it means", 0, "<"),
            )
        )
        for name, total_ns in known_items:
            lines.append(
                _format_table_row(
                    (name, name_width, "<"),
                    (_format_percentage(total_ns, known_total), 7, ">"),
                    (_format_duration(total_ns), 12, ">"),
                    (_overhead_bucket_description(name), 0, "<"),
                )
            )
        lines.append(
            _format_table_row(
                ("known_overhead_total", name_width, "<"),
                ("100.0%", 7, ">"),
                (_format_duration(known_total), 12, ">"),
                ("sum of known extra host-side overhead only", 0, "<"),
            )
        )
    else:
        lines.append("  known_overhead_total: none")

    if excluded_buckets:
        lines.append("Excluded mixed time (not added to total):")
        lines.extend(
            _format_table_header(
                ("Bucket", 24, "<"),
                ("Total", 12, ">"),
                ("Why excluded", 0, "<"),
            )
        )
        for name, total_ns in excluded_buckets.most_common(limit):
            lines.append(
                _format_table_row(
                    (name, 24, "<"),
                    (_format_duration(total_ns), 12, ">"),
                    (_excluded_bucket_description(name), 0, "<"),
                )
            )

    if unclassified_buckets:
        lines.append("Unclassified profiled time:")
        lines.extend(
            _format_table_header(
                ("Bucket", 24, "<"),
                ("Total", 12, ">"),
            )
        )
        for name, total_ns in unclassified_buckets.most_common(limit):
            lines.append(_format_table_row((name, 24, "<"), (_format_duration(total_ns), 12, ">")))

    return lines


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
    ranked_items = sorted(items, key=lambda item: item[1]["total_ns"], reverse=True)[:limit]
    total_ns = sum(stats["total_ns"] for _, stats in ranked_items)
    name_width = max(len("Name"), max(len(name) for name, _ in ranked_items))
    name_width = min(name_width, 36)

    lines = [f"{title}:"]
    lines.extend(
        _format_table_header(
            ("Name", name_width, "<"),
            ("Calls", 7, ">"),
            ("Share", 7, ">"),
            ("Total", 12, ">"),
            ("Avg", 12, ">"),
            ("Max", 12, ">"),
        )
    )

    for name, stats in ranked_items:
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

        lines.append(
            _format_table_row(
                (name, name_width, "<"),
                (str(stats["count"]), 7, ">"),
                (_format_percentage(stats["total_ns"], total_ns), 7, ">"),
                (_format_duration(stats["total_ns"]), 12, ">"),
                (_format_duration(_safe_average(stats["total_ns"], stats["count"])), 12, ">"),
                (_format_duration(stats["max_ns"]), 12, ">"),
            )
        )
        if phase_items:
            lines.extend(
                _format_wrapped_items(
                    "phases",
                    [
                        f"{_short_nested_name(phase)}={_format_duration(duration_ns)} "
                        f"({_format_percentage(duration_ns, stats['total_ns'])})"
                        for phase, duration_ns in phase_items[:4]
                    ],
                )
            )
        if counter_items:
            lines.extend(
                _format_wrapped_items(
                    "counters",
                    [f"{_short_nested_name(counter_name)}={value}" for counter_name, value in counter_items[:4]],
                )
            )
    return lines


def _format_phase_totals(snapshot: dict[str, Any], limit: int) -> list[str]:
    phase_totals = snapshot["phase_totals"]
    if not phase_totals:
        return ["Phase totals: none"]

    items = sorted(phase_totals.items(), key=lambda item: item[1]["total_ns"], reverse=True)[:limit]
    total_ns = sum(stats["total_ns"] for _, stats in items)
    name_width = max(len("Phase"), max(len(name) for name, _ in items))
    name_width = min(name_width, 44)

    lines = ["Phase totals:"]
    lines.extend(
        _format_table_header(
            ("Phase", name_width, "<"),
            ("Calls", 7, ">"),
            ("Share", 7, ">"),
            ("Total", 12, ">"),
            ("Avg", 12, ">"),
            ("Max", 12, ">"),
        )
    )
    for name, stats in items:
        lines.append(
            _format_table_row(
                (name, name_width, "<"),
                (str(stats["count"]), 7, ">"),
                (_format_percentage(stats["total_ns"], total_ns), 7, ">"),
                (_format_duration(stats["total_ns"]), 12, ">"),
                (_format_duration(_safe_average(stats["total_ns"], stats["count"])), 12, ">"),
                (_format_duration(stats["max_ns"]), 12, ">"),
            )
        )
    return lines


def _format_counter_totals(snapshot: dict[str, Any], limit: int) -> list[str]:
    counter_totals = snapshot["counter_totals"]
    if not counter_totals:
        return ["Counter totals: none"]

    items = Counter(counter_totals).most_common(limit)
    name_width = max(len("Counter"), max(len(name) for name, _ in items))
    name_width = min(name_width, 52)

    lines = ["Counter totals:"]
    lines.extend(
        _format_table_header(
            ("Counter", name_width, "<"),
            ("Value", 12, ">"),
        )
    )
    for name, value in items:
        lines.append(_format_table_row((name, name_width, "<"), (str(value), 12, ">")))
    return lines


def _format_guard_reasons(snapshot: dict[str, Any], limit: int) -> list[str]:
    guard_reason_totals = snapshot["guard_reason_totals"]
    if not guard_reason_totals:
        return ["Guard failure reasons: none"]

    items = Counter(guard_reason_totals).most_common(limit)
    lines = ["Guard failure reasons:"]
    lines.extend(
        _format_table_header(
            ("Count", 7, ">"),
            ("Reason", 0, "<"),
        )
    )
    for reason, value in items:
        lines.append(_format_table_row((f"{value}x", 7, ">"), (reason, 0, "<")))
    return lines


def _format_table_header(*columns: tuple[str, int, str]) -> list[str]:
    return [
        _format_table_row(*columns),
        _format_table_rule(columns),
    ]


def _format_table_rule(columns: tuple[tuple[str, int, str], ...]) -> str:
    pieces = []
    for _, width, _ in columns:
        rule_width = width if width > 0 else 12
        pieces.append("-" * rule_width)
    return "  " + " ".join(pieces)


def _format_table_row(*columns: tuple[str, int, str]) -> str:
    pieces = []
    for value, width, align in columns:
        text = str(value)
        if width > 0 and len(text) > width:
            text = text[: width - 1] + "…"
        if width <= 0:
            pieces.append(text)
        elif align == "<":
            pieces.append(f"{text:<{width}}")
        else:
            pieces.append(f"{text:>{width}}")
    return "  " + " ".join(pieces)


def _format_wrapped_items(label: str, items: list[str], *, width: int = 108) -> list[str]:
    if not items:
        return []

    prefix = f"    {label:<8} "
    continuation_prefix = " " * len(prefix)
    lines: list[str] = []
    current_line = prefix
    for item in items:
        token = item if current_line == prefix else f" | {item}"
        if current_line != prefix and len(current_line) + len(token) > width:
            lines.append(current_line)
            current_line = continuation_prefix + item
            continue
        current_line += token
    lines.append(current_line)
    return lines


def _format_percentage(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.0%"
    return f"{(part / whole) * 100:.1f}%"


def _format_normalized_duration(total_ns: int, count: int, unit: str) -> str:
    if count <= 0:
        return "n/a"
    return f"{_format_duration(total_ns // count)}/{unit}"


def _short_nested_name(name: str) -> str:
    if "." not in name:
        return name
    return name.split(".")[-1]


def _estimate_compile_event_count(snapshot: dict[str, Any]) -> int:
    counter_totals = snapshot.get("counter_totals", {})
    compile_events = max(
        int(counter_totals.get("dynamo.counter.stats.unique_graphs", 0)),
        int(counter_totals.get("compile_cache.miss", 0)),
        int(counter_totals.get("torch_compile.rbln_backend_calls", 0)),
    )
    if compile_events <= 0 and "compile_stack" in _summarize_added_overhead_buckets(snapshot)[0]:
        return 1
    return compile_events


def _estimate_profiled_call_count(snapshot: dict[str, Any]) -> int:
    return sum(stats["count"] for stats in snapshot.get("call_totals", {}).values())


_KNOWN_EAGER_OVERHEAD_PHASES = frozenset(
    {
        "ops.finalize_output_tensor",
        "ops.extract_device_id_from_inputs",
        "ops.broadcast_args_general",
        "ops.prepare_args_for_contiguous",
        "ops.is_cpu_fallback_cases",
        "ops.can_use_out_tensor_directly",
        "ops.cpu_fallback_path",
    }
)

_KNOWN_COMPILE_OVERHEAD_PHASES = frozenset(
    {
        "torch_compile.api",
        "compile_cache.total",
        "compile_cache.lock_wait",
        "compile_cache.torch_compile_api",
        "compile_wrapper.auto_determine_tp_if_needed",
        "compile_wrapper.auto_tp_resolution",
        "compile_wrapper.recompile_with_tp_size",
        "compile_wrapper.handle_tp_failover",
    }
)

_KNOWN_FALLBACK_GLUE_PHASES = frozenset(
    {
        "compile_wrapper.attempt_cpu_fallback",
        "compile_wrapper.cpu_fallback",
    }
)


def _classify_phase_for_overhead_estimate(phase_name: str) -> tuple[str, str]:
    if phase_name == "compile_wrapper.compiled_call":
        return ("compiled_execution_wall", "excluded")
    if phase_name == "ops.cpu_fallback.exec_cpu_op":
        return ("cpu_fallback_compute", "excluded")
    if phase_name == "compile_wrapper.cpu_fallback.exec_original":
        return ("compile_fallback_compute", "excluded")

    if phase_name.startswith("dynamo.metric."):
        return ("compile_stack", "known")
    if phase_name in _KNOWN_COMPILE_OVERHEAD_PHASES:
        return ("compile_stack", "known")
    if phase_name in _KNOWN_EAGER_OVERHEAD_PHASES:
        return ("eager_dispatch_glue", "known")
    if phase_name in _KNOWN_FALLBACK_GLUE_PHASES:
        return ("fallback_glue", "known")
    if phase_name.startswith("ops.cpu_fallback."):
        return ("fallback_glue", "known")
    if phase_name.startswith("compile_wrapper.cpu_fallback."):
        return ("fallback_glue", "known")

    return (phase_name, "unclassified")


def _overhead_bucket_description(bucket_name: str) -> str:
    if bucket_name == "compile_stack":
        return "torch.compile entry, cache, Dynamo compile, guards, and failover control"
    if bucket_name == "eager_dispatch_glue":
        return "RBLN eager dispatch preprocessing and output handling"
    if bucket_name == "fallback_glue":
        return "fallback-specific glue such as to_cpu/to_device and device resolution"
    return "classified extra overhead"


def _excluded_bucket_description(bucket_name: str) -> str:
    if bucket_name == "compiled_execution_wall":
        return "contains actual compiled execution wall time, not just extra overhead"
    if bucket_name == "cpu_fallback_compute":
        return "actual CPU operator compute during eager fallback"
    if bucket_name == "compile_fallback_compute":
        return "actual original function compute during compile fallback"
    return "excluded from known overhead total"


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
