import os
import unittest
from unittest.mock import patch

os.environ.setdefault("TORCH_RBLN_DIAGNOSE", "1")

import torch_rbln._internal.profiling as profiling
from torch_rbln._internal.profiling import (
    _reset_profiler_after_fork,
    format_rbln_overhead_summary,
    get_rbln_overhead_phase_exclusive_snapshot,
    get_rbln_overhead_profile_snapshot,
    has_rbln_overhead_profile_samples,
    log_rbln_overhead_summary,
    maybe_emit_rbln_overhead_summary,
    profile_call_context,
    record_counter,
    record_dynamo_state_delta,
    record_phase_duration,
    reset_rbln_overhead_profile,
    wrap_registered_dispatch_functions,
)


class TestRblnProfiling(unittest.TestCase):
    def tearDown(self):
        reset_rbln_overhead_profile()
        super().tearDown()

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_wrap_registered_dispatch_functions_profiles_top_level_only(self):
        namespace = {}

        def add_rbln(value):
            record_phase_duration("ops.inner_phase", 5)
            return value + 1

        def add_out_rbln(value):
            return namespace["add_rbln"](value)

        add_rbln.__module__ = __name__
        add_out_rbln.__module__ = __name__
        namespace["add_rbln"] = add_rbln
        namespace["add_out_rbln"] = add_out_rbln

        wrap_registered_dispatch_functions(namespace, module_name=__name__)

        with patch("torch_rbln._internal.profiling.time.perf_counter_ns", side_effect=[100, 160]):
            self.assertEqual(namespace["add_out_rbln"](1), 2)

        snapshot = get_rbln_overhead_profile_snapshot()
        self.assertEqual(snapshot["call_totals"][("eager_op", "add_out_rbln")]["count"], 1)
        self.assertNotIn(("eager_op", "add_rbln"), snapshot["call_totals"])
        self.assertEqual(
            snapshot["phase_by_context"][("eager_op", "add_out_rbln", "ops.inner_phase")]["total_ns"],
            5,
        )

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_record_dynamo_state_delta_tracks_compile_metrics_and_guards(self):
        before = {
            "compile_metrics_ns": {"backend_compile": 100, "entire_frame_compile": 150},
            "counter_values": {"stats.unique_graphs": 1},
            "guard_failures": {"foo.py:1:forward": ("tensor shape mismatch",)},
        }
        after = {
            "compile_metrics_ns": {"backend_compile": 300, "entire_frame_compile": 450},
            "counter_values": {"stats.unique_graphs": 3, "stats.calls_captured": 4},
            "guard_failures": {
                "foo.py:1:forward": ("tensor shape mismatch", "dtype mismatch"),
                "bar.py:9:forward": ("requires_grad mismatch",),
            },
        }

        with profile_call_context("compiled_add", "compiled_fn", allow_nested=True):
            record_dynamo_state_delta(before, after)

        snapshot = get_rbln_overhead_profile_snapshot()
        self.assertEqual(snapshot["phase_totals"]["dynamo.metric.backend_compile"]["total_ns"], 200)
        self.assertEqual(snapshot["phase_totals"]["dynamo.metric.entire_frame_compile"]["total_ns"], 300)
        self.assertEqual(snapshot["counter_totals"]["dynamo.counter.stats.unique_graphs"], 2)
        self.assertEqual(snapshot["counter_totals"]["dynamo.counter.stats.calls_captured"], 4)
        self.assertEqual(snapshot["counter_totals"]["dynamo.guard_failures"], 2)
        self.assertEqual(snapshot["guard_reason_totals"]["dtype mismatch"], 1)
        self.assertEqual(snapshot["guard_reason_totals"]["requires_grad mismatch"], 1)
        self.assertEqual(
            snapshot["phase_by_context"][("compiled_fn", "compiled_add", "dynamo.metric.backend_compile")]["total_ns"],
            200,
        )

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_summary_contains_calls_phases_and_counters(self):
        with profile_call_context("add_rbln", "eager_op", allow_nested=False):
            record_phase_duration("compile_cache.total", 10_000)
            record_counter("compile_cache.hit", 3)

        summary = format_rbln_overhead_summary(top_n=5)

        self.assertIn("Runtime overhead summary", summary)
        self.assertIn("Overview:", summary)
        self.assertIn("Eager dispatch calls", summary)
        self.assertIn("add_rbln", summary)
        self.assertIn("Phase totals", summary)
        self.assertIn("Counter totals", summary)
        self.assertIn("compile_cache.hit", summary)
        self.assertIn("Warm-up adjusted view", summary)
        self.assertIn("phases", summary)
        self.assertIn("counters", summary)
        self.assertIn("Share", summary)
        self.assertIn("Added overhead estimate", summary)
        self.assertIn("known_overhead_total", summary)
        self.assertIn("pid=", summary)

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_overhead_estimate_separates_known_and_excluded_time(self):
        with profile_call_context("compiled_add", "compiled_fn", allow_nested=True):
            record_phase_duration("torch_compile.api", 1_000_000)
            record_phase_duration("dynamo.metric.build_guards", 2_000_000)
        with profile_call_context("add_rbln", "eager_op", allow_nested=False):
            record_phase_duration("ops.finalize_output_tensor", 500_000)
        record_phase_duration("ops.cpu_fallback.exec_cpu_op", 9_000_000)

        summary = format_rbln_overhead_summary(top_n=5)
        exclusive_snapshot = get_rbln_overhead_phase_exclusive_snapshot()

        self.assertEqual(exclusive_snapshot["torch_compile.api"]["total_ns"], 1_000_000)
        self.assertEqual(exclusive_snapshot["ops.cpu_fallback.exec_cpu_op"]["total_ns"], 9_000_000)
        self.assertIn("compile_stack", summary)
        self.assertIn("eager_dispatch_glue", summary)
        self.assertIn("cpu_fallback_compute", summary)
        self.assertIn("one_time_warmup", summary)
        self.assertIn("recurring_steady_state", summary)
        self.assertIn("/compile", summary)
        self.assertIn("/call", summary)
        self.assertIn("3.500ms", summary)
        self.assertIn("9.000ms", summary)

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_log_summary_uses_callback_and_can_reset(self):
        with profile_call_context("add_rbln", "eager_op", allow_nested=False):
            record_phase_duration("compile_cache.total", 10_000)

        messages = []
        summary = log_rbln_overhead_summary(messages.append, reset=True, top_n=3)

        self.assertIsNotNone(summary)
        self.assertEqual(messages, summary.splitlines())
        self.assertIn("add_rbln", summary)
        self.assertFalse(has_rbln_overhead_profile_samples())

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_maybe_emit_summary_is_quiet_when_no_samples_exist(self):
        self.assertFalse(has_rbln_overhead_profile_samples())
        self.assertIsNone(maybe_emit_rbln_overhead_summary())

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    @patch("torch_rbln._internal.profiling.multiprocessing.util.Finalize")
    @patch("torch_rbln._internal.profiling.atexit.register")
    def test_profile_event_registers_process_exit_reporters(self, atexit_register, finalize):
        profiling._PROFILE_ATEXIT_REGISTERED = False
        profiling._PROFILE_MP_FINALIZER_REGISTERED = False
        profiling._PROFILE_MP_FINALIZER = None

        record_counter("compile_cache.hit", 1)

        atexit_register.assert_called_once()
        finalize.assert_called_once()

    @patch.dict("os.environ", {"TORCH_RBLN_PROFILE": "ON"}, clear=False)
    def test_reset_profiler_after_fork_clears_parent_state(self):
        with profile_call_context("parent_add", "eager_op", allow_nested=False):
            record_phase_duration("compile_cache.total", 10)
            record_counter("compile_cache.hit", 1)

        _reset_profiler_after_fork()

        snapshot = get_rbln_overhead_profile_snapshot()
        self.assertEqual(snapshot["call_totals"], {})
        self.assertEqual(snapshot["phase_totals"], {})
        self.assertEqual(snapshot["counter_totals"], {})


if __name__ == "__main__":
    unittest.main()
