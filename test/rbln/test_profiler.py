# Owner(s): ["module: PrivateUse1"]

"""User-level tests for ``torch.rbln.explain()`` — the hidden-overhead profiler.

A normal PyTorch op can silently round-trip the host or fall back to CPU for
reasons the user never asked for. These tests drive real ops on the ``rbln``
device and assert the profiler surfaces those hidden events, and ONLY those.

Mapping to the profiler signal taxonomy discussed in design (A-E):

  * copy host-bounce  -> ``hidden_host_bounce``   (torch-side) [core]
  * A recompile       -> ``dispatch.recompile_miss`` (torch-side)
  * B cpu_fallback    -> ``dispatch.cpu_fallback``    (torch-side)
  * C device idle / D command-stream / leaf-byte host-traffic -> deliberately
                         EXCLUDED. Timelines and utilization are torch.profiler's
                         job (a row per NPU with ``RBLN_PROFILER=1``), out of this
                         profiler's hidden-overhead scope.
  * E memory gauge    -> ``device_memory`` (caching allocator). NOT a hidden-overhead
                         signal; kept because every device tensor allocation goes
                         through the caching allocator, so it is complete.
  * runtime copy time -> ``runtime`` (per-primitive time + calls inside the runtime's
                         copy calls). Context, not a verdict signal.

These tests therefore assert the profiler's *honesty and scope* — that it
carries the hidden-overhead signals plus the memory gauge, and EXCLUDES the
out-of-scope dispatch/utilization counters.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  -- registers the ``rbln`` device + ``torch.rbln``


DEV = "rbln"


def _runtime_calls(dump):
    """Calls per runtime copy primitive in an explain() dump (primitives with no call omitted)."""
    return {name: v["calls"] for name, v in dump["runtime"]["by_primitive"].items()}


@pytest.mark.test_set_ci
class TestProfilerCopyBounce(TestCase):
    """The core signal: a hidden host round-trip behind a plain ``copy_``."""

    def test_contiguous_copy_stays_on_device(self):
        # A contiguous same-shape/dtype copy_ is one device->device copy: no host bounce,
        # and the runtime copy counters show a v2v and nothing read back to the host.
        x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
        y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
        with torch.rbln.explain() as p:
            y.copy_(x)
        d = p.dump()
        self.assertEqual(d["hidden_host_bounce"]["total_count"], 0)
        calls = _runtime_calls(d)
        self.assertGreaterEqual(calls.get("v2v", 0), 1)
        self.assertEqual(calls.get("v2h", 0) + calls.get("v2h_multi", 0), 0)

    def test_d2d_int_cast_bounces_with_bytes(self):
        # A device->device cast (int64->int32) is not a byte copy, so neither the direct
        # v2v nor the strided v2v engine takes it: it round-trips the host -> explain shows
        # copy_d2d_host_bounce with bytes, so the region is not clean.
        s = torch.tensor([3, 4, 5, 6], dtype=torch.int64).to(DEV)
        with torch.rbln.explain() as p:
            _ = s.to(torch.int32)
        hb = p.dump()["hidden_host_bounce"]
        self.assertGreaterEqual(hb["by_site"]["copy_d2d_host_bounce"]["count"], 1)
        self.assertGreater(hb["total_bytes"], 0)  # real bytes were moved through host
        self.assertFalse(p.verdict()["clean"])  # a host bounce fired -> not clean


@pytest.mark.test_set_ci
class TestProfilerDispatchSignals(TestCase):
    """A (recompile) and B (cpu_fallback) read from the existing dispatch counters."""

    def test_B_cpu_fallback_counted(self):
        with torch.rbln.explain() as p:
            a = torch.ones(32, 32, device=DEV, dtype=torch.int32)  # non-fp16 -> CPU fallback
            _ = a + a
        disp = p.dump()["dispatch"]
        self.assertGreaterEqual(disp["cpu_fallback"], 1)
        # COST is surfaced too (wall ns spent in cpu_fallback), so a report can tell
        # many-cheap fallbacks from few-expensive ones. >= 0 (0 only on a _C predating it).
        self.assertIn("cpu_fallback_ns", disp)
        self.assertGreaterEqual(disp["cpu_fallback_ns"], 0)
        self.assertFalse(p.verdict()["clean"])  # ran on CPU -> not clean

    def test_A_recompile_counted(self):
        # First use of an unusual shape forces a compile; the profiler must count
        # it as a recompile/miss (the actionable "you are recompiling" signal).
        # NOTE: whether a *repeat* is served from warm cache is a property of the
        # warm cache, not of the profiler, so it is deliberately not asserted here.
        with torch.rbln.explain() as p:
            x = torch.randn(29, 31, device=DEV, dtype=torch.float16)
            _ = x + x
        self.assertGreaterEqual(p.dump()["dispatch"]["recompile_miss"], 1)
        self.assertFalse(p.verdict()["clean"])  # a compile happened -> not clean


@pytest.mark.test_set_ci
class TestProfilerFallbackRegimes(TestCase):
    """explain must render each dispatch regime distinctly. A device run neither falls
    back nor moves data to the host. A CPU fallback copies its inputs to the host and
    its results back: that traffic is the fallback's own cost, shown as runtime copy
    calls next to the cpu_fallback row, not as a separate host bounce. These lock that
    distinction so a regression can't silently turn one regime into another.
    (host-bounce WITHOUT fallback is covered by TestProfilerCopyBounce.)"""

    @pytest.mark.usefixtures("enable_deploy_mode")
    def test_device_run_no_fallback_no_transfer(self):
        # Contiguous fp16 elementwise runs on-device: no CPU fallback, no host bounce,
        # nothing copied to or from the host. Deploy mode, because outside it the dispatch
        # shim reads every device input to the host to scan it for NaN/Inf. (A one-time
        # recompile may still occur; not asserted here.)
        x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
        for _ in range(3):
            _ = x * 2  # warm
        with torch.rbln.explain() as p:
            _ = x * 2
        d = p.dump()
        self.assertEqual(d["dispatch"]["cpu_fallback"], 0)
        self.assertEqual(d["hidden_host_bounce"]["total_count"], 0)
        calls = _runtime_calls(d)
        for prim in ("v2h", "h2v", "v2h_multi", "h2v_multi"):
            self.assertEqual(calls.get(prim, 0), 0, prim)

    def test_fallback_moves_inputs_and_results(self):
        # int ops fall back EVERY op; each one reads its input to the host and writes its
        # result back. explain surfaces the cpu_fallback count + its wall-time, the
        # traffic as runtime copy calls, and no host bounce.
        a = torch.arange(256, dtype=torch.int32).to(DEV)

        def chain():
            r = a
            for _ in range(8):
                r = r - 1
            return r

        chain()  # warm
        with torch.rbln.explain() as p:
            chain()
        d = p.dump()
        self.assertGreaterEqual(d["dispatch"]["cpu_fallback"], 8)
        self.assertIn("cpu_fallback_ns", d["dispatch"])  # COST is surfaced
        self.assertEqual(d["hidden_host_bounce"]["total_count"], 0)
        calls = _runtime_calls(d)
        self.assertGreaterEqual(calls.get("v2h", 0) + calls.get("v2h_multi", 0), 8)
        self.assertGreaterEqual(calls.get("h2v", 0) + calls.get("h2v_multi", 0), 8)

    @pytest.mark.usefixtures("enable_deploy_mode")
    def test_a_narrow_result_runs_on_device_through_the_host(self):
        # The compiler lays a [2, 2] mm result out padded, not as torch holds it, so the op
        # runs on the device and its result is decoded on the host: no CPU fallback, one
        # op_arg_through_host per call, and the value torch computes.
        a = torch.randn(2, 1024, dtype=torch.float16)
        x = a.to(DEV)
        _ = x @ x.t()  # warm
        with torch.rbln.explain() as p:
            out = x @ x.t()
        d = p.dump()
        self.assertEqual(d["dispatch"]["cpu_fallback"], 0)
        self.assertGreaterEqual(d["hidden_host_bounce"]["by_site"]["op_arg_through_host"]["count"], 1)
        self.assertFalse(p.verdict()["clean"])
        torch.testing.assert_close(out.cpu().float(), a.float() @ a.float().t(), rtol=2e-2, atol=1.0)


@pytest.mark.test_set_ci
class TestProfilerTruthfulnessAndScope(TestCase):
    """The verdict must never claim more truth than it has (C/E), and must not
    carry signals the user cannot act on (D)."""

    def test_in_scope_sections_only(self):
        # The in-scope context (device-memory gauge, runtime copy time) is present; the
        # out-of-scope dispatch/utilization counters (leaf-byte traffic, command streams,
        # device idle) must NOT be surfaced.
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        d = p.dump()
        self.assertIn("device_memory", d)  # E (resource gauge, kept)
        self.assertIn("runtime", d)
        # out of hidden-overhead scope — must not be surfaced:
        self.assertNotIn("runtime_host_traffic", d)  # leaf bytes
        self.assertNotIn("command_streams", d)  # D
        self.assertNotIn("device_idle", d)  # C

    def test_E_device_memory_gauge(self):
        # Pure allocation (no compute) — the gauge is the caching allocator's reserved
        # high-water mark, so it reads a sane non-zero peak >= current, and it agrees
        # with torch.rbln.memory_reserved() on a single device.
        with torch.rbln.explain() as p:
            t = torch.empty(2048, 2048, device=DEV, dtype=torch.float16)
        m = p.dump()["device_memory"]
        self.assertGreaterEqual(m["peak_bytes"], m["current_bytes"])
        self.assertGreaterEqual(m["current_bytes"], t.numel() * t.element_size())
        if m["devices"] == 1:
            self.assertEqual(m["current_bytes"], torch.rbln.memory_reserved(DEV))

    def test_D_command_stream_is_not_a_verdict_signal(self):
        # Command-stream count / structural padding are intentionally excluded
        # from the verdict (measurable but not user-actionable).
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        keys = set(p.verdict().keys())
        for forbidden in ("command_stream", "cs_count", "padding", "fragmentation"):
            self.assertNotIn(forbidden, keys)

    def test_empty_region_is_clean(self):
        # No hidden event fired -> clean. ``clean`` is a FACT ("did anything fire"),
        # not a RED/AMBER/GREEN severity grade (the tool observes, it does not grade).
        with torch.rbln.explain() as p:
            pass
        self.assertTrue(p.verdict()["clean"])
        self.assertEqual(p.verdict()["hidden_host_bounces"], 0)

    def test_regions_are_independent_deltas(self):
        with torch.rbln.explain() as p1:
            s = torch.tensor([3, 4, 5, 6], dtype=torch.int64).to(DEV)
            _ = s.to(torch.int32)  # int d2d cast -> host bounce (fp16 v2v can't serve int)
        self.assertGreaterEqual(p1.dump()["hidden_host_bounce"]["total_count"], 1)
        # a fresh region must not inherit the previous region's incidents.
        with torch.rbln.explain() as p2:
            x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
            y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
            y.copy_(x)
        self.assertEqual(p2.dump()["hidden_host_bounce"]["total_count"], 0)


@pytest.mark.test_set_ci
class TestProfilerApi(TestCase):
    def test_report_is_str_and_dump_shape(self):
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        self.assertIsInstance(p.report(), str)
        d = p.dump()
        self.assertIn("hidden_host_bounce", d)
        self.assertIn("dispatch", d)
        self.assertIn("host_threads", d)
        self.assertIn("notes", d)

    def test_verdict_is_factual_not_graded(self):
        # The tool OBSERVES; it does not grade. verdict() carries a factual ``clean``
        # flag (+ ``reasons``) and NO RED/AMBER/GREEN ``status``. report() shows a
        # factual [clean]/[overhead] marker, never a severity colour.
        with torch.rbln.explain() as p:
            a = torch.ones(16, 16, device=DEV, dtype=torch.int32)  # int32 -> cpu_fallback
            _ = a + a
        v = p.verdict()
        self.assertIn("clean", v)
        self.assertNotIn("status", v)  # no RED/AMBER/GREEN grade
        self.assertIsInstance(v["clean"], bool)
        self.assertFalse(v["clean"])  # a fallback fired
        rep = p.report()
        self.assertIn("[overhead:", rep)  # marker carries a factual signal count, e.g. "[overhead: 1 signal]"
        for banned in ("[ OK ]", "[WARN]", "[BAD ]", "RBLN EXPLAIN - RED", "GREEN", "AMBER"):
            self.assertNotIn(banned, rep)

    def test_explain_steady_isolates_cold_compile(self):
        # explain_steady profiles the WARM (steady-state) call. A stable-shape op
        # compiles once (cold) then hits the warm cache, so steady-state recompile
        # must be <= the cold sample's. This is the one-time-vs-every-step split.
        a = torch.randn(48, 48, device=DEV, dtype=torch.float16)
        cold, warm = torch.rbln.explain_steady(lambda: a + a, warmup=3, return_cold=True)
        self.assertLessEqual(warm.dump()["dispatch"]["recompile_miss"], cold.dump()["dispatch"]["recompile_miss"])
        self.assertIsInstance(warm.verdict()["clean"], bool)

    def test_A_where_traceback_is_opt_in(self):
        # (A) WHERE: default explain() captures NO call-site (opt-in => adds
        # nothing); explain(with_stack=True) captures the Python call-site of the op.
        def _do_fallback():
            a = torch.ones(16, 16, device=DEV, dtype=torch.int32)  # int32 -> cpu_fallback
            return a + a

        with torch.rbln.explain() as off:
            _do_fallback()
        self.assertEqual(off.dump()["trace_by_op"], {})  # nothing unless asked

        import torch_rbln._C as _C

        if not hasattr(_C, "_explain_set_trace"):
            self.skipTest("trace capture not exposed by this _C build")
        with torch.rbln.explain(with_stack=True) as on:
            _do_fallback()
        tbo = on.dump()["trace_by_op"]
        self.assertIn("aten::add.out", tbo)
        self.assertIn("_do_fallback", tbo["aten::add.out"])  # the user's call-site frame

    def test_with_stack_trace_is_deprecated_alias(self):
        # with_stack= is the torch-parity name; trace= is kept as a back-compat alias
        # and must set the same capture gate.
        self.assertTrue(torch.rbln.explain(trace=True)._trace)
        self.assertTrue(torch.rbln.explain(with_stack=True)._trace)
        self.assertFalse(torch.rbln.explain()._trace)

    def test_A_where_traces_bounce_site(self):
        # (A) WHERE also covers host BOUNCES (not just cpu_fallback/recompile): with
        # with_stack=True the bounced copy's Python call-site is captured under the site
        # name, so the report can point at the offending copy. Previously a bounce was
        # counted but unlocatable. OFF by default (a plain region captures nothing).
        import torch_rbln._C as _C

        if not hasattr(_C, "_explain_set_trace"):
            self.skipTest("trace capture not exposed by this _C build")

        def _do_bounce():
            s = torch.tensor([7, 8], dtype=torch.int64).to(DEV)
            return s.to(torch.int32)  # int d2d cast -> copy_d2d_host_bounce

        with torch.rbln.explain() as off:
            _do_bounce()
        self.assertEqual(off.dump()["trace_by_op"], {})  # opt-in: nothing unless asked

        with torch.rbln.explain(with_stack=True) as on:
            _do_bounce()
        d = on.dump()
        if d["hidden_host_bounce"]["by_site"]["copy_d2d_host_bounce"]["count"] < 1:
            self.skipTest("int cast did not bounce on this runtime")
        tbo = d["trace_by_op"]
        self.assertIn("copy_d2d_host_bounce", tbo)  # the bounce site was captured
        self.assertIn("_do_bounce", tbo["copy_d2d_host_bounce"])  # user's call-site frame
        rep = on.report()
        self.assertIn("host_bounce/d2d_copy:", rep)  # grouped detail block header (display label)
        self.assertIn("_do_bounce", rep)  # the captured call-site is shown under it

    def test_diff_reports_only_what_changed_between_two_regions(self):
        # explain doesn't know lifecycle; diff compares two regions the USER places.
        # int32 add falls back EVERY call -> persists; a stable fp16 shape compiles
        # once then hits warm cache -> its recompile is gone in the later region.
        a = torch.randn(40, 40, device=DEV, dtype=torch.float16)
        i = torch.ones(16, 16, device=DEV, dtype=torch.int32)
        with torch.rbln.explain() as r1:  # "early": first use of the fp16 shape
            _ = a + a
            _ = i + i
        for _ in range(2):
            _ = a + a  # warm the fp16 shape (USER-supplied structure)
        with torch.rbln.explain() as r2:  # "later": fp16 shape now warm
            _ = a + a
            _ = i + i
        dd = r1.diff(r2).dump()
        # int32 fallback recurs across both -> persists; recompile does not grow.
        self.assertGreaterEqual(dd["signals"]["cpu_fallback"]["b"], 1)
        self.assertIn("cpu_fallback", dd["persists"])
        self.assertLessEqual(dd["signals"]["recompile"]["b"], dd["signals"]["recompile"]["a"])
        self.assertIsInstance(r1.diff(r2).report(), str)


@pytest.mark.test_set_ci
class TestProfilerHostCostContext(TestCase):
    """Host-cost context signals: (A) which fallback ops lack a fast-path handler,
    (E) host CPU oversubscription, (B) time inside the runtime's copy calls vs
    torch-side dispatch."""

    def test_A_unaccelerated_lists_only_unhandled_fallback_ops(self):
        import torch_rbln._C as _C

        if not hasattr(_C, "_cpu_fast_path_registered"):
            self.skipTest("fast-path registry query not exposed by this _C build")
        a = torch.tensor([5, 3, 9, 1], dtype=torch.int32).to(DEV)
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(DEV)
        with torch.rbln.explain() as p:
            _ = a - b  # int -> cpu_fallback
        d = p.dump()
        fbo = d["cpu_fallback_by_op"]
        unaccel = d.get("cpu_fallback_unaccelerated", [])
        # self-consistency (robust to handlers landing later): every listed op is a
        # fallback op with no registered handler; any handled fallback op is excluded.
        for op in unaccel:
            self.assertIn(op, fbo)
            self.assertFalse(_C._cpu_fast_path_registered(op))
        for op in fbo:
            if _C._cpu_fast_path_registered(op):
                self.assertNotIn(op, unaccel)
        if unaccel:
            self.assertIn("no fast-path handler", p.report())

    def test_E_host_threads_resource_fact(self):
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        ht = p.dump()["host_threads"]
        self.assertGreaterEqual(ht["cores"], 1)
        self.assertIsInstance(ht["oversubscribed"], bool)
        # oversubscribed iff intended host parallelism exceeds the allowed cores.
        self.assertEqual(ht["oversubscribed"], ht["cores"] > 0 and ht["intended_threads"] > ht["cores"])

    def test_B_runtime_time_split(self):
        import torch_rbln._C as _C
        from torch_rbln.profiler import _RT_PRIMS

        if not hasattr(_C, "_rt_timing_get"):
            self.skipTest("rt-timing not exposed by this _C build")
        a = torch.tensor([5, 3, 9, 1], dtype=torch.int32).to(DEV)
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(DEV)
        with torch.rbln.explain() as p:
            for _ in range(20):
                _ = (a - b).cpu()  # fallback v2h/h2v + the .cpu() v2h copy calls
        rt = p.dump().get("runtime")
        self.assertIsNotNone(rt)
        self.assertGreaterEqual(rt["total_ns"], 0)
        self.assertGreaterEqual(rt["wall_fraction"], 0.0)
        self.assertLessEqual(rt["wall_fraction"], 1.0)
        for prim, vv in rt["by_primitive"].items():
            self.assertIn(prim, _RT_PRIMS)
            self.assertGreater(vv["calls"], 0)
        self.assertGreaterEqual(rt["by_primitive"]["v2h"]["calls"], 20)
        self.assertIn("runtime:", p.report())

    def test_B_gated_off_outside_region(self):
        # ON==OFF guard: runtime copies OUTSIDE any explain region must NOT be counted
        # (the timers are gated off + reset at region entry).
        import torch_rbln._C as _C

        if not hasattr(_C, "_rt_timing_get"):
            self.skipTest("rt-timing not exposed by this _C build")
        a = torch.tensor([1, 2], dtype=torch.int32).to(DEV)
        _ = (a - a).cpu()  # copy calls OUTSIDE a region -> gate off -> uncounted
        with torch.rbln.explain() as p:
            pass  # empty region
        rt = p.dump().get("runtime")
        self.assertIsNotNone(rt)
        self.assertEqual(rt["total_ns"], 0)


@pytest.mark.test_set_ci
class TestProfilerReportFormat(TestCase):
    """Output-format contract: torch.profiler-style, ASCII, torch-parity time units."""

    def test_fmt_time_matches_torch_format_time(self):
        # Adaptive us/ms/s with 3 decimals, no space, ASCII "us" -- byte-identical to
        # torch.autograd.profiler_util._format_time (explain reads like a torch table).
        from torch_rbln.profiler import _fmt_time

        self.assertEqual(_fmt_time(500), "0.500us")
        self.assertEqual(_fmt_time(190_000), "190.000us")
        self.assertEqual(_fmt_time(66_430_000), "66.430ms")
        self.assertEqual(_fmt_time(5_790_680_000), "5.791s")

    def test_note_prefix_marks_a_suggestion(self):
        from torch_rbln.profiler import _fix

        self.assertEqual(_fix("make contiguous"), "try: make contiguous")
        self.assertEqual(_fix(""), "")  # empty note stays empty (no bare prefix)

    def test_table_uses_double_dash_for_na_and_is_ascii(self):
        from torch_rbln.profiler import _table

        out = "\n".join(
            _table(
                ["Signal", "Count", "Bytes", "Note"],
                [["dispatch/cpu_fallback", "3", "--", "try: graph mode"]],
                ["l", "r", "r", "l"],
            )
        )
        self.assertIn("--", out)  # torch's not-applicable token
        self.assertTrue(out.isascii())  # no Unicode glyphs (log/Slack safe)

    def test_repr_before_stop_is_placeholder_not_report(self):
        # __repr__ mirrors torch._dynamo.explain's ExplainOutput (object renders as its
        # report), but a still-open region has no data yet -> a safe placeholder, no crash.
        p = torch.rbln.explain()
        self.assertIn("active", repr(p))

    def test_str_equals_report_after_region(self):
        with torch.rbln.explain() as p:
            pass
        self.assertEqual(str(p), p.report())  # print(p) == print(p.report())

    def test_report_is_ascii_and_header_labels(self):
        with torch.rbln.explain() as p:
            x = torch.randn(8, 8, device=DEV, dtype=torch.float16)
            _ = x + 1
        rep = p.report()
        self.assertTrue(rep.isascii())  # P6: ASCII-only output
        self.assertIn("RBLN EXPLAIN", rep)
        self.assertIn("region wall", rep)  # P1-5: the wall label carries the 'region' qualifier
        if "mem " in rep:
            self.assertIn("peak, reserved", rep)  # P0-3: caching allocator reserved footprint, labeled

    def test_report_format_contract(self):
        # Lock the report shape without needing hardware: feed a canned dump through the
        # real verdict()/report() and assert the structural contract (grouped blocks, fixed
        # row order, region-wall label, thousands separators, the runtime context line).
        import types

        from torch_rbln import profiler as _p

        gb, kb = 1024**3, 1024
        dump = {
            "wall_ns": 5_791_000_000,
            "device_memory": {"peak_bytes": 5 * gb, "current_bytes": 4 * gb, "devices": 1},
            "host_threads": {"oversubscribed": True, "intended_threads": 8, "cores": 1},
            "runtime": {
                "total_ns": 24_673_000,
                "wall_fraction": 0.004,
                "by_primitive": {"h2v": {"ns": 7_289_000, "calls": 3600}, "v2h": {"ns": 5_455_000, "calls": 2400}},
            },
            "hidden_host_bounce": {
                "by_site": {"copy_d2d_host_bounce": {"count": 1600, "bytes": 12 * kb}},
                "total_count": 1600,
                "total_bytes": 12 * kb,
            },
            "dispatch": {"cpu_fallback": 1600, "recompile_miss": 0, "cpu_fallback_ns": 41_084_000},
            "cpu_fallback_by_op": {"aten::sub.out": 800},
            "cpu_fallback_reasons": {"dtype-not-fp16": 1600},
            "trace_by_op": {},
            "notes": [],
        }
        o = types.SimpleNamespace()
        o.dump = lambda: dump
        o.verdict = types.MethodType(_p.RBLNExplain.verdict, o)
        rep = _p.RBLNExplain.report(o)

        self.assertTrue(rep.isascii())
        self.assertIn("[overhead: 2 signals]", rep)  # marker carries a factual signal-kind count
        self.assertIn("region wall", rep)  # P1-5
        self.assertIn("device mem 5.00 GB peak, reserved)", rep)  # P0-3
        self.assertIn("runtime: 24.673ms in copy calls (0.4% of region wall)", rep)
        self.assertIn("h2v 7.3ms/3600  v2h 5.5ms/2400", rep)  # per-primitive, sorted by time
        self.assertIn("host_bounce/d2d_copy", rep)  # P1-6 trimmed label
        self.assertIn("1,600", rep)  # P1-7 thousands separators
        self.assertIn("try: " + _p._FIX_SHORT["copy_d2d_host_bounce"], rep)  # the fix leads the Note
        self.assertIn("dispatch/cpu_fallback:", rep)  # P1-1 grouped detail block
        self.assertIn("where? -> rerun with explain(with_stack=True)", rep)
        # P1-4 fixed category order (NOT sorted by cost): host_bounce -> dispatch
        self.assertLess(rep.index("host_bounce/d2d_copy"), rep.index("dispatch/cpu_fallback"))

    def test_help_resolves_report_labels(self):
        # help() takes the label the report shows, including the trimmed host_bounce ones.
        from torch_rbln import profiler as _p

        p = _p.RBLNExplain()
        p.dump = lambda: {"notes": []}
        self.assertEqual(p.help("host_bounce/d2d_copy"), _p._REMEDY["copy_d2d_host_bounce"])
        self.assertEqual(p.help("host_bounce/strided_v2v_cpu_fallback"), _p._REMEDY["strided_v2v_cpu_fallback"])
        self.assertEqual(p.help("dispatch/cpu_fallback"), _p._REMEDY["cpu_fallback"])


@pytest.mark.test_set_ci
class TestProfilerRegionSafety(TestCase):
    """Regions own process-global instrumentation, so they must not overlap. A guard raises
    on overlap, and cleanup is guaranteed on any error (failed start OR mid-region raise)."""

    def test_region_released_on_exception(self):
        # start()+stop() release the global guard/gate via try/finally even if the body
        # raises, so instrumentation never leaks ON into the next region.
        from torch_rbln import profiler as _prof

        self.assertFalse(_prof._active)
        with self.assertRaises(ValueError):
            with torch.rbln.explain():
                _ = torch.ones(4, 4, device=DEV, dtype=torch.int32) + 1
                raise ValueError("boom")
        self.assertFalse(_prof._active)  # released despite the exception

    def test_overlapping_region_raises(self):
        # Regions must not overlap: a nested (or concurrent) region raises rather than
        # silently corrupting the open region's timer/trace/counters. The outer region
        # still releases the guard on the way out.
        from torch_rbln import profiler as _prof

        self.assertFalse(_prof._active)
        with self.assertRaises(RuntimeError):
            with torch.rbln.explain():
                self.assertTrue(_prof._active)
                with torch.rbln.explain():  # overlap -> raises on start
                    pass
        self.assertFalse(_prof._active)  # outer released on the way out


@pytest.mark.test_set_ci
class TestProfilerRuntimeContract(TestCase):
    """Guard the positional contracts with _C. If a probe's cardinality drifts (a site,
    primitive or reason added or removed), these FAIL in CI -- *detection* -- instead of
    the Python mapping silently truncating (zip) or mislabelling counts. An equal-cardinality
    reorder is NOT catchable from here; the enums and these tuples must change together."""

    def test_positional_axis_lengths_match_runtime(self):
        from torch_rbln.profiler import (
            _BOUNCE_SITES,
            _FALLBACK_REASON_NAMES,
            _read_bounces,
            _read_fallback_reasons,
            _read_rt_timing,
            _RT_PRIMS,
        )

        # BounceSite: core binding, always present.
        self.assertEqual(
            len(_read_bounces()), len(_BOUNCE_SITES), "BounceSite count drifted; sync _BOUNCE_SITES with RBLNProfiler.h"
        )
        # RtIdx primitives (if the (B) timer binding is present).
        rt = _read_rt_timing()
        if rt is not None:
            self.assertEqual(len(rt), len(_RT_PRIMS), "RtIdx count drifted; sync _RT_PRIMS with RBLNFunctions.cpp")
        # cpu_fallback reason histogram (if the binding is present).
        fr = _read_fallback_reasons()
        if fr:
            self.assertEqual(len(fr), len(_FALLBACK_REASON_NAMES), "fallback-reason count drifted; sync names")

    def test_dump_maps_rt_timing_positionally(self):
        # Device-free: the (ns, calls) pairs map 1:1 onto _RT_PRIMS in order, a primitive
        # with no call is left out, and the total and wall fraction cover every primitive.
        from torch_rbln.profiler import _BOUNCE_SITES, _RT_PRIMS

        p = torch.rbln.explain()
        p._bounces = [(0, 0)] * len(_BOUNCE_SITES)
        p._dispatch = (0, 0, 0, 0, 0, 0, 0)
        p._fallback_by_op, p._recompile_by_op, p._fallback_reasons = {}, {}, []
        p._trace_by_op, p._wall_ns = {}, 1_000_000
        p._rt_timing = [(1000 * (i + 1), 0 if i == 1 else i + 1) for i in range(len(_RT_PRIMS))]
        rt = p.dump()["runtime"]
        expected = {name: {"ns": 1000 * (i + 1), "calls": i + 1} for i, name in enumerate(_RT_PRIMS) if i != 1}
        self.assertEqual(rt["by_primitive"], expected)
        self.assertEqual(rt["total_ns"], sum(1000 * (i + 1) for i in range(len(_RT_PRIMS))))
        self.assertAlmostEqual(rt["wall_fraction"], rt["total_ns"] / 1_000_000)


if __name__ == "__main__":
    run_tests()
