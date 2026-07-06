# Owner(s): ["module: PrivateUse1"]

"""User-level tests for ``torch.rbln.explain()`` — the hidden-overhead profiler.

A normal PyTorch op can silently round-trip the host or fall back to CPU for
reasons the user never asked for. These tests drive real ops on the ``rbln``
device and assert the profiler surfaces those hidden events, and ONLY those.

Mapping to the profiler signal taxonomy discussed in design (A-E):

  * copy host-bounce  -> ``hidden_host_bounce``   (torch-side) [core]
  * A recompile       -> ``dispatch.recompile_miss`` (torch-side)
  * B cpu_fallback    -> ``dispatch.cpu_fallback``    (torch-side)
  * runtime hidden-d2h cause -> ``runtime_residency`` (rebel runtime) [core]
  * C device idle / D command-stream / leaf-byte host-traffic -> deliberately
                         EXCLUDED. They are a different profiler's job
                         (dispatch/utilization, à la torch.profiler/nsys), out of
                         this profiler's hidden-overhead scope, and on the
                         executed path live in the TVM runtime where they read ~0
                         -- surfacing them would mislead. The one hidden signal
                         still missing (side-effect host-sync on the TVM
                         graph-exec path) is a tracked follow-up, not these.
  * E memory gauge    -> ``device_memory`` (rebel runtime). NOT a hidden-overhead
                         signal; kept only because it is accurate and complete
                         (every device alloc routes through BufferAllocator).

These tests therefore assert the profiler's *honesty and scope* — that it
carries the hidden-overhead signals plus the memory gauge, and EXCLUDES the
out-of-scope dispatch/utilization counters.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  -- registers the ``rbln`` device + ``torch.rbln``


DEV = "rbln"


@pytest.mark.test_set_ci
class TestProfilerCopyBounce(TestCase):
    """The core signal: a hidden host round-trip behind a plain ``copy_``."""

    def test_contiguous_copy_no_torch_side_bounce(self):
        # A contiguous same-shape/dtype copy_ does NOT take torch-rbln's own host
        # path (no torch-side bounce). NOTE: the runtime residency witness may
        # still flag a hidden d2h if device residency isn't established for these
        # freshly-created tensors — that is a real finding, not a test failure —
        # so we assert only the torch-side invariant here, and let
        # test_empty_region_is_clean cover the unconditional-clean case.
        with torch.rbln.explain() as p:
            x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
            y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
            y.copy_(x)
        self.assertEqual(p.dump()["hidden_host_bounce"]["total_count"], 0)

    def test_d2d_int_cast_bounces_with_bytes(self):
        # An int-dtype device->device cast (int64->int32) cannot use the fp16 on-device
        # v2v engine, so it round-trips the host -> explain shows copy_d2d_host_bounce
        # with bytes, so the region is not clean. (A plain non-contiguous fp16 slice copy
        # no longer bounces here -- the runtime's strided v2v now handles it on-device.)
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
    """explain must render each dispatch regime distinctly. The decode-path study
    showed a CPU fallback is NOT automatically a host transfer: int arithmetic on a
    contiguous, host-resident tensor falls back per op but stays host-resident (lazy
    v-mem, borrow chain) — so its only cost is fallback wall-time, not d2h/bounce.
    A non-contiguous input can't be borrowed and DOES round-trip host. These lock
    that distinction so a regression can't silently turn one regime into another.
    (host-bounce WITHOUT fallback is covered by TestProfilerCopyBounce.)"""

    def test_device_run_no_fallback_no_bounce(self):
        # Contiguous fp16 elementwise runs on-device: no CPU fallback, no host
        # bounce. (A one-time recompile may still occur; not asserted here.)
        x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
        for _ in range(3):
            _ = x * 2  # warm
        with torch.rbln.explain() as p:
            _ = x * 2
        d = p.dump()
        self.assertEqual(d["dispatch"]["cpu_fallback"], 0)
        self.assertEqual(d["hidden_host_bounce"]["total_count"], 0)

    def test_fallback_without_transfer(self):
        # int ops on a CONTIGUOUS host-resident tensor fall back EVERY op but stay
        # host-resident -> NO host bounce, NO hidden d2h. explain surfaces the
        # cpu_fallback count + its wall-time, and zero transfer signals.
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
        self.assertEqual(d["hidden_host_bounce"]["total_count"], 0)  # contiguous -> borrowed, no copy
        if d["runtime_residency"]["available"]:
            self.assertEqual(d["runtime_residency"]["total_count"], 0)

    def test_fallback_with_transfer(self):
        # An int op on a NON-CONTIGUOUS tensor cannot be borrowed (borrow rejects
        # non-contig), so the fallback routes through a real host copy -> explain
        # shows BOTH a cpu_fallback AND a host bounce with bytes.
        m = torch.arange(256, dtype=torch.int32).reshape(16, 16).to(DEV)
        _ = m.t() - 1  # warm
        with torch.rbln.explain() as p:
            _ = m.t() - 1
        d = p.dump()
        self.assertGreaterEqual(d["dispatch"]["cpu_fallback"], 1)
        self.assertGreaterEqual(d["hidden_host_bounce"]["total_count"], 1)
        self.assertGreater(d["hidden_host_bounce"]["total_bytes"], 0)


@pytest.mark.test_set_ci
class TestProfilerTruthfulnessAndScope(TestCase):
    """The verdict must never claim more truth than it has (C/E), and must not
    carry signals the user cannot act on (D)."""

    def test_runtime_signals_present_or_honestly_pending(self):
        # When the loaded librbln exposes the runtime counters, the in-scope
        # sections (hidden-d2h residency + the device-memory gauge) must be
        # present. The out-of-scope dispatch/utilization counters (leaf-byte
        # traffic, command streams, device idle) must NOT be surfaced. On an
        # older runtime, the profiler must HONESTLY mark them pending — never a
        # false clean.
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        d = p.dump()
        rr = d["runtime_residency"]
        if rr["available"]:
            self.assertIn("total_count", rr)
            self.assertIn("by_reason", rr)
            self.assertIn("device_memory", d)  # E (resource gauge, kept)
            # out of hidden-overhead scope — must not be surfaced:
            self.assertNotIn("runtime_host_traffic", d)  # leaf bytes
            self.assertNotIn("command_streams", d)  # D
            self.assertNotIn("device_idle", d)  # C
        else:
            pending = " ".join(d["pending_runtime_signals"]).lower()
            self.assertIn("d2h", pending)

    def test_E_device_memory_gauge(self):
        # Pure allocation (no compute) — the gauge is a process-level high-water
        # mark, so it reads a sane non-zero peak >= current.
        with torch.rbln.explain() as p:
            _ = torch.empty(2048, 2048, device=DEV, dtype=torch.float16)
        d = p.dump()
        if not d["runtime_residency"]["available"]:
            self.skipTest("runtime gauge not exposed by the loaded librbln")
        m = d["device_memory"]
        self.assertGreaterEqual(m["peak_bytes"], m["current_bytes"])
        self.assertGreater(m["peak_bytes"], 0)

    def test_runtime_hidden_d2h_is_cause_tagged(self):
        # A plain contiguous copy_ of a freshly-created tensor bounces at the
        # RUNTIME level (torch-side sees nothing). The profiler must not only
        # count it but attribute the CAUSE — and every incident must map to a
        # named reason (no unattributed/unknown).
        with torch.rbln.explain() as p:
            x = torch.randn(256, 256, device=DEV, dtype=torch.float16)
            y = torch.empty(256, 256, device=DEV, dtype=torch.float16)
            y.copy_(x)
        rr = p.dump()["runtime_residency"]
        if not rr["available"]:
            self.skipTest("runtime residency counter not exposed by the loaded librbln")
        self.assertGreaterEqual(rr["total_count"], 1)
        attributed = sum(v["count"] for v in rr["by_reason"].values())
        self.assertEqual(attributed, rr["total_count"])  # no unattributed hidden d2h
        self.assertTrue(any(v["count"] > 0 for v in rr["by_reason"].values()))

    def test_runtime_h2d_push_counted(self):
        # A device op (matmul) consuming host-latest inputs must push them to the
        # device -> the manager-emitted real_host_sync_h2d counter fires. Symmetric
        # to the d2h counter; without it the push direction (the lazy push at the
        # device-consume boundary) is invisible. A host-served write never reaches it.
        a = torch.randn(256, 256, dtype=torch.float16).to(DEV)  # host-latest (USER_VIEW)
        b = torch.randn(256, 256, dtype=torch.float16).to(DEV)
        _ = (a @ b).to("cpu")  # warm: compile/recompile out of the measured region
        a = torch.randn(256, 256, dtype=torch.float16).to(DEV)
        b = torch.randn(256, 256, dtype=torch.float16).to(DEV)
        with torch.rbln.explain() as p:
            _ = (a @ b).to("cpu")
        rr = p.dump()["runtime_residency"]
        if not rr.get("available") or "real_host_sync_h2d" not in rr:
            self.skipTest("runtime h2d counter not exposed by the loaded librbln")
        # both operands are host-latest -> each is pushed to device for the matmul.
        self.assertGreaterEqual(rr["real_host_sync_h2d"]["count"], 1)
        self.assertGreater(rr["real_host_sync_h2d"]["bytes"], 0)
        self.assertIn("host->device push", p.report())

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
        self.assertIn("runtime_residency", d)
        self.assertIn("pending_runtime_signals", d)

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
        self.assertIn("[overhead]", rep)
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
        # nothing); explain(trace=True) captures the Python call-site of the op.
        def _do_fallback():
            a = torch.ones(16, 16, device=DEV, dtype=torch.int32)  # int32 -> cpu_fallback
            return a + a

        with torch.rbln.explain() as off:
            _do_fallback()
        self.assertEqual(off.dump()["trace_by_op"], {})  # nothing unless asked

        import torch_rbln._C as _C

        if not hasattr(_C, "_explain_set_trace"):
            self.skipTest("trace capture not exposed by this _C build")
        with torch.rbln.explain(trace=True) as on:
            _do_fallback()
        tbo = on.dump()["trace_by_op"]
        self.assertIn("aten::add.out", tbo)
        self.assertIn("_do_fallback", tbo["aten::add.out"])  # the user's call-site frame

    def test_A_where_traces_bounce_site(self):
        # (A) WHERE also covers host BOUNCES (not just cpu_fallback/recompile): with
        # trace=True the bounced copy's Python call-site is captured under the site
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

        with torch.rbln.explain(trace=True) as on:
            _do_bounce()
        d = on.dump()
        if d["hidden_host_bounce"]["by_site"]["copy_d2d_host_bounce"]["count"] < 1:
            self.skipTest("int cast did not bounce on this runtime")
        tbo = d["trace_by_op"]
        self.assertIn("copy_d2d_host_bounce", tbo)  # the bounce site was captured
        self.assertIn("_do_bounce", tbo["copy_d2d_host_bounce"])  # user's call-site frame
        self.assertIn("at host_bounce/copy_d2d_host_bounce", on.report())

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
    (E) host CPU oversubscription, (B) rebel-runtime (librbln) boundary time vs
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

    def test_B_rebel_runtime_time_split(self):
        import torch_rbln._C as _C

        if not hasattr(_C, "_rt_timing_get"):
            self.skipTest("rt-timing not exposed by this _C build")
        a = torch.tensor([5, 3, 9, 1], dtype=torch.int32).to(DEV)
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(DEV)
        with torch.rbln.explain() as p:
            for _ in range(20):
                _ = (a - b).cpu()  # exercises borrow / return / acquire / v2h boundary calls
        rt = p.dump().get("rebel_runtime")
        self.assertIsNotNone(rt)
        self.assertGreaterEqual(rt["total_ns"], 0)
        self.assertGreaterEqual(rt["wall_fraction"], 0.0)
        self.assertLessEqual(rt["wall_fraction"], 1.0)
        prims = {"v2v", "v2v_multi", "borrow", "acquire", "return", "v2h", "h2v"}
        for prim, vv in rt["by_primitive"].items():
            self.assertIn(prim, prims)
            self.assertGreater(vv["calls"], 0)
        self.assertTrue(rt["by_primitive"])  # boundary calls happened -> recorded
        self.assertIn("rebel runtime", p.report())

    def test_B_gated_off_outside_region(self):
        # ON==OFF guard: librbln work OUTSIDE any explain region must NOT be counted
        # (the timers are gated off + reset at region entry).
        import torch_rbln._C as _C

        if not hasattr(_C, "_rt_timing_get"):
            self.skipTest("rt-timing not exposed by this _C build")
        a = torch.tensor([1, 2], dtype=torch.int32).to(DEV)
        _ = (a - a).cpu()  # boundary calls OUTSIDE a region -> gate off -> uncounted
        with torch.rbln.explain() as p:
            pass  # empty region
        rt = p.dump().get("rebel_runtime")
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

    def test_note_prefixes_split_action_from_fact(self):
        from torch_rbln.profiler import _fix, _fyi

        self.assertEqual(_fix("make contiguous"), "fix: make contiguous")
        self.assertEqual(_fyi("served on host"), "fyi: served on host")
        self.assertEqual(_fix(""), "")  # empty note stays empty (no bare prefix)
        self.assertEqual(_fyi(""), "")

    def test_table_uses_double_dash_for_na_and_is_ascii(self):
        from torch_rbln.profiler import _table

        out = "\n".join(
            _table(
                ["Signal", "Count", "Bytes", "Note"],
                [["dispatch/cpu_fallback", "3", "--", "fix: graph mode"]],
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
        if "mem " in rep:
            self.assertIn("peak (process)", rep)  # P5: gauge is process-wide, labeled


@pytest.mark.test_set_ci
class TestProfilerRegionSafety(TestCase):
    """Regions drive process-global instrumentation; a refcount owns it and cleanup is
    guaranteed on exception (so nested/concurrent regions and mid-region errors are safe)."""

    def test_refcount_released_on_exception(self):
        # stop()'s try/finally releases the global refcount even if the body raises,
        # so the timer/trace gate never leaks ON into the next region.
        from torch_rbln import profiler as _prof

        self.assertEqual(_prof._active_regions, 0)
        with self.assertRaises(ValueError):
            with torch.rbln.explain():
                _ = torch.ones(4, 4, device=DEV, dtype=torch.int32) + 1
                raise ValueError("boom")
        self.assertEqual(_prof._active_regions, 0)  # released despite the exception

    def test_nested_region_warns_and_preserves_outer(self):
        # A nested region warns and must NOT reset/disable the outer region's timer/trace.
        import warnings as _w

        from torch_rbln import profiler as _prof

        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            with torch.rbln.explain() as outer:
                self.assertEqual(_prof._active_regions, 1)
                with torch.rbln.explain():
                    self.assertEqual(_prof._active_regions, 2)  # nested counted
                    _ = torch.ones(4, 4, device=DEV, dtype=torch.int32) + 1
                self.assertEqual(_prof._active_regions, 1)  # inner released, outer still owns
                _ = torch.ones(4, 4, device=DEV, dtype=torch.int32) + 1  # outer keeps measuring
        self.assertEqual(_prof._active_regions, 0)
        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in rec))
        self.assertIsInstance(outer.dump(), dict)  # outer not corrupted by the nested region


if __name__ == "__main__":
    run_tests()
