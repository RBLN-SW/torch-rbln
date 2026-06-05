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
        # test_empty_region_is_green cover the unconditional-GREEN case.
        with torch.rbln.explain() as p:
            x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
            y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
            y.copy_(x)
        self.assertEqual(p.dump()["hidden_host_bounce"]["total_count"], 0)

    def test_noncontiguous_d2d_copy_bounces_with_bytes(self):
        with torch.rbln.explain() as p:
            big = torch.randn(512, 512, device=DEV, dtype=torch.float16)
            dst = torch.empty(512, 256, device=DEV, dtype=torch.float16)
            dst.copy_(big[:, :256])  # non-contiguous src -> NOT direct -> host round-trip
        hb = p.dump()["hidden_host_bounce"]
        self.assertGreaterEqual(hb["by_site"]["copy_d2d_host_bounce"]["count"], 1)
        self.assertGreater(hb["total_bytes"], 0)  # real bytes were moved through host
        self.assertEqual(p.verdict()["status"], "RED")


@pytest.mark.test_set_ci
class TestProfilerDispatchSignals(TestCase):
    """A (recompile) and B (cpu_fallback) read from the existing dispatch counters."""

    def test_B_cpu_fallback_counted(self):
        with torch.rbln.explain() as p:
            a = torch.ones(32, 32, device=DEV, dtype=torch.int32)  # non-fp16 -> CPU fallback
            _ = a + a
        self.assertGreaterEqual(p.dump()["dispatch"]["cpu_fallback"], 1)
        self.assertIn(p.verdict()["status"], ("AMBER", "RED"))

    def test_A_recompile_counted(self):
        # First use of an unusual shape forces a compile; the profiler must count
        # it as a recompile/miss (the actionable "you are recompiling" signal).
        # NOTE: whether a *repeat* is served from warm cache is a property of the
        # warm cache, not of the profiler, so it is deliberately not asserted here.
        with torch.rbln.explain() as p:
            x = torch.randn(29, 31, device=DEV, dtype=torch.float16)
            _ = x + x
        self.assertGreaterEqual(p.dump()["dispatch"]["recompile_miss"], 1)
        self.assertIn(p.verdict()["status"], ("AMBER", "RED"))  # a compile -> not GREEN


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

    def test_D_command_stream_is_not_a_verdict_signal(self):
        # Command-stream count / structural padding are intentionally excluded
        # from the verdict (measurable but not user-actionable).
        with torch.rbln.explain() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        keys = set(p.verdict().keys())
        for forbidden in ("command_stream", "cs_count", "padding", "fragmentation"):
            self.assertNotIn(forbidden, keys)

    def test_empty_region_is_green(self):
        with torch.rbln.explain() as p:
            pass
        self.assertEqual(p.verdict()["status"], "GREEN")
        self.assertEqual(p.verdict()["hidden_host_bounces"], 0)

    def test_regions_are_independent_deltas(self):
        with torch.rbln.explain() as p1:
            big = torch.randn(256, 256, device=DEV, dtype=torch.float16)
            dst = torch.empty(256, 128, device=DEV, dtype=torch.float16)
            dst.copy_(big[:, :128])
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

    def test_explain_steady_isolates_cold_compile(self):
        # explain_steady profiles the WARM (steady-state) call. A stable-shape op
        # compiles once (cold) then hits the warm cache, so steady-state recompile
        # must be <= the cold sample's. This is the one-time-vs-every-step split.
        a = torch.randn(48, 48, device=DEV, dtype=torch.float16)
        cold, warm = torch.rbln.explain_steady(lambda: a + a, warmup=3, return_cold=True)
        self.assertLessEqual(warm.dump()["dispatch"]["recompile_miss"], cold.dump()["dispatch"]["recompile_miss"])
        self.assertIn(warm.verdict()["status"], ("GREEN", "AMBER", "RED"))


if __name__ == "__main__":
    run_tests()
