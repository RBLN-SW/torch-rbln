# Owner(s): ["module: PrivateUse1"]

"""User-level tests for ``torch.rbln.profile()`` — the hidden-overhead profiler.

A normal PyTorch op can silently round-trip the host or fall back to CPU for
reasons the user never asked for. These tests drive real ops on the ``rbln``
device and assert the profiler surfaces those hidden events, and ONLY those.

Mapping to the profiler signal taxonomy discussed in design (A-E):

  * copy host-bounce  -> ``hidden_host_bounce``   (torch-side, this build) [core]
  * A recompile       -> ``dispatch.recompile_miss`` (torch-side, this build)
  * B cpu_fallback    -> ``dispatch.cpu_fallback``    (torch-side, this build)
  * C device idle     -> pending: owned by the rebel-compiler runtime collector,
                         NOT wired in this build -> must be reported as pending,
                         never as a false GREEN.
  * D command-stream  -> deliberately NOT a verdict signal (not user-actionable).
  * E memory gauge    -> pending: rebel-compiler runtime collector.

The C/D/E tests therefore assert the profiler's *honesty and scope*, not a
number we cannot yet measure truthfully.
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
        with torch.rbln.profile() as p:
            x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
            y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
            y.copy_(x)
        self.assertEqual(p.dump()["hidden_host_bounce"]["total_count"], 0)

    def test_noncontiguous_d2d_copy_bounces_with_bytes(self):
        with torch.rbln.profile() as p:
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
        with torch.rbln.profile() as p:
            a = torch.ones(32, 32, device=DEV, dtype=torch.int32)  # non-fp16 -> CPU fallback
            _ = a + a
        self.assertGreaterEqual(p.dump()["dispatch"]["cpu_fallback"], 1)
        self.assertIn(p.verdict()["status"], ("AMBER", "RED"))

    def test_A_recompile_counted(self):
        # First use of an unusual shape forces a compile; the profiler must count
        # it as a recompile/miss (the actionable "you are recompiling" signal).
        # NOTE: whether a *repeat* is served from warm cache is a property of the
        # warm cache, not of the profiler, so it is deliberately not asserted here.
        with torch.rbln.profile() as p:
            x = torch.randn(29, 31, device=DEV, dtype=torch.float16)
            _ = x + x
        self.assertGreaterEqual(p.dump()["dispatch"]["recompile_miss"], 1)
        self.assertIn(p.verdict()["status"], ("AMBER", "RED"))  # a compile -> not GREEN


@pytest.mark.test_set_ci
class TestProfilerTruthfulnessAndScope(TestCase):
    """The verdict must never claim more truth than it has (C/E), and must not
    carry signals the user cannot act on (D)."""

    def test_runtime_signals_present_or_honestly_pending(self):
        # When the loaded librbln exposes the runtime counters, the C/D/E sections
        # (host traffic bytes, command streams, device idle, device memory) must
        # be present. On an older runtime, the profiler must HONESTLY mark them
        # pending — never a false clean.
        with torch.rbln.profile() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        d = p.dump()
        rr = d["runtime_residency"]
        if rr["available"]:
            self.assertIn("total_count", rr)
            self.assertIn("by_reason", rr)
            self.assertIn("runtime_host_traffic", d)  # h2v/d2h leaf bytes
            self.assertIn("command_streams", d)  # D
            self.assertIn("device_idle", d)  # C
            self.assertIn("device_memory", d)  # E
        else:
            pending = " ".join(d["pending_runtime_signals"]).lower()
            self.assertIn("d2h", pending)

    def test_E_device_memory_gauge(self):
        # Pure allocation (no compute) — the gauge is a process-level high-water
        # mark, so it reads a sane non-zero peak >= current.
        with torch.rbln.profile() as p:
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
        with torch.rbln.profile() as p:
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
        with torch.rbln.profile() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        keys = set(p.verdict().keys())
        for forbidden in ("command_stream", "cs_count", "padding", "fragmentation"):
            self.assertNotIn(forbidden, keys)

    def test_empty_region_is_green(self):
        with torch.rbln.profile() as p:
            pass
        self.assertEqual(p.verdict()["status"], "GREEN")
        self.assertEqual(p.verdict()["hidden_host_bounces"], 0)

    def test_regions_are_independent_deltas(self):
        with torch.rbln.profile() as p1:
            big = torch.randn(256, 256, device=DEV, dtype=torch.float16)
            dst = torch.empty(256, 128, device=DEV, dtype=torch.float16)
            dst.copy_(big[:, :128])
        self.assertGreaterEqual(p1.dump()["hidden_host_bounce"]["total_count"], 1)
        # a fresh region must not inherit the previous region's incidents.
        with torch.rbln.profile() as p2:
            x = torch.randn(64, 64, device=DEV, dtype=torch.float16)
            y = torch.empty(64, 64, device=DEV, dtype=torch.float16)
            y.copy_(x)
        self.assertEqual(p2.dump()["hidden_host_bounce"]["total_count"], 0)


@pytest.mark.test_set_ci
class TestProfilerApi(TestCase):
    def test_report_is_str_and_dump_shape(self):
        with torch.rbln.profile() as p:
            _ = torch.randn(8, 8, device=DEV, dtype=torch.float16)
        self.assertIsInstance(p.report(), str)
        d = p.dump()
        self.assertIn("hidden_host_bounce", d)
        self.assertIn("dispatch", d)
        self.assertIn("runtime_residency", d)
        self.assertIn("pending_runtime_signals", d)


if __name__ == "__main__":
    run_tests()
