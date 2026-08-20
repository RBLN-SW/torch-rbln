# Owner(s): ["module: PrivateUse1"]

"""rbln device slices must line up with the host op that launched them.

Runs models on a real NPU under ``torch.profiler`` (CPU + PrivateUse1) and, for every run,
checks the device slices it produced against the host CPU slice that launched them. Two
runtime modes, selected by ``RBLN_RUNTIME_FORCE_SYNC``:

* sync dispatch (``RBLN_RUNTIME_FORCE_SYNC=1``) -- the runtime blocks on device completion
  after each op, so the device work nests inside the launching op: we assert *containment*
  (device work starts at/after the launch AND ends inside the op).
* async dispatch (``RBLN_RUNTIME_FORCE_SYNC=0``, the default) -- the runtime submits and
  returns (RuntimeInstance uses ``RunAsync`` + in-flight tracking), so device work may
  outlive the launching op. We assert only the *start ordering* (device work never begins
  before its launch). NOTE: with the profiler on, per-run counter readback currently drains
  each op so the trace still comes out contained; asserting only the async-safe bound keeps
  this test valid once async profiling lands and the device work really does outlive the op.

The launching op is the innermost enclosing CPU slice the run's source marker sits in: the
"Torch-Compiled Region" for graph mode, the aten op for eager. A broken clock anchor
misplaces device slices by orders of magnitude, so either check fails.

Correlation is read from the ``launch_id`` annotation the rbln bridge stamps on both the
``rbln_run`` source marker and every device slice a run produced: we group trace slices by
``launch_id``, use the ``rbln_run`` marker to locate the enclosing op, and check the run's
device slices against it. (These two strings are the producer<->consumer contract; a drift
surfaces as zero correlated launches and fails loudly here.)

The arrows drawn from that correlation are chrome-trace ``ph`` "s"/"f" events, which the
slice checks never see, so one test walks them directly: every marker owns one flow start and
every start reaches a device slice of the same ``launch_id``.

Requires a live REBEL NPU and ``RBLN_PROFILER=1`` (device profiling must be on at runtime
creation, else no device slices are recorded).
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  -- registers the rbln device + the kineto bridge
from test.utils import is_atom_device


DEVICE = torch.device("rbln:0")
DTYPE = torch.float16

REGION_NAME = "Torch-Compiled Region"  # torch.compile's per-region CPU slice
RUN_MARKER = "rbln_run"  # the run's source marker (host launch point)
LAUNCH_ID_ARG = "launch_id"  # annotation on the source + every device slice of one run
FLOW_CAT = "ac2g"  # category of the flow-arrow events (ph "s" start / "f" finish)
RBLN_FLOW_BASE = 0xF0000000  # the flow-id block the emitter owns
RBLN_FLOW_SPAN = 0x01000000
DEVICE_CATS = ("kernel", "gpu_memcpy")
RUNTIME_CAT = "privateuse1_runtime"
TS_EPS_US = 1e-3  # a flow start sits on its marker's timestamp; allow json round-trip noise


class _MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def enable_rbln_device_profiler(monkeypatch, tmp_path):
    """Device profiling must be on at runtime creation for the kineto session to record slices."""
    monkeypatch.setenv("RBLN_PROFILER", "1")
    monkeypatch.setenv("RBLN_PROFILER_DIR", str(tmp_path))


@pytest.fixture
def force_sync_dispatch(monkeypatch):
    """Block on device completion after each runtime op (RuntimeInstance drains per op)."""
    monkeypatch.setenv("RBLN_RUNTIME_FORCE_SYNC", "1")


@pytest.fixture
def async_dispatch(monkeypatch):
    """Default async dispatch: the runtime submits and returns without blocking per op."""
    monkeypatch.setenv("RBLN_RUNTIME_FORCE_SYNC", "0")


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.skipif(
    is_atom_device(),
    reason="rbln kineto export is REBEL-only; rbln_kineto_is_active() reports inactive on ATOM",
)
@pytest.mark.usefixtures("enable_rbln_device_profiler")
class TestKinetoRegionDeviceAlignment(TestCase):
    """rbln device slices must line up with the host op that launched them."""

    @staticmethod
    def _profiled_events(run_in_window):
        """Profile ``run_in_window()`` (CPU + PrivateUse1); return the chrome trace events."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            run_in_window()
            torch.rbln.synchronize()  # flush device work into the window
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            prof.export_chrome_trace(path)
            with open(path) as fh:
                data = json.load(fh)
        finally:
            os.unlink(path)
        return data["traceEvents"] if isinstance(data, dict) else data

    def _two_graph_events(self):
        """Compile two distinct graphs, warm them up, then profile one run of each."""
        m1 = torch.compile(_MLP1().to(DEVICE, dtype=DTYPE), backend="rbln")
        m2 = torch.compile(_MLP2().to(DEVICE, dtype=DTYPE), backend="rbln")
        x1 = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
        x2 = torch.randn(4, 64, device=DEVICE, dtype=DTYPE)
        m1(x1), m2(x2)  # compile before profiling
        torch.rbln.synchronize()
        return self._profiled_events(lambda: (m1(x1), m2(x2)))

    @staticmethod
    def _launch_id(ev):
        args = ev.get("args")
        if not isinstance(args, dict) or LAUNCH_ID_ARG not in args:
            return None
        try:
            return int(str(args[LAUNCH_ID_ARG]))
        except (TypeError, ValueError):
            return None

    def _assert_launches_aligned(self, events, name_contains, min_launches, sync):
        """Group slices by ``launch_id``; for each run, its device slices start at/after the
        launch (always) and end inside the launching op when ``sync``. ``name_contains``
        filters that op (a compiled region); None means any CPU op (eager)."""
        slices = [e for e in events if e.get("ph") == "X" and isinstance(e.get("name"), str)]

        by_launch = {}
        for e in slices:
            lid = self._launch_id(e)
            if lid is not None:
                by_launch.setdefault(lid, []).append(e)
        self.assertGreaterEqual(
            len(by_launch),
            min_launches,
            f"expected >= {min_launches} launches with a '{LAUNCH_ID_ARG}' annotation, got {len(by_launch)} "
            "-- is RBLN_PROFILER=1 and the NPU runtime active? profiling must be on at runtime creation.",
        )

        checked = 0
        for lid, group in by_launch.items():
            sources = [e for e in group if e["name"] == RUN_MARKER]
            self.assertEqual(len(sources), 1, f"launch {lid}: expected one '{RUN_MARKER}' source, got {len(sources)}")
            src_ts = float(sources[0]["ts"])
            src_pid, src_tid = sources[0].get("pid"), sources[0].get("tid")

            enclosing = [
                e
                for e in slices
                if e.get("pid") == src_pid
                and e.get("tid") == src_tid
                and float(e["dur"]) > 0
                and float(e["ts"]) <= src_ts <= float(e["ts"]) + float(e["dur"])
                and (name_contains is None or name_contains in e["name"])
            ]
            label = name_contains or "host op"
            self.assertTrue(enclosing, f"launch {lid}: '{RUN_MARKER}' at ts={src_ts} not inside any '{label}'")
            host = max(enclosing, key=lambda e: float(e["ts"]))  # innermost = latest-starting
            h_end = float(host["ts"]) + float(host["dur"])

            device_slices = [e for e in group if e["name"] != RUN_MARKER]
            self.assertTrue(device_slices, f"launch {lid}: no device slices linked to the run")
            self.assertTrue(
                [e for e in group if e.get("cat") in DEVICE_CATS],
                f"launch {lid}: no {DEVICE_CATS} slice carries this launch_id",
            )
            for dev in device_slices:
                start = float(dev["ts"])
                end = start + float(dev.get("dur", 0.0))
                # MAIN: device work never starts before the run that launched it.
                self.assertGreaterEqual(
                    start, src_ts, f"launch {lid}: '{dev['name']}' starts {src_ts - start:.1f}us before its launch"
                )
                if sync:
                    # Sync dispatch: device work also ends inside the launching op.
                    self.assertLessEqual(
                        end, h_end, f"launch {lid}: '{dev['name']}' ends {end - h_end:.1f}us after '{host['name']}'"
                    )
                checked += 1
        self.assertGreater(checked, 0, "no device slices were checked")

    def _assert_flow_arrows(self, events, min_launches):
        """Ids in our block must account for themselves: one flow start per ``rbln_run``
        marker, each start's finishes covering exactly that launch's runtime slices, and
        nothing left over -- so a foreign id landing inside the block breaks a count. Ids
        outside the block are another producer's. Arrows are ``ph`` "s"/"f", which the
        ``ph == "X"`` checks never see."""
        slices = [e for e in events if e.get("ph") == "X" and isinstance(e.get("name"), str)]
        markers = [e for e in slices if e["name"] == RUN_MARKER and self._launch_id(e) is not None]
        flows = [
            e
            for e in events
            if e.get("cat") == FLOW_CAT and RBLN_FLOW_BASE <= int(e["id"]) < RBLN_FLOW_BASE + RBLN_FLOW_SPAN
        ]
        starts = [e for e in flows if e.get("ph") == "s"]
        finishes = [e for e in flows if e.get("ph") == "f"]

        self.assertGreaterEqual(
            len(markers), min_launches, f"expected >= {min_launches} '{RUN_MARKER}' markers, got {len(markers)}"
        )
        self.assertEqual(
            len(starts),
            len(markers),
            f"expected one '{FLOW_CAT}' flow start per '{RUN_MARKER}' marker, got {len(starts)} for {len(markers)}"
            " -- a surplus start means another producer emits ids inside the rbln block",
        )

        finishes_by_flow = {}
        for e in finishes:
            finishes_by_flow.setdefault(e["id"], []).append(e)

        seen = set()
        for start in starts:
            flow = start["id"]
            self.assertNotIn(flow, seen, f"flow {flow}: two starts share one flow id")
            seen.add(flow)

            source = [
                m
                for m in markers
                if m.get("pid") == start.get("pid")
                and m.get("tid") == start.get("tid")
                and abs(float(m["ts"]) - float(start["ts"])) <= TS_EPS_US
            ]
            self.assertEqual(
                len(source), 1, f"flow {flow}: start at ts={start['ts']} matches {len(source)} '{RUN_MARKER}' markers"
            )
            lid = self._launch_id(source[0])

            sinks = finishes_by_flow.pop(flow, [])
            self.assertTrue(sinks, f"flow {flow} (launch {lid}): start with no finish -- arrow reaches nothing")
            for fin in sinks:
                landed = [
                    s
                    for s in slices
                    if s.get("pid") == fin.get("pid")
                    and s.get("tid") == fin.get("tid")
                    and float(s["ts"]) <= float(fin["ts"]) <= float(s["ts"]) + float(s.get("dur", 0.0))
                ]
                self.assertTrue(landed, f"flow {flow}: finish at ts={fin['ts']} lands on no slice")
                self.assertIn(
                    lid,
                    [self._launch_id(s) for s in landed],
                    f"flow {flow}: finish landed on {[s['name'] for s in landed]}, none of launch {lid}",
                )

            expected = [
                sl
                for sl in slices
                if sl.get("cat") == RUNTIME_CAT and self._launch_id(sl) == lid and sl is not source[0]
            ]
            self.assertEqual(
                len(sinks),
                len(expected),
                f"flow {flow} (launch {lid}): {len(sinks)} arrows for {len(expected)} runtime slices "
                f"{[sl['name'] for sl in expected]}",
            )

        self.assertFalse(finishes_by_flow, f"flow finishes with no start: {sorted(finishes_by_flow)}")

    @pytest.mark.usefixtures("force_sync_dispatch")
    def test_sync_dispatch_device_contained(self):
        # RBLN_RUNTIME_FORCE_SYNC=1: the run blocks on the device, so device work nests in its region.
        events = self._two_graph_events()
        self.assertGreater(
            len([e for e in events if e.get("ph") == "X" and REGION_NAME in str(e.get("name"))]),
            0,
            f"no '{REGION_NAME}' CPU slice in the trace",
        )
        self._assert_launches_aligned(events, name_contains=REGION_NAME, min_launches=2, sync=True)

    @pytest.mark.usefixtures("async_dispatch")
    def test_async_dispatch_device_after_launch(self):
        # RBLN_RUNTIME_FORCE_SYNC=0 (async dispatch): assert only that device work starts after
        # its launch (it may outlive the region once async profiling no longer drains per run).
        events = self._two_graph_events()
        self._assert_launches_aligned(events, name_contains=REGION_NAME, min_launches=2, sync=False)

    @pytest.mark.usefixtures("force_sync_dispatch")
    def test_flow_arrows_reach_the_device_slices(self):
        events = self._two_graph_events()
        self._assert_flow_arrows(events, min_launches=2)

    @pytest.mark.usefixtures("force_sync_dispatch")
    def test_eager_ops_device_contained(self):
        # Eager compiles per op too, but emits no "Torch-Compiled Region"; the launcher is the aten op.
        model = _MLP1().to(DEVICE, dtype=DTYPE)
        x = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
        model(x)  # compile before profiling
        torch.rbln.synchronize()

        events = self._profiled_events(lambda: model(x))
        self._assert_launches_aligned(events, name_contains=None, min_launches=1, sync=True)


if __name__ == "__main__":
    run_tests()
