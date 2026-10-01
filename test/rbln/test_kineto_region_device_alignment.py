# Owner(s): ["module: PrivateUse1"]

"""torch.profiler shows the work the rbln runtime ran on a row per NPU, tied to the host op that
launched it.

Runs models on an NPU under ``torch.profiler`` (CPU + PrivateUse1) with ``RBLN_PROFILER=1``, which
the rbln kineto plugin reads when the torch.profiler session starts. The exported chrome trace then
carries, for what the runtime recorded:

* a process row ``NPU <id>`` per NPU, with a lane (thread) per kind of activity: ``compute`` jobs as
  ``kernel`` slices, ``copy`` jobs as ``gpu_memcpy``, ``host`` steps of a run as
  ``privateuse1_runtime``, and ``collective`` jobs as ``privateuse1_driver``;
* a zero-length ``rbln launch`` marker on the host thread at each launch (where a thread queued work
  on a stream), with a ``launch_id`` that every activity of that launch carries too;
* an ``ac2g`` flow arrow from each marker to every activity of its launch.

For every launch the tests check that its activities sit on an NPU row in the lane of their kind
and start no earlier than the launch, and that the marker lies inside the host op that launched it:
the "Torch-Compiled Region" for graph mode, the aten op for eager. A broken clock anchor misplaces
activities by orders of magnitude, so either check fails. A run is queued on the stream, so its
device work may outlive the launching op; only the start is bounded.

The arrows are chrome-trace ``ph`` "s"/"f" events, which the slice checks never see, so one test
walks them directly: every marker owns one flow start, and the start's finishes land on exactly the
activities of its launch.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  -- registers the rbln device + the kineto plugin


DEVICE = torch.device("rbln:0")
DTYPE = torch.float16

REGION_NAME = "Torch-Compiled Region"  # torch.compile's per-region CPU slice
LAUNCH_MARKER = "rbln launch"  # zero-length slice at the host launch point
LAUNCH_ID_ARG = "launch_id"  # annotation on the marker + every activity of one launch
NPU_ROW_PREFIX = "NPU "  # process name of a device row
LANE_OF_CATEGORY = {
    "kernel": "compute",
    "gpu_memcpy": "copy",
    "privateuse1_runtime": "host",
    "privateuse1_driver": "collective",
}
FLOW_CAT = "ac2g"  # category of the flow-arrow events (ph "s" start / "f" finish)
RBLN_FLOW_BASE = 0xF0000000  # the flow-id block the emitter owns
RBLN_FLOW_SPAN = 0x01000000
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
def enable_rbln_profiler(monkeypatch):
    """The rbln plugin records only when RBLN_PROFILER=1 as the torch.profiler session starts."""
    monkeypatch.setenv("RBLN_PROFILER", "1")


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_rbln_profiler")
class TestKinetoRegionDeviceAlignment(TestCase):
    """rbln activities sit on NPU rows and line up with the host op that launched them."""

    @staticmethod
    def _profiled_events(run_in_window):
        """Profile ``run_in_window()`` (CPU + PrivateUse1); return the chrome trace events."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
            run_in_window()
            torch.rbln.synchronize()  # finish the device work inside the window
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

    @staticmethod
    def _slices(events):
        return [e for e in events if e.get("ph") == "X" and isinstance(e.get("name"), str)]

    def _npu_lanes(self, events):
        """(pid, tid) -> lane name for every lane of an ``NPU <id>`` row."""
        meta = [e for e in events if e.get("ph") == "M" and isinstance(e.get("args"), dict)]
        npus = {
            e["pid"]
            for e in meta
            if e.get("name") == "process_name" and str(e["args"].get("name", "")).startswith(NPU_ROW_PREFIX)
        }
        self.assertTrue(npus, f"no '{NPU_ROW_PREFIX}<id>' process row in the trace -- is RBLN_PROFILER=1?")
        return {
            (e["pid"], e["tid"]): e["args"].get("name")
            for e in meta
            if e.get("name") == "thread_name" and e["pid"] in npus
        }

    def _assert_launches_aligned(self, events, name_contains, min_launches):
        """Group slices by ``launch_id``. Each launch has one marker inside the launching op
        (a compiled region when ``name_contains`` is given, any CPU op otherwise), and its
        activities sit on an NPU lane of their kind and start no earlier than the launch."""
        slices = self._slices(events)
        lanes = self._npu_lanes(events)

        by_launch = {}
        for e in slices:
            lid = self._launch_id(e)
            if lid is not None:
                by_launch.setdefault(lid, []).append(e)
        self.assertGreaterEqual(
            len(by_launch),
            min_launches,
            f"expected >= {min_launches} launches with a '{LAUNCH_ID_ARG}' annotation, got {len(by_launch)}",
        )

        computing = 0
        for lid, group in by_launch.items():
            markers = [e for e in group if e["name"] == LAUNCH_MARKER]
            self.assertEqual(
                len(markers), 1, f"launch {lid}: expected one '{LAUNCH_MARKER}' marker, got {len(markers)}"
            )
            marker = markers[0]
            launch_ts = float(marker["ts"])

            enclosing = [
                e
                for e in slices
                if e.get("pid") == marker.get("pid")
                and e.get("tid") == marker.get("tid")
                and e is not marker
                and float(e["dur"]) > 0
                and float(e["ts"]) <= launch_ts <= float(e["ts"]) + float(e["dur"])
                and (name_contains is None or name_contains in e["name"])
            ]
            label = name_contains or "host op"
            self.assertTrue(enclosing, f"launch {lid}: '{LAUNCH_MARKER}' at ts={launch_ts} not inside any '{label}'")

            activities = [e for e in group if e is not marker]
            self.assertTrue(activities, f"launch {lid}: a marker with no activity")
            for act in activities:
                lane = lanes.get((act.get("pid"), act.get("tid")))
                self.assertIsNotNone(lane, f"launch {lid}: '{act['name']}' is not on an NPU row")
                self.assertEqual(
                    lane,
                    LANE_OF_CATEGORY.get(act.get("cat")),
                    f"launch {lid}: '{act['name']}' ({act.get('cat')}) sits in lane '{lane}'",
                )
                self.assertGreaterEqual(
                    float(act["ts"]),
                    launch_ts,
                    f"launch {lid}: '{act['name']}' starts {launch_ts - float(act['ts']):.1f}us before its launch",
                )
            computing += any(act.get("cat") == "kernel" for act in activities)
        self.assertGreaterEqual(computing, min_launches, "too few launches ran a compute job")

    def _assert_flow_arrows(self, events, min_launches):
        """Ids in our block must account for themselves: one flow start per ``rbln launch``
        marker, each start's finishes covering exactly the activities of that launch, and
        nothing left over -- so a foreign id landing inside the block breaks a count. Ids
        outside the block are another producer's. Arrows are ``ph`` "s"/"f", which the
        ``ph == "X"`` checks never see."""
        slices = self._slices(events)
        markers = [e for e in slices if e["name"] == LAUNCH_MARKER and self._launch_id(e) is not None]
        flows = [
            e
            for e in events
            if e.get("cat") == FLOW_CAT and RBLN_FLOW_BASE <= int(e["id"]) < RBLN_FLOW_BASE + RBLN_FLOW_SPAN
        ]
        starts = [e for e in flows if e.get("ph") == "s"]
        finishes = [e for e in flows if e.get("ph") == "f"]

        self.assertGreaterEqual(
            len(markers), min_launches, f"expected >= {min_launches} '{LAUNCH_MARKER}' markers, got {len(markers)}"
        )
        self.assertEqual(
            len(starts),
            len(markers),
            f"expected one '{FLOW_CAT}' flow start per '{LAUNCH_MARKER}' marker, got {len(starts)} for {len(markers)}"
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
                len(source),
                1,
                f"flow {flow}: start at ts={start['ts']} matches {len(source)} '{LAUNCH_MARKER}' markers",
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

            expected = [sl for sl in slices if self._launch_id(sl) == lid and sl is not source[0]]
            self.assertEqual(
                len(sinks),
                len(expected),
                f"flow {flow} (launch {lid}): {len(sinks)} arrows for {len(expected)} activities "
                f"{[sl['name'] for sl in expected]}",
            )

        self.assertFalse(finishes_by_flow, f"flow finishes with no start: {sorted(finishes_by_flow)}")

    def test_graph_runs_on_npu_rows_after_launch(self):
        events = self._two_graph_events()
        self.assertGreater(
            len([e for e in self._slices(events) if REGION_NAME in e["name"]]),
            0,
            f"no '{REGION_NAME}' CPU slice in the trace",
        )
        self._assert_launches_aligned(events, name_contains=REGION_NAME, min_launches=2)

    def test_flow_arrows_reach_the_activities(self):
        events = self._two_graph_events()
        self._assert_flow_arrows(events, min_launches=2)

    def test_eager_ops_on_npu_rows_after_launch(self):
        # Eager runs a compiled function per op too, but emits no "Torch-Compiled Region";
        # the launcher is the aten op.
        model = _MLP1().to(DEVICE, dtype=DTYPE)
        x = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
        model(x)  # compile before profiling
        torch.rbln.synchronize()

        events = self._profiled_events(lambda: model(x))
        self._assert_launches_aligned(events, name_contains=None, min_launches=1)


if __name__ == "__main__":
    run_tests()
