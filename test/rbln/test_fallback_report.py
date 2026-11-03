# Owner(s): ["module: PrivateUse1"]

"""The ``--fallback-report`` plugin in ``test/conftest.py`` and the window a test opens
with ``measure_cpu_fallbacks()``: a test's record is the CPU fallbacks inside its
windows and nothing else, and the records reach the JSON from a serial run and from
an xdist worker alike. The counter reader is replaced, so no NPU is involved.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from _pytest.reports import TestReport
from torch.testing._internal.common_utils import run_tests, TestCase

from test import utils
from test.conftest import CpuFallbackReport
from test.utils import measure_cpu_fallbacks


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in
# pyproject.toml.
pytestmark = pytest.mark.torch_rbln_only

_CONTROLLER = SimpleNamespace()  # no `workerinput`: a serial run, or xdist's controller
_WORKER = SimpleNamespace(workerinput={})  # an xdist worker

_SUITE = "test/ops/test_ops.py::TestCommonPRIVATEUSE1::"
_ADD, _MUL, _UNIFORM = ("aten::add.out", "dtype"), ("aten::mul.out", "align"), ("aten::uniform_", "registered")


def _finish(wrapper, value=None):
    """Resume a hook wrapper past its yield with `value`; returns what it returns."""
    try:
        wrapper.send(value)
    except StopIteration as done:
        return done.value
    msg = "the hook wrapper did not finish"
    raise AssertionError(msg)


def _run_call(plugin, nodeid, windows, outcome="passed"):
    """Drive the plugin through one test call whose body opens one window per
    (before, after) pair in `windows`, the counter reads at the window's ends; returns
    the call report as pytest would log it.
    """
    item = SimpleNamespace(nodeid=nodeid, name=nodeid.rsplit("::", 1)[1], stash=pytest.Stash())
    call = plugin.pytest_runtest_call(item)
    next(call)  # the plugin opens the test's recorder, then the test body runs
    for before, after in windows:
        with (
            mock.patch.object(utils, "_cpu_fallback_counts", side_effect=[before, after]),
            measure_cpu_fallbacks(),
        ):
            pass
    _finish(call)
    report = TestReport(nodeid, ("test_ops.py", 0, item.name), {}, outcome, None, "call")
    makereport = plugin.pytest_runtest_makereport(item, SimpleNamespace(when="call"))
    next(makereport)
    return _finish(makereport, report)


def _written(plugin):
    plugin.pytest_sessionfinish()
    return json.loads(plugin.path.read_text())


class TestCpuFallbackReport(TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "fallback.json"

    def test_each_test_gets_the_fallbacks_inside_its_own_windows(self):
        plugin = CpuFallbackReport(_CONTROLLER, self.path)
        # The counters only grow; a window's share is the growth over it, so an op that
        # already had a count gets the increase, not the total, and windows add up.
        first = _run_call(plugin, _SUITE + "test_compare_cpu_add_rbln_float16", [({}, {_ADD: 2})])
        second = _run_call(
            plugin,
            _SUITE + "test_compare_cpu_mul_rbln_float16",
            [({_ADD: 2}, {_ADD: 5}), ({_ADD: 5}, {_ADD: 5, _MUL: 1})],
        )
        third = _run_call(
            plugin, _SUITE + "test_compare_cpu_sub_rbln_float16", [({_ADD: 5, _MUL: 1}, {_ADD: 5, _MUL: 1})]
        )
        for report in (first, second, third):
            plugin.pytest_runtest_logreport(report)

        self.assertEqual(
            _written(plugin),
            {
                first.nodeid: {
                    "windows_total": 1,
                    "windows_with_fallbacks": 1,
                    "call_outcome": "passed",
                    "cpu_fallback": 2,
                    "cpu_fallback_reasons_by_op": {"aten::add.out": {"dtype": 2}},
                    "cpu_fallback_reset_detected": False,
                },
                second.nodeid: {
                    "windows_total": 2,
                    "windows_with_fallbacks": 2,
                    "call_outcome": "passed",
                    "cpu_fallback": 4,
                    "cpu_fallback_reasons_by_op": {"aten::add.out": {"dtype": 3}, "aten::mul.out": {"align": 1}},
                    "cpu_fallback_reset_detected": False,
                },
                third.nodeid: {  # no fallback is a record too: the op ran on the device
                    "windows_total": 1,
                    "windows_with_fallbacks": 0,
                    "call_outcome": "passed",
                    "cpu_fallback": 0,
                    "cpu_fallback_reasons_by_op": {},
                    "cpu_fallback_reset_detected": False,
                },
            },
        )

    def test_fallbacks_outside_the_windows_are_not_the_tests(self):
        plugin = CpuFallbackReport(_CONTROLLER, self.path)
        # Between two windows the harness fell back (input generation: uniform_); the
        # record carries the op under test only. A test that opens no window has no
        # record at all, whatever fell back during its call.
        windowed = _run_call(
            plugin,
            _SUITE + "test_compare_cpu_add_rbln_float16",
            [({}, {_ADD: 1}), ({_ADD: 1, _UNIFORM: 31}, {_ADD: 2, _UNIFORM: 31})],
        )
        unwindowed = _run_call(plugin, _SUITE + "test_out_add_rbln_float16", [])
        self.assertEqual(unwindowed.user_properties, [])
        for report in (windowed, unwindowed):
            plugin.pytest_runtest_logreport(report)

        self.assertEqual(
            _written(plugin),
            {
                windowed.nodeid: {
                    "windows_total": 2,
                    "windows_with_fallbacks": 2,
                    "call_outcome": "passed",
                    "cpu_fallback": 2,
                    "cpu_fallback_reasons_by_op": {"aten::add.out": {"dtype": 2}},
                    "cpu_fallback_reset_detected": False,
                }
            },
        )

    def test_the_window_is_a_no_op_without_a_report(self):
        # Outside a --fallback-report run no recorder is set, and the window does not
        # even read the counters.
        with mock.patch.object(utils, "_cpu_fallback_counts") as reads, measure_cpu_fallbacks():
            pass
        self.assertEqual(reads.call_count, 0)

    def test_a_failed_call_keeps_its_outcome_and_a_reset_is_marked(self):
        plugin = CpuFallbackReport(_CONTROLLER, self.path)
        # A count below where the window started means the test reset the counters
        # inside it, so its fallbacks can be missed.
        report = _run_call(
            plugin, _SUITE + "test_compare_cpu_add_rbln_float16", [({_ADD: 5}, {_ADD: 1})], outcome="failed"
        )
        plugin.pytest_runtest_logreport(report)

        self.assertEqual(
            _written(plugin),
            {
                report.nodeid: {
                    "windows_total": 1,
                    "windows_with_fallbacks": 0,
                    "call_outcome": "failed",
                    "cpu_fallback": 0,
                    "cpu_fallback_reasons_by_op": {},
                    "cpu_fallback_reset_detected": True,
                }
            },
        )

    def test_skipped_subtest_and_other_phase_reports_are_left_out(self):
        plugin = CpuFallbackReport(_CONTROLLER, self.path)
        skipped = _run_call(plugin, _SUITE + "test_compare_cpu_add_rbln_float16", [({}, {_ADD: 1})], outcome="skipped")
        subtest = _run_call(plugin, _SUITE + "test_compare_cpu_mul_rbln_float16", [({}, {_ADD: 1})])
        subtest.context = "a subtest"  # pytest-subtests marks a subtest's report so
        # A setup or teardown report never carries the counts; nor does a call report
        # whose counter read raised (the raise failed the call instead).
        sub = _SUITE + "test_compare_cpu_sub_rbln_float16"
        setup = TestReport(sub, ("test_ops.py", 0, "x"), {}, "passed", None, "setup")
        bare_call = TestReport(sub, ("test_ops.py", 0, "x"), {}, "failed", None, "call")
        for report in (skipped, subtest, setup, bare_call):
            plugin.pytest_runtest_logreport(report)

        self.assertEqual(_written(plugin), {})

    def test_an_xdist_worker_hands_its_records_to_the_controller(self):
        worker = CpuFallbackReport(_WORKER, Path(self._dir.name) / "worker.json")
        controller = CpuFallbackReport(_CONTROLLER, self.path)
        report = _run_call(worker, _SUITE + "test_compare_cpu_add_rbln_float16", [({}, {_ADD: 2})])
        # The worker logs the report too, but only the controller writes.
        worker.pytest_runtest_logreport(report)
        worker.pytest_sessionfinish()
        self.assertFalse(worker.path.exists())
        # xdist serializes a report to send it to the controller; the counts ride along
        # in its user properties.
        controller.pytest_runtest_logreport(TestReport._from_json(report._to_json()))

        self.assertEqual(
            _written(controller),
            {
                report.nodeid: {
                    "windows_total": 1,
                    "windows_with_fallbacks": 1,
                    "call_outcome": "passed",
                    "cpu_fallback": 2,
                    "cpu_fallback_reasons_by_op": {"aten::add.out": {"dtype": 2}},
                    "cpu_fallback_reset_detected": False,
                }
            },
        )


if __name__ == "__main__":
    run_tests()
