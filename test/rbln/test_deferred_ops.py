# Owner(s): ["module: PrivateUse1"]

"""
TORCH_RBLN_DEFER_OPS: eager ops that a stream defers and then runs as one function
give the same values as ops run as they are called.
"""

import os
import threading
import time
from unittest import mock

import pytest
import torch
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase

import torch_rbln._C as _C
from test.utils import run_in_isolated_process
from torch_rbln._internal import compile_cache


# Fusing can keep an intermediate in the device's dlfloat16 where an op run alone rounds it to
# float16, so a fused result can differ from an eager one by a float16 rounding per op.
_TOLERANCE = {"atol": 2e-2, "rtol": 2e-2}


def _deferring(on: bool = True):
    return mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "1" if on else "0"})


def _capped(limit: int):
    return mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS_MAX": str(limit)})


def _bucketing():
    return mock.patch.dict(os.environ, {"TORCH_RBLN_EAGER_BUCKETS": "1"})


def _counts() -> dict:
    return dict(_C._deferred_stats())


def _since(before: dict) -> dict:
    return {k: v - before[k] for k, v in _counts().items()}


def _segments(change: dict) -> int:
    return change["fused_runs"] + change["op_runs"]


def _pair(columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Tensors of a shape of their own, so that no other test runs the segments of a test.
    x, y = (torch.randn(8, columns, dtype=torch.float16) for _ in range(2))
    return x.to("rbln"), y.to("rbln")


def _chain(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    for i in range(n):
        if i % 4 == 0:
            x = x * 0.5
        elif i % 4 == 1:
            x = x + y
        elif i % 4 == 2:
            x = torch.neg(x)
        else:
            x = x - y
    return x


# Ops by letter, for the bodies of loops below.
_OPS = {
    "0": lambda x, y: x * 0.5,
    "1": lambda x, y: x + y,
    "2": lambda x, y: torch.neg(x),
    "3": lambda x, y: x - y,
    "4": lambda x, y: x * 0.25,
    "5": lambda x, y: x * y,
    "6": lambda x, y: torch.sigmoid(x),
    "7": lambda x, y: x + 0.5,
}

# Bodies of loops that no shorter run of ops repeats twice over, however many times the body
# repeats: a loop over one of them up to 128 ops long is found at the body's length.
_BODY10 = "0102010313"
_OTHER_BODY10 = "4546454757"
_BODY50 = "01020103010201202101201020120210130102010301020313"
_BODY130 = (
    "0102010301020120210120102012021013010201030102012021012010201202102010210120102012021012010201"
    "301020103010201202101201020120210323"
)
# Ops in which no run of ops repeats twice over.
_STRAIGHT40 = "0102012021012010201202102010210120102012"


def _body(x: torch.Tensor, y: torch.Tensor, word: str) -> torch.Tensor:
    for letter in word:
        x = _OPS[letter](x, y)
    return x


def _loop(x: torch.Tensor, y: torch.Tensor, word: str, laps: int) -> torch.Tensor:
    for _ in range(laps):
        x = _body(x, y, word)
    return x


def _prefixed_loop(x: torch.Tensor, y: torch.Tensor, laps: int) -> torch.Tensor:
    # Two ops no lap of the loop holds, before the loop.
    return _loop(torch.sigmoid(x) + 0.5, y, _BODY10, laps)


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestDeferredOps(TestCase):
    def setUp(self):
        self.x = torch.randn(8, 256, dtype=torch.float16)
        self.y = torch.randn(8, 256, dtype=torch.float16)
        self.xd = self.x.to("rbln")
        self.yd = self.y.to("rbln")

    # Twenty ops repeat four: the first run finds the loop at its eighth op, and the later runs follow
    # it from their first; each run is five laps.
    @parametrize("n, segments", [(5, 3), (20, 15)])
    def test_a_chain_gives_what_it_gives_eagerly(self, n, segments):
        with _deferring(False):
            expected = _chain(self.xd, self.yd, n).cpu()
        before = _counts()
        with _deferring():
            # Op by op the first time, compiled the second, from the cache the third.
            for _ in range(3):
                torch.testing.assert_close(_chain(self.xd, self.yd, n).cpu(), expected, **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 3 * n)
        self.assertEqual(_segments(change), segments)
        self.assertGreaterEqual(change["fused_runs"], 2)

    def test_a_repeated_segment_runs_its_function(self):
        with _deferring():
            for _ in range(2):
                _chain(self.xd, self.yd, 6).cpu()
            before = _counts()
            for _ in range(3):
                _chain(self.xd, self.yd, 6).cpu()
        change = _since(before)
        self.assertEqual(change["fused_runs"], 3)
        self.assertEqual(change["compiled"], 0)
        self.assertEqual(change["op_runs"], 0)

    def test_a_segment_the_compiler_refuses_runs_op_by_op(self):
        # A shape no other test runs, so that no other test compiled the segment first.
        x, y = self.xd[:, :192].contiguous(), self.yd[:, :192].contiguous()
        with _deferring(False):
            expected = _chain(x, y, 6).cpu()

        def refuse(*args, **kwargs):
            raise RuntimeError("refused for the test")

        with _deferring():
            _chain(x, y, 6).cpu()
            before = _counts()
            with mock.patch.object(compile_cache, "compile_logical", refuse):
                for _ in range(3):
                    torch.testing.assert_close(_chain(x, y, 6).cpu(), expected, **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["refused"], 1)
        self.assertEqual(change["op_runs"], 3)
        self.assertEqual(change["fused_runs"], 0)

    def test_defer_ops_max_bounds_a_segment(self):
        before = _counts()
        with _deferring(), _capped(3):
            _chain(self.xd, self.yd, 7).cpu()
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 7)
        self.assertEqual(_segments(change), 3)

    def test_straight_line_ops_run_where_their_results_are_needed(self):
        x, y = _pair(576)
        with _deferring():
            before = _counts()
            r = _body(x, y, _STRAIGHT40)
            self.assertEqual(_segments(_since(before)), 0)
            result = r.cpu()
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 40)
        self.assertEqual(_segments(change), 1)
        self.assertEqual(change["loops_found"], 0)
        torch.testing.assert_close(result, _body(x.cpu(), y.cpu(), _STRAIGHT40), **_TOLERANCE)

    def test_a_loop_compiles_its_lap_once_and_runs_later_laps_fused(self):
        x, y = _pair(640)
        with _deferring(False):
            expected = _loop(x, y, _BODY10, 12).cpu()
        with _deferring():
            before = _counts()
            r = _loop(x, y, _BODY10, 12)
            # The first two laps are found together, and each later lap runs once the next op comes.
            self.assertEqual(_segments(_since(before)), 11)
            result = r.cpu()
        change = _since(before)
        self.assertEqual(change["loops_found"], 1)
        self.assertEqual(change["compiled"], 1)
        self.assertEqual(change["op_runs"], 1)
        self.assertEqual(change["fused_runs"], 11)
        torch.testing.assert_close(result, expected, **_TOLERANCE)

    def test_a_long_loop_body_runs_as_one_segment_a_lap(self):
        x, y = _pair(704)
        _C._deferred_clear()
        with _deferring():
            before = _counts()
            _loop(x, y, _BODY50, 6).cpu()
            change = _since(before)
            kinds = _C._deferred_segment_count()
            # A bound of two laps leaves room to find the loop; a bound under two laps does not.
            found = {}
            for limit in (100, 99):
                _C._deferred_clear()
                before = _counts()
                with _capped(limit):
                    _loop(x, y, _BODY50, 6).cpu()
                found[limit] = _since(before)["loops_found"]
        self.assertEqual(change["loops_found"], 1)
        self.assertEqual(_segments(change), 6)
        self.assertEqual(kinds, 1)
        self.assertEqual(found, {100: 1, 99: 0})

    def test_ops_before_a_loop_run_as_a_part_of_their_own(self):
        x, y = _pair(960)
        with _deferring(False):
            expected = _prefixed_loop(x, y, 6).cpu()
        with _deferring():
            before = _counts()
            result = _prefixed_loop(x, y, 6).cpu()
            change = _since(before)
            for _ in range(2):
                torch.testing.assert_close(_prefixed_loop(x, y, 6).cpu(), expected, **_TOLERANCE)
        # The two ops before the loop, the two laps found after them, and four more laps.
        self.assertEqual(change["loops_found"], 1)
        self.assertEqual(_segments(change), 7)
        self.assertEqual(change["op_runs"], 2)
        torch.testing.assert_close(result, expected, **_TOLERANCE)

    def test_a_compile_cut_short_in_a_part_runs_the_parts_after_it(self):
        x, y = _pair(1024)
        other_x, other_y = _pair(1024)
        # On the host, so that no block the device hands the result holds it already.
        expected = _prefixed_loop(other_x.cpu(), other_y.cpu(), 2)

        class Cut(BaseException):
            pass

        def cut(*args, **kwargs):
            raise Cut

        with _deferring():
            # The first run sees the part before the laps once; the second compiles it.
            _prefixed_loop(x, y, 2).cpu()
            r = _prefixed_loop(other_x, other_y, 2)
            with mock.patch.object(compile_cache, "compile_logical", cut), self.assertRaises(Cut):
                torch.rbln.synchronize()
            result = r.cpu()
        torch.testing.assert_close(result, expected, **_TOLERANCE)

    def test_an_op_that_cannot_read_the_bound_is_not_deferred(self):
        x = torch.zeros(8, 256, dtype=torch.float16).to("rbln")
        with _deferring(False):
            # The op then writes `x` in place, through the warm cache.
            x.add_(1)
            x.sub_(1)
        with _deferring(), _capped(-1), self.assertRaisesRegex(Exception, "TORCH_RBLN_DEFER_OPS_MAX"):
            x.add_(1)
        self.assertTrue(bool((x.cpu() == 0).all()))

    def test_an_out_tensor_one_lap_writes_is_read_by_the_next(self):
        x, y = _pair(1088)

        def run():
            buf = x.clone()
            for _ in range(6):
                t = buf * 0.5
                torch.add(t, y, out=buf)
            return buf.cpu()

        with _deferring(False):
            expected = run()
        with _deferring():
            before = _counts()
            # Op by op, compiled, and from the cache.
            for _ in range(3):
                torch.testing.assert_close(run(), expected, **_TOLERANCE)
        self.assertEqual(_since(before)["loops_found"], 1)

    def test_both_results_of_an_op_one_lap_makes_are_read_by_the_next(self):
        x, y = _pair(1152)
        op = compile_cache.compiled_op(_TwoResults(), (x, y), {}, x.device)

        def run():
            # Two ops before the loop, so that each lap ends with the op of two results. Every lap's
            # results are kept, so that a later lap reads them where they lie.
            p, q = x * 2, y * 3
            kept = []
            for _ in range(7):
                r = q * 0.25
                s = p + y
                p, q = op.run(op.tensors((r, s), {}))
                kept += [p, q]
            return [t.cpu() for t in kept]

        with _deferring(False):
            expected = run()
        with _deferring():
            before = _counts()
            for _ in range(3):
                for got, want in zip(run(), expected):
                    torch.testing.assert_close(got, want, **_TOLERANCE)
        # The ops before the loop depart from it, so each run finds it anew.
        self.assertEqual(_since(before)["loops_found"], 3)

    def test_an_input_each_lap_updates_runs_though_nothing_holds_the_laps_results(self):
        cache = torch.zeros(1, 8, 512, 64, dtype=torch.int16).to("rbln")
        update = self._cache_update(cache)

        def run():
            cache.zero_()
            for _ in range(6):
                update(cache) * 0.5
            return cache.cpu()

        with _deferring(False):
            expected = run()
        with _deferring():
            before = _counts()
            for _ in range(3):
                self.assertEqual(run(), expected)
        change = _since(before)
        self.assertEqual(change["loops_found"], 1)
        self.assertEqual(_segments(change), 18)

    def test_a_loop_that_changes_is_found_anew(self):
        x, y = _pair(832)
        with _deferring(False):
            expected = _loop(_loop(x, y, _BODY10, 4), y, _OTHER_BODY10, 4).cpu()
        with _deferring():
            before = _counts()
            r = _loop(_loop(x, y, _BODY10, 4), y, _OTHER_BODY10, 4)
            result = r.cpu()
        change = _since(before)
        self.assertEqual(change["loops_found"], 2)
        self.assertEqual(change["laps_queued"], 8)
        self.assertEqual(_segments(change), 8)
        torch.testing.assert_close(result, expected, **_TOLERANCE)

    def test_a_lap_longer_than_128_ops_is_not_found(self):
        x, y = _pair(896)
        with _deferring():
            before = _counts()
            r = _loop(x, y, _BODY130, 3)
            self.assertEqual(_segments(_since(before)), 0)
            result = r.cpu()
        change = _since(before)
        self.assertEqual(change["loops_found"], 0)
        self.assertEqual(_segments(change), 1)
        torch.testing.assert_close(result, _loop(x.cpu(), y.cpu(), _BODY130, 3), **_TOLERANCE)

    def test_in_place_ops(self):
        def run(x, y):
            z = x.clone()
            for _ in range(4):
                z.mul_(0.5).add_(y)
            return z.cpu()

        with _deferring(False):
            expected = run(self.xd, self.yd)
        before = _counts()
        with _deferring():
            torch.testing.assert_close(run(self.xd, self.yd), expected, **_TOLERANCE)
        self.assertGreater(_since(before)["deferred_ops"], 0)

    def test_an_input_written_after_an_op_read_it(self):
        with _deferring():
            a = self.xd.clone()
            b = a * 2
            a.add_(1)
            c = b + a
            results = (b.cpu(), a.cpu(), c.cpu())
        expected = (self.x * 2, self.x + 1, self.x * 2 + (self.x + 1))
        for got, want in zip(results, expected):
            torch.testing.assert_close(got, want, **_TOLERANCE)

    def test_a_result_held_only_through_a_view(self):
        with _deferring():
            r = self.xd * 2
            v = r.view(-1)
            del r
            s = v * 3
            dropped = (self.xd + 1) * 4
            results = (v.cpu(), s.cpu(), dropped.cpu())
        expected = ((self.x * 2).view(-1), (self.x * 2).view(-1) * 3, (self.x + 1) * 4)
        for got, want in zip(results, expected):
            torch.testing.assert_close(got, want, **_TOLERANCE)

    def test_an_op_over_a_view_of_a_deferred_result(self):
        with _deferring():
            r = self.xd * 2
            row = r[1] * 3
            columns = r[:, :128] + 1
            t = r.t().contiguous() - 1
            results = (row.cpu(), columns.cpu(), t.cpu(), r.cpu())
        r = self.x * 2
        expected = (r[1] * 3, r[:, :128] + 1, r.t().contiguous() - 1, r)
        for got, want in zip(results, expected):
            torch.testing.assert_close(got, want, **_TOLERANCE)

    def test_copies_run_the_deferred_ops_first(self):
        with _deferring():
            a = self.xd.clone()
            b = a * 2
            a.copy_(self.y)
            c = torch.empty_like(b)
            c.copy_(b)
            d = a * 2
            item = d[0, 0].item()
            results = (b.cpu(), c.cpu(), d.cpu())
        expected = (self.x * 2, self.x * 2, self.y * 2)
        for got, want in zip(results, expected):
            torch.testing.assert_close(got, want, **_TOLERANCE)
        self.assertAlmostEqual(item, float(self.y[0, 0] * 2), delta=2e-2)

    def test_a_cpu_fallback_between_deferred_ops(self):
        with _deferring():
            r = self.xd * 2
            top = torch.argmax(r, dim=-1)
            s = r.cumsum(dim=-1) * 0.5
            results = (top.cpu(), s.cpu())
        r = self.x * 2
        self.assertEqual(results[0], torch.argmax(r, dim=-1))
        torch.testing.assert_close(results[1], r.float().cumsum(dim=-1).half() * 0.5, atol=0.25, rtol=2e-2)

    def test_a_compiled_graph_between_deferred_ops(self):
        graph = torch.compile(lambda t: t * 3 + 1, backend="rbln")
        with _deferring(False):
            expected = graph(self.xd * 2).cpu()
        with _deferring():
            r = self.xd * 2
            g = graph(r)
            h = g + r
            results = (g.cpu(), h.cpu())
        torch.testing.assert_close(results[0], expected, **_TOLERANCE)
        torch.testing.assert_close(results[1], expected + self.x * 2, **_TOLERANCE)

    def test_synchronize_runs_the_deferred_ops(self):
        with _deferring():
            before = _counts()
            r = self.xd * 2
            s = r + 1
            self.assertEqual(_segments(_since(before)), 0)
            torch.rbln.synchronize()
            self.assertEqual(_segments(_since(before)), 1)
            torch.testing.assert_close(s.cpu(), self.x * 2 + 1, **_TOLERANCE)

    def test_streams(self):
        side = torch.rbln.Stream()
        with _deferring():
            r = self.xd * 2
            with torch.rbln.stream(side):
                s = r + 1
                t = s * 3
            side.synchronize()
            event = torch.rbln.Event()
            u = t - 1
            event.record()
            event.synchronize()
            results = (s.cpu(), t.cpu(), u.cpu())
        r = self.x * 2
        expected = (r + 1, (r + 1) * 3, (r + 1) * 3 - 1)
        for got, want in zip(results, expected):
            torch.testing.assert_close(got, want, **_TOLERANCE)

    def _bucketed(self, rows):
        # Results of bucketed ops, whose storage holds the rows of the bucket of six.
        with _bucketing():
            return self.xd[:rows].contiguous() * 1, self.yd[:rows].contiguous() * 1

    def test_a_segment_of_one_bucket_compiles_once_for_the_bucket(self):
        inputs = {rows: self._bucketed(rows) for rows in (5, 6)}
        with _deferring(False):
            expected = {rows: _chain(*inputs[rows], 6).cpu() for rows in inputs}
        before = _counts()
        _C._rt_timing_enable(True)
        self.addCleanup(_C._rt_timing_enable, False)
        _C._rt_timing_reset()
        with _deferring(), _bucketing():
            # Op by op the first time, compiled the second, and the same function for six rows.
            for rows in (5, 5, 6):
                result = _chain(*inputs[rows], 6)
                self.assertEqual(result.untyped_storage().nbytes(), 6 * result[0].nbytes)
                torch.testing.assert_close(result.cpu(), expected[rows], **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 18)
        self.assertEqual(change["compiled"], 1)
        self.assertEqual(change["fused_runs"], 2)
        # The runs write the results where they lie, through no device copy.
        self.assertEqual(_C._rt_timing_get()[0][1], 0)

    def test_a_bucketed_loop_compiles_its_lap_once_for_the_bucket(self):
        inputs = {rows: self._bucketed(rows) for rows in (5, 6)}
        with _deferring(False):
            expected = {rows: _loop(*inputs[rows], _BODY10, 6).cpu() for rows in inputs}
        before = _counts()
        with _deferring(), _bucketing():
            for rows in (5, 6):
                torch.testing.assert_close(_loop(*inputs[rows], _BODY10, 6).cpu(), expected[rows], **_TOLERANCE)
        change = _since(before)
        # The loop of six rows follows the laps found in the loop of five, through the same function.
        self.assertEqual(change["loops_found"], 1)
        self.assertEqual(change["compiled"], 1)
        self.assertEqual(change["fused_runs"], 11)

    def test_a_segment_mixing_shapes_compiles_for_its_rows(self):
        w = self.yd[0].contiguous()

        def run(x):
            # `w * 2` does not compute by rows, so the segment compiles for the rows it is handed.
            return (x * 0.5 + w * 2).cpu()

        inputs = {rows: self._bucketed(rows)[0] for rows in (5, 6)}
        with _deferring(False):
            expected = {rows: run(inputs[rows]) for rows in inputs}
        before = _counts()
        with _deferring(), _bucketing():
            for rows in (5, 5, 6, 6):
                torch.testing.assert_close(run(inputs[rows]), expected[rows], **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 12)
        self.assertEqual(change["compiled"], 2)
        self.assertEqual(change["fused_runs"], 2)

    def test_a_bucketed_op_over_an_input_without_room_runs_as_it_is_called(self):
        # Five rows copied from the host take an allocation with no room for the bucket's six.
        x = self.x[:5].to("rbln")
        with _deferring(), _bucketing():
            before = _counts()
            s = torch.sigmoid(x)
            self.assertEqual(_since(before)["deferred_ops"], 0)
            t = torch.sigmoid(s)
            self.assertEqual(_since(before)["deferred_ops"], 1)
            result = t.cpu()
        torch.testing.assert_close(result, torch.sigmoid(torch.sigmoid(self.x[:5])), **_TOLERANCE)

    def test_launch_blocking_runs_each_op_as_it_is_called(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "1", "RBLN_LAUNCH_BLOCKING": "1"}):
            run_in_isolated_process(_defers_nothing_worker)

    @parametrize("change", ["set_", "data", "t_"])
    def test_a_deferred_op_reads_its_input_as_it_was_when_called(self, change):
        x = torch.randn(64, 64, dtype=torch.float16)
        a, other = x.to("rbln"), torch.randn(64, 64, dtype=torch.float16).to("rbln")
        with _deferring(False):
            expected = (a * 2).cpu()
        with _deferring():
            before = _counts()
            b = a * 2
            self.assertEqual(_since(before)["deferred_ops"], 1)
            # Each queues nothing on the device, so the deferred op has not run yet.
            if change == "set_":
                a.set_(other)
            elif change == "data":
                a.data = other
            else:
                a.t_()
            result = b.cpu()
        torch.testing.assert_close(result, expected)

    def test_a_deferred_op_writes_its_input_where_it_lay_when_called(self):
        with _deferring(False):
            self.xd.clone().mul_(3).cpu()
        a = self.xd.clone()
        old, replacement = a.detach(), self.yd.clone()
        with _deferring():
            before = _counts()
            a.mul_(3)
            self.assertEqual(_since(before)["deferred_ops"], 1)
            a.set_(replacement)
            results = (old.cpu(), replacement.cpu())
        torch.testing.assert_close(results[0], self.x * 3, **_TOLERANCE)
        self.assertEqual(results[1], self.y)

    def test_another_thread_waits_for_a_segment_being_queued(self):
        x, y = _pair(64)
        with _deferring(False):
            expected = _chain(x, y, 6).cpu()
        _C._deferred_clear()
        compile_segment = compile_cache.compile_segment
        compiling = threading.Event()

        def slow_compile(*args, **kwargs):
            compiling.set()
            # Time for the other thread to ask for the result while the segment is not queued yet.
            time.sleep(0.5)
            return compile_segment(*args, **kwargs)

        seen = {}

        def read():
            compiling.wait(30)
            seen["result"] = r.cpu()

        with _deferring():
            # The first run leaves other values where the second's result is allocated.
            _chain(y, x, 6).cpu()
            r = _chain(x, y, 6)
            reader = threading.Thread(target=read)
            with mock.patch.object(compile_cache, "compile_segment", slow_compile):
                reader.start()
                # This thread takes the segment and compiles it, as its second run.
                torch.rbln.synchronize()
                reader.join()
        self.assertTrue(compiling.is_set())
        torch.testing.assert_close(seen["result"], expected, **_TOLERANCE)

    def test_a_segment_the_compiler_fails_on_runs_op_by_op(self):
        x, y = _pair(320)
        with _deferring(False):
            expected = _chain(x, y, 6).cpu()

        def fail(*args, **kwargs):
            raise TypeError("a compiler bug, for the test")

        side = torch.rbln.Stream()
        with _deferring():
            _chain(y, x, 6).cpu()
            before = _counts()
            with mock.patch.object(compile_cache, "compile_logical", fail):
                for _ in range(2):
                    with torch.rbln.stream(side):
                        r = _chain(x, y, 6)
                    side.synchronize()
                    torch.testing.assert_close(r.cpu(), expected, **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["refused"], 1)
        self.assertEqual(change["op_runs"], 2)

    def test_a_compile_cut_short_still_runs_the_ops(self):
        x, y = _pair(384)
        with _deferring(False):
            expected = _chain(x, y, 6).cpu()

        class Cut(BaseException):
            pass

        def cut(*args, **kwargs):
            raise Cut

        with _deferring():
            _chain(y, x, 6).cpu()
            r = _chain(x, y, 6)
            with mock.patch.object(compile_cache, "compile_logical", cut), self.assertRaises(Cut):
                torch.rbln.synchronize()
            torch.testing.assert_close(r.cpu(), expected, **_TOLERANCE)

    def _failing_run(self, x, y):
        # A segment's second run compiles it; this function takes one tensor where it gets two.
        with _deferring(False):
            _chain(x, y, 6).cpu()
        _chain(x, y, 6).cpu()
        mismatched = compile_cache.compiled_op(torch.nn.Identity(), (x,), {}, x.device).function
        return mock.patch.object(compile_cache, "compile_segment", lambda *args, **kwargs: mismatched)

    def test_an_error_running_a_segment_leaves_the_stream_set(self):
        x, y = _pair(448)
        side = torch.rbln.Stream()
        kept = []
        with _deferring():
            with self._failing_run(x, y), self.assertRaisesRegex(RuntimeError, "takes 1 tensors"):
                with torch.rbln.stream(side):
                    kept.append(_chain(x, y, 6))
            _C._deferred_clear()
            self.assertEqual(torch.rbln.current_stream(), torch.rbln.default_stream())
            torch.testing.assert_close(_chain(x, y, 2).cpu(), _chain(x.cpu(), y.cpu(), 2), **_TOLERANCE)

    def test_an_error_running_a_segment_ends_the_profiler_session(self):
        x, y = _pair(512)
        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        kept = []
        with _deferring(), mock.patch.dict(os.environ, {"RBLN_PROFILER": "1"}):
            with self._failing_run(x, y), self.assertRaisesRegex(RuntimeError, "takes 1 tensors"):
                with profile(activities=activities):
                    kept.append(_chain(x, y, 6))
            _C._deferred_clear()
            with profile(activities=activities) as session:
                (x * 2).cpu()
        self.assertTrue(any(e.device_type == torch.autograd.DeviceType.PrivateUse1 for e in session.events()))

    def test_a_segment_whose_results_nothing_holds_does_not_run(self):
        with _deferring(False):
            torch.mul(self.xd, 3).cpu()
        with _deferring():
            before = _counts()
            torch.mul(self.xd, 3)
            torch.rbln.synchronize()
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 1)
        self.assertEqual(_segments(change), 0)

    def test_an_out_tensor_is_written_whatever_holds_its_bytes(self):
        buf = torch.zeros(8, 256, dtype=torch.float16).to("rbln")
        with _deferring(False):
            torch.mul(self.xd, 2, out=torch.empty_like(buf))
        with _deferring():
            # A tensor of a storage of its own over the bytes of `buf`, which nothing holds once
            # the op is deferred.
            alias = torch.rbln._create_tensor_from_ptr(buf.data_ptr(), buf.shape, buf.dtype)
            before = _counts()
            torch.mul(self.xd, 2, out=alias)
            self.assertEqual(_since(before)["deferred_ops"], 1)
            del alias
            result = buf.cpu()
        torch.testing.assert_close(result, self.x * 2, **_TOLERANCE)

    def test_an_op_reads_what_another_storage_over_its_bytes_wrote(self):
        buf = torch.zeros(8, 256, dtype=torch.float16).to("rbln")
        # A tensor of a storage of its own over the bytes of `buf`.
        alias = torch.rbln._create_tensor_from_ptr(buf.data_ptr(), buf.shape, buf.dtype)
        with _deferring():
            # Three runs, so that a segment of both ops would run op by op, then compiled, then from the cache.
            for _ in range(3):
                buf.zero_()
                torch.mul(self.xd, 3, out=alias)
                result = (buf + 1).cpu()
                torch.testing.assert_close(result, self.x * 3 + 1, **_TOLERANCE)

    def _cache_update(self, cache):
        # An op that writes `cache` in place, which binds as torch holds it for an integer cache.
        others = (
            torch.ones(1, 8, 4, 64, dtype=torch.int16).to("rbln"),
            torch.tensor([[3]], dtype=torch.int16).to("rbln"),
            self.xd,
        )
        op = compile_cache.compiled_op(_CacheUpdate(), (cache, *others), {}, self.xd.device)
        return lambda cache: op.run(op.tensors((cache, *others), {}))

    def test_an_op_updating_an_input_runs_though_nothing_holds_its_results(self):
        cache = torch.zeros(1, 8, 512, 64, dtype=torch.int16).to("rbln")
        update = self._cache_update(cache)
        with _deferring():
            before = _counts()
            update(cache)
            self.assertEqual(_since(before)["deferred_ops"], 1)
            torch.rbln.synchronize()
        self.assertEqual(_segments(_since(before)), 1)

    def test_an_op_updating_an_input_orders_with_other_views_of_it(self):
        base = torch.zeros(9, 512, 64, dtype=torch.int16).to("rbln")
        first, second = base[:8].view(1, 8, 512, 64), base[1:].view(1, 8, 512, 64)
        update = self._cache_update(first)
        with _deferring():
            before = _counts()
            update(first)
            # Another view of bytes the segment writes: the segment runs before this run is deferred.
            update(second)
            self.assertEqual(_since(before)["deferred_ops"], 2)
            self.assertEqual(_segments(_since(before)), 1)
            torch.rbln.synchronize()
        self.assertEqual(_segments(_since(before)), 2)

    def test_a_fused_run_keeps_its_ops_in_the_compile_cache(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_COMPILED_OPS": "10", "TORCH_RBLN_DEPLOY": "ON"}):
            run_in_isolated_process(_fused_ops_stay_compiled_worker)

    # It fills the device's memory, which no test running beside it may need.
    @pytest.mark.single_worker
    def test_a_run_that_fails_to_allocate_leaves_the_segment_as_it_was(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "1", "TORCH_RBLN_DEPLOY": "ON"}):
            run_in_isolated_process(_failed_allocation_worker)

    def test_empty_cache_runs_the_deferred_ops_first(self):
        big = torch.ones(64, 1024, 1024, dtype=torch.float16, device="rbln")
        with _deferring(False):
            (big[0] * 2).cpu()
        torch.rbln.empty_cache()
        reserved = torch.rbln.memory_reserved()
        with _deferring():
            before = _counts()
            r = big[0] * 2
            self.assertEqual(_since(before)["deferred_ops"], 1)
            del big
            torch.rbln.empty_cache()
            self.assertLess(torch.rbln.memory_reserved(), reserved - 100 * 2**20)
            self.assertTrue(bool((r.cpu() == 2).all()))


instantiate_parametrized_tests(TestDeferredOps)


class _CacheUpdate(torch.nn.Module):
    def forward(self, cache, state, position, x):
        torch.ops.rbln_custom_ops.rbln_cache_update(cache, state, position, torch.tensor(2))
        return x * 2


class _TwoResults(torch.nn.Module):
    def forward(self, a, b):
        return a * 2, b * 3


def _fused_ops_stay_compiled_worker():
    # Each round runs the hot chain, deferred, then six ops of shapes of their own as they are
    # called, more than the ten compiled ops the process keeps besides the chain's four.
    x, y = (torch.randn(8, 448, dtype=torch.float16).to("rbln") for _ in range(2))
    others = [torch.randn(8, 64 * (j + 1), dtype=torch.float16).to("rbln") for j in range(36)]
    fused = []
    for round_ in range(6):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "1"}):
            _chain(x, y, 6).cpu()
        for other in others[round_ * 6 : round_ * 6 + 6]:
            (other * 2).cpu()
        fused.append(dict(_C._deferred_stats())["fused_runs"])
    # Op by op the first time, then the compiled function every time.
    assert fused == [0, 1, 2, 3, 4, 5], fused


def _failed_allocation_worker():
    a = torch.randn(8, 256, dtype=torch.float16).to("rbln")
    b = torch.randn(8192, 4096, dtype=torch.float16).to("rbln")
    op = compile_cache.compiled_op(_TwoResults(), (a, b), {}, a.device)
    with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "0"}):
        (a + 1).cpu()
    out = torch.zeros_like(a)
    fill = []
    for chunk in (2**30, 2**25):
        try:
            while True:
                fill.append(torch.empty(chunk, dtype=torch.uint8, device="rbln"))
        except RuntimeError:
            pass
    try:
        # The run gets `out` for its first result, and fails to allocate its second.
        op.run(op.tensors((a, b), {}), [out])
        raise AssertionError("the run allocated a result with no device memory left")
    except RuntimeError as e:
        assert "out of memory" in str(e), e
    del fill
    r = a + 1
    torch.testing.assert_close(r.cpu(), (a.cpu() + 1), **_TOLERANCE)
    assert bool((out.cpu() == 0).all()), "the failed run left a write of another op's result to its out"


def _defers_nothing_worker():
    x = torch.randn(8, 256, dtype=torch.float16)
    before = dict(_C._deferred_stats())
    torch.testing.assert_close((x.to("rbln") * 2 + 1).cpu(), x * 2 + 1, **_TOLERANCE)
    assert dict(_C._deferred_stats())["deferred_ops"] == before["deferred_ops"]


if __name__ == "__main__":
    run_tests()
