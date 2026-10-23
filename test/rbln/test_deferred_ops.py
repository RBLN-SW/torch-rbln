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


def _deferring(limit: int = 32):
    return mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": str(limit)})


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


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestDeferredOps(TestCase):
    def setUp(self):
        self.x = torch.randn(8, 256, dtype=torch.float16)
        self.y = torch.randn(8, 256, dtype=torch.float16)
        self.xd = self.x.to("rbln")
        self.yd = self.y.to("rbln")

    @parametrize("n", [5, 20])
    def test_a_chain_gives_what_it_gives_eagerly(self, n):
        expected = _chain(self.xd, self.yd, n).cpu()
        before = _counts()
        with _deferring():
            # Op by op the first time, compiled the second, from the cache the third.
            for _ in range(3):
                torch.testing.assert_close(_chain(self.xd, self.yd, n).cpu(), expected, **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 3 * n)
        self.assertEqual(_segments(change), 3)
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
        expected = _chain(self.xd, self.yd, 6).cpu()

        def refuse(*args, **kwargs):
            raise RuntimeError("refused for the test")

        with _deferring():
            _chain(self.xd, self.yd, 6).cpu()
            before = _counts()
            with mock.patch.object(compile_cache, "compile_logical", refuse):
                for _ in range(3):
                    torch.testing.assert_close(_chain(self.xd, self.yd, 6).cpu(), expected, **_TOLERANCE)
        change = _since(before)
        self.assertEqual(change["refused"], 1)
        self.assertEqual(change["op_runs"], 3)
        self.assertEqual(change["fused_runs"], 0)

    def test_a_segment_holds_at_most_the_limit(self):
        before = _counts()
        with _deferring(3):
            _chain(self.xd, self.yd, 7).cpu()
        change = _since(before)
        self.assertEqual(change["deferred_ops"], 7)
        self.assertEqual(_segments(change), 3)

    def test_in_place_ops(self):
        def run(x, y):
            z = x.clone()
            for _ in range(4):
                z.mul_(0.5).add_(y)
            return z.cpu()

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

    def test_launch_blocking_runs_each_op_as_it_is_called(self):
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "32", "RBLN_LAUNCH_BLOCKING": "1"}):
            run_in_isolated_process(_defers_nothing_worker)

    @parametrize("change", ["set_", "data", "t_"])
    def test_a_deferred_op_reads_its_input_as_it_was_when_called(self, change):
        x = torch.randn(64, 64, dtype=torch.float16)
        a, other = x.to("rbln"), torch.randn(64, 64, dtype=torch.float16).to("rbln")
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with _deferring(0):
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
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "32", "TORCH_RBLN_DEPLOY": "ON"}):
            run_in_isolated_process(_failed_allocation_worker)

    def test_empty_cache_runs_the_deferred_ops_first(self):
        big = torch.ones(64, 1024, 1024, dtype=torch.float16, device="rbln")
        with _deferring(0):
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
        with mock.patch.dict(os.environ, {"TORCH_RBLN_DEFER_OPS": "32"}):
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
