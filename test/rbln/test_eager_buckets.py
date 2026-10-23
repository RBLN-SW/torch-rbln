# Owner(s): ["module: PrivateUse1"]

"""
With TORCH_RBLN_EAGER_BUCKETS on, an eager elementwise op whose result has two
or more rows along its leading axis, of a multiple of 64 elements, runs a
function compiled for the bucket of that extent (1, 2, 3, 4, 6, 8, 12, 16, 24,
...) over its rows. It compiles once
per bucket rather than once per extent, gives exactly the result it gives
unbucketed, and its result keeps the storage of the bucket's rows, so that the
next such op reads it in place. Any other input reads the rows past its own in
place when its allocation's block holds them, and through a padded copy when
it does not.
"""

import os
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

import torch_rbln._C as _C
from torch_rbln._internal import compile_cache, warm_cache


# A row of 512 bytes, the caching allocator's granularity, so that an allocation
# of the call's rows ends where its block does.
H = 256

# A call's rows, and the rows of its bucket.
ROWS = ((2, 2), (5, 6), (13, 16))

# Each takes x and y of the call's rows, w of one row's shape, and r of one row.
OPS = {
    "add_broadcast_row": lambda x, y, w, r: x + r,
    "mul_broadcast_trailing": lambda x, y, w, r: x * w,
    "sub_alpha": lambda x, y, w, r: torch.sub(x, y, alpha=2),
    "div": lambda x, y, w, r: x / y,
    "mul_python_scalar": lambda x, y, w, r: x * 3.0,
    "maximum": lambda x, y, w, r: torch.maximum(x, y),
    "sigmoid": lambda x, y, w, r: torch.sigmoid(x),
    "neg": lambda x, y, w, r: -x,
    "abs": lambda x, y, w, r: x.abs(),
    "silu": lambda x, y, w, r: torch.nn.functional.silu(x),
    "clamp": lambda x, y, w, r: x.clamp(-0.5, 0.5),
    "pow": lambda x, y, w, r: x**2,
    "gt_scalar": lambda x, y, w, r: x > 0.25,
    "le": lambda x, y, w, r: x <= y,
}


def _buckets(on: bool):
    return mock.patch.dict(os.environ, {"TORCH_RBLN_EAGER_BUCKETS": "1" if on else "0"})


def _storage_rows(t: torch.Tensor) -> int:
    return t.untyped_storage().nbytes() // t[0].nbytes


def _device_copies() -> int:
    """memcpy_v2v calls since the last reset, padded copies included."""
    return _C._rt_timing_get()[0][1]


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.test_set_ci
class TestEagerBuckets(TestCase):
    def setUp(self):
        compile_cache.clear_rbln_compile_cache()
        warm_cache.clear()

    def _inputs(self, rows, device):
        x = torch.randn(rows, H, dtype=torch.float16)
        y = torch.rand(rows, H, dtype=torch.float16) + 0.5
        w = torch.randn(H, dtype=torch.float16)
        r = torch.randn(1, H, dtype=torch.float16)
        return tuple(t.to(device) for t in (x, y, w, r))

    @parametrize("op", list(OPS))
    def test_bucketed_result_equals_unbucketed(self, device, op):
        fn = OPS[op]
        for rows, extent in ROWS:
            inputs = self._inputs(rows, device)
            with _buckets(False):
                expected = fn(*inputs)
            with _buckets(True):
                result = fn(*inputs)
            self.assertEqual(_storage_rows(expected), rows)
            self.assertEqual(_storage_rows(result), extent)
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.cpu(), expected.cpu(), atol=0, rtol=0)

    def test_in_place_op_takes_a_copy_of_the_bucketed_result(self, device):
        x, y, _, _ = self._inputs(5, device)
        expected = x.clone()
        with _buckets(False):
            expected.add_(y)
            expected.add_(y)
        data = x.data_ptr()
        with _buckets(True):
            x.add_(y)
            x.add_(y)
            compiled = len(compile_cache._compiled_op_cache)
            other, more, _, _ = self._inputs(6, device)
            other.add_(more)
        self.assertEqual(len(compile_cache._compiled_op_cache), compiled)
        self.assertEqual(x.data_ptr(), data)
        self.assertEqual(_storage_rows(x), 5)
        self.assertEqual(x.cpu(), expected.cpu(), atol=0, rtol=0)

    def test_compiles_once_per_bucket(self, device):
        # 1 runs the function of its own extent; 2 to 17 run those of 2, 3, 4, 6, 8, 12, 16 and 24.
        with _buckets(True):
            for rows in range(1, 18):
                torch.sigmoid(torch.randn(rows, H, dtype=torch.float16).to(device))
        self.assertEqual(len(compile_cache._compiled_op_cache), 9)

    def test_compiles_once_per_extent_with_buckets_off(self, device):
        with _buckets(False):
            for rows in range(1, 18):
                result = torch.sigmoid(torch.randn(rows, H, dtype=torch.float16).to(device))
                self.assertEqual(_storage_rows(result), rows)
        self.assertEqual(len(compile_cache._compiled_op_cache), 17)

    def test_warm_cache_serves_every_extent_of_a_bucket(self, device):
        with _buckets(True):
            torch.sigmoid(torch.randn(5, H, dtype=torch.float16).to(device))
            hits = _C._dispatch_shim_diag_dump()[2]
            for rows in (6, 5, 6):
                torch.sigmoid(torch.randn(rows, H, dtype=torch.float16).to(device))
        self.assertEqual(_C._dispatch_shim_diag_dump()[2], hits + 3)
        self.assertEqual(len(compile_cache._compiled_op_cache), 1)

    def test_an_input_without_room_binds_through_a_padded_copy(self, device):
        x = torch.randn(5, H, dtype=torch.float16).to(device)
        _C._rt_timing_enable(True)
        try:
            with _buckets(True):
                _C._rt_timing_reset()
                y = torch.sigmoid(x)
                self.assertEqual(_device_copies(), 1)
                torch.sigmoid(y)
                self.assertEqual(_device_copies(), 1)
        finally:
            _C._rt_timing_enable(False)

    def test_an_input_binds_over_the_rest_of_its_block(self, device):
        # 5 rows of 256 KiB take a block of 2 MiB, which holds the bucket's 6.
        x = torch.randn(5, 128 * 1024, dtype=torch.float16).to(device)
        with _buckets(False):
            expected = torch.sigmoid(x)
        _C._rt_timing_enable(True)
        try:
            with _buckets(True):
                _C._rt_timing_reset()
                result = torch.sigmoid(x)
                self.assertEqual(_device_copies(), 0)
        finally:
            _C._rt_timing_enable(False)
        self.assertEqual(_storage_rows(result), 6)
        self.assertEqual(result.cpu(), expected.cpu(), atol=0, rtol=0)

    @parametrize("shape", [(5, 100), (5,)])
    def test_rows_of_another_count_than_a_multiple_of_64_elements_keep_their_extent(self, device, shape):
        x = torch.randn(shape, dtype=torch.float16).to(device)
        y = torch.randn(shape, dtype=torch.float16).to(device)
        with _buckets(False):
            expected = x + y
        with _buckets(True):
            result = x + y
        self.assertEqual(_storage_rows(result), 5)
        self.assertEqual(result.cpu(), expected.cpu(), atol=0, rtol=0)

    def test_a_reduction_compiles_per_extent(self, device):
        with _buckets(True):
            for rows in (2, 3, 5):
                torch.randn(rows, H, dtype=torch.float16).to(device).mean(-1)
        self.assertEqual(len(compile_cache._compiled_op_cache), 3)

    def test_a_callers_out_is_written_in_place(self, device):
        x, y, _, _ = self._inputs(5, device)
        out = torch.empty(5, H, dtype=torch.float16, device=device)
        data = out.data_ptr()
        with _buckets(True):
            torch.add(x, y, out=out)
        self.assertEqual(out.data_ptr(), data)
        self.assertEqual(_storage_rows(out), 5)
        self.assertEqual(out.cpu(), (x + y).cpu(), atol=0, rtol=0)

    def test_a_view_runs_the_function_of_its_own_extent(self, device):
        x = torch.randn(H, 5, dtype=torch.float16).to(device)
        with _buckets(True):
            result = torch.sigmoid(x.t())
        self.assertEqual(_storage_rows(result), 5)
        self.assertEqual(result.stride(), torch.sigmoid(x.cpu().t()).stride())

    def test_a_function_that_cannot_compute_by_rows_takes_the_calls_extent(self, device):
        # The function's result is its input, which a call of fewer rows cannot pass.
        x = torch.randn(5, H, dtype=torch.float16).to(device)
        op = compile_cache.compiled_op(_Identity(), (x,), {}, x.device, (6, [0]))
        self.assertFalse(op.function.row_wise)
        self.assertEqual(op.run([x]).cpu(), x.cpu(), atol=0, rtol=0)
        self.assertEqual(len(compile_cache._compiled_op_cache), 2)

    @parametrize("bucket", ["refused", "not_row_wise"])
    def test_a_bucket_without_a_function_by_rows_runs_each_call_on_the_device(self, device, bucket):
        compile_op = compile_cache._compile
        bucketed = []

        def compile(module, args, kwargs, device_index, rows):
            if rows:
                bucketed.append(rows)
                if bucket == "refused":
                    raise ValueError("the compiler refuses the bucket")
                rows = ()
            return compile_op(module, args, kwargs, device_index, rows)

        xs = [torch.randn(rows, H, dtype=torch.float16).to(device) for rows in (5, 5, 6)]
        with _buckets(False):
            expected = [torch.sigmoid(x).cpu() for x in xs]
        compile_cache.clear_rbln_compile_cache()
        warm_cache.clear()
        no_fallback = mock.patch.dict(os.environ, {"TORCH_RBLN_DISABLE_FALLBACK": "compile_error"})
        with mock.patch.object(compile_cache, "_compile", compile), _buckets(True), no_fallback:
            hits = _C._dispatch_shim_diag_dump()[2]
            results = [torch.sigmoid(x) for x in xs]
            hits = _C._dispatch_shim_diag_dump()[2] - hits
        self.assertEqual(bucketed, [(0,)])
        self.assertEqual(hits, 1)
        self.assertEqual(len(compile_cache._compiled_op_cache), 3)
        for x, result, values in zip(xs, results, expected):
            self.assertEqual(_storage_rows(result), x.size(0))
            self.assertEqual(result.cpu(), values, atol=0, rtol=0)

    def test_a_functional_call_checks_its_arguments_as_torch_does(self, device):
        x = torch.randn(3, H, dtype=torch.float16).to(device)
        y = torch.randn(4, H, dtype=torch.float16).to(device)
        with _buckets(True), self.assertRaisesRegex(RuntimeError, "must match the size of tensor b"):
            torch.add(x, y)


instantiate_device_type_tests(TestEagerBuckets, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
