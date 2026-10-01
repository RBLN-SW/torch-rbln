# Owner(s): ["module: PrivateUse1"]
"""fill_ on a view of a device tensor writes only that region, without reading it back.

Zeros are filled on the device; a dense view gets any other value from one host pattern; a
strided view is written run by run from one host pattern through the h2v batch copy. None of
these reads the storage back, which the boundary timers of the host copies show.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln.profiler import _read_rt_timing, _RT_PRIMS, _rt_timing_enable, _rt_timing_reset


DEVICE = torch.device("rbln:0")
_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")


def _device_reads(fn):
    """Device-to-host copies fn() issued, counted by the host copy boundary timers."""
    torch.rbln.synchronize()
    _rt_timing_reset()
    _rt_timing_enable(True)
    try:
        fn()
        torch.rbln.synchronize()
    finally:
        _rt_timing_enable(False)
    calls = {name: cnt for name, (_ns, cnt) in zip(_RT_PRIMS, _read_rt_timing())}
    return calls["v2h"] + calls["v2h_multi"]


def _on_device(shape, dtype=torch.bfloat16):
    if dtype.is_floating_point:
        return torch.randn(shape).to(dtype).to(DEVICE)
    return torch.randint(-50, 50, shape, dtype=dtype).to(DEVICE)


@pytest.mark.test_set_ci
@needs_device
class TestFillRegionDevice(TestCase):
    def _check(self, make_view, value, shape=(1, 474, 3072), dtype=torch.bfloat16, reads=0):
        x = _on_device(shape, dtype)
        ref = x.cpu()
        make_view(ref).fill_(value)
        self.assertEqual(_device_reads(lambda: make_view(x).fill_(value)), reads, f"{dtype}: fill_ read back")
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_tail_slice_single_run(self):
        # rows [38:) of a 3-D buffer: one contiguous run
        self._check(lambda t: t[:, 38:, :], 0.0)

    def test_strided_column_region(self):
        # many runs: a column slice through a 2-D tensor
        self._check(lambda t: t[:, 1000:1500], 1.5, shape=(512, 3072))

    def test_full_contiguous_tensor(self):
        self._check(lambda t: t, -2.0, shape=(64, 4096))

    def test_dense_permuted_tensor(self):
        # a transposed tensor still covers one byte range, written as a whole
        self._check(lambda t: t.t(), 3.0, shape=(256, 512))

    def test_dtypes(self):
        for dtype, value in (
            (torch.int32, -7),
            (torch.int16, 3),
            (torch.bfloat16, 0.5),
            (torch.float32, 3.25),
            (torch.float16, 0.5),
        ):
            with self.subTest(dtype=dtype):
                self._check(lambda t: t[:, 8:], value, shape=(16, 1024), dtype=dtype)

    def test_negative_zero_is_not_a_zero_fill(self):
        # -0.0 has its sign bit set, so it must not take the all-zero-bytes fill
        x = _on_device((64,), torch.float32)
        x.fill_(-0.0)
        self.assertTrue(torch.all(torch.signbit(x.cpu())))

    def test_fresh_empty_storage(self):
        x = torch.empty(32, 1024, dtype=torch.bfloat16, device=DEVICE)

        def fill():
            x.fill_(2.5)
            x[:, :100].fill_(-1.0)

        self.assertEqual(_device_reads(fill), 0)
        host = x.cpu()
        self.assertTrue(torch.all(host[:, :100] == -1.0) and torch.all(host[:, 100:] == 2.5))

    def test_region_above_bulk_caps_is_split_and_correct(self):
        # more entries and more bytes than one bulk call may carry (RBLNHostBatch.cpp caps):
        # H2VBatch must split the submit
        self._check(lambda t: t[:, :262144], 1.0, shape=(32, 524288))

    def test_dense_fill_above_the_pattern_cap(self):
        # 64 MiB of bf16: written from a 16 MiB pattern chunk by chunk
        self._check(lambda t: t, 7.0, shape=(8192, 4096))

    def test_overlapping_unfold_view_stays_correct(self):
        # unfold windows overlap with positive strides; bulk destinations must be disjoint,
        # so this view goes through a CPU tensor
        x = _on_device((64,))
        ref = x.cpu()
        ref.unfold(0, 4, 1).fill_(1.0)
        x.unfold(0, 4, 1).fill_(1.0)
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_too_many_runs_falls_back_correctly(self):
        # one-element runs above kMaxFillRuns go through a CPU tensor
        x = _on_device((8192, 64))
        ref = x.cpu()
        ref[:, 3].fill_(1.0)
        x[:, 3].fill_(1.0)
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_zero_full_allocation_fills_on_device(self):
        x = _on_device((64, 1024))
        self.assertEqual(_device_reads(x.zero_), 0)
        self.assertTrue(torch.all(x.cpu() == 0))


if __name__ == "__main__":
    run_tests()
