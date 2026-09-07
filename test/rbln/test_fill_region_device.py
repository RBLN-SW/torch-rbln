# Owner(s): ["module: PrivateUse1"]
"""fill_ on a view of a device-resident storage writes only that region, on the device.

The borrow path read the whole storage back, filled on the host, and left the storage
host-latest for the next device consumer to re-upload. The region path writes the view's
runs through the runtime's h2v copy and the storage stays device-latest.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln import _C


DEVICE = torch.device("rbln:0")
_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")
_identity = torch.compile(lambda t: torch.maximum(t, t), backend="rbln", dynamic=False)


def _host_sync():
    (dc, db), (hc, hb) = _C._rt_prof_host_sync()
    return dc, db, hc, hb


def _delta(a, b):
    return tuple(y - x for x, y in zip(a, b))


def _d2h_bytes_of(fn):
    """D2H bytes the runtime moved while running fn()."""
    before = _host_sync()
    out = fn()
    torch.rbln.synchronize()
    return _delta(before, _host_sync())[1], out


def _assert_data_on_device(tc, x, msg=""):
    """Reading x back must move every byte from the device: the data still lives there."""
    moved, _ = _d2h_bytes_of(x.cpu)
    tc.assertEqual(moved, x.numel() * x.element_size(), f"{msg}: storage is not device-latest ({moved} B moved)")


def _device_latest(shape, dtype=torch.bfloat16):
    x = (
        torch.randn(shape).to(dtype).to(DEVICE)
        if dtype.is_floating_point
        else torch.randint(-50, 50, shape, dtype=dtype).to(DEVICE)
    )
    y = _identity(x)  # a compiled graph's output lives on the device (device-latest)
    torch.rbln.synchronize()
    return y


@pytest.mark.test_set_ci
@needs_device
class TestFillRegionDevice(TestCase):
    def _check(self, make_view, value, shape=(1, 474, 3072), dtype=torch.bfloat16, resident=True):
        x = _device_latest(shape, dtype)
        ref = x.cpu()
        make_view(ref).fill_(value)
        before = _host_sync()
        make_view(x).fill_(value)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        if resident:
            self.assertEqual(d[0], 0, f"{dtype}: region fill read the storage back: {d}")
            self.assertLessEqual(
                d[3], make_view(x).numel() * x.element_size(), f"{dtype}: uploaded more than the region: {d}"
            )
            _assert_data_on_device(self, x, f"{dtype}: storage left the device")
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_tail_slice_single_run(self):
        # rows [38:) of a 3-D buffer: one contiguous run
        self._check(lambda t: t[:, 38:, :], 0.0)

    def test_strided_column_region(self):
        # many runs: a column slice through a 2-D tensor
        self._check(lambda t: t[:, 1000:1500], 1.5, shape=(512, 3072))

    def test_full_contiguous_device_latest_tensor(self):
        # a full fill of a device-latest tensor no longer round-trips either
        self._check(lambda t: t, -2.0, shape=(64, 4096))

    def test_dtypes_native_physical(self):
        # dtypes the device stores as-is: region fill stays on the device
        for dtype, value in ((torch.int32, -7), (torch.int16, 3), (torch.bfloat16, 0.5)):
            with self.subTest(dtype=dtype):
                self._check(lambda t: t[:, 8:], value, shape=(16, 1024), dtype=dtype)

    def test_dtypes_converted_physical_stay_correct(self):
        # float16/float32 are held as dlf16 on the device (user dtype != physical dtype). The
        # runtime routes a converting host write to the user view (exact bytes for host readers),
        # so these take the same whole-entry host path as before; only correctness is asserted.
        # A device-side fill for converting views needs the runtime to accept a pattern in the
        # physical dtype.
        for dtype, value in ((torch.float32, 3.25), (torch.float16, 0.5)):
            with self.subTest(dtype=dtype):
                self._check(lambda t: t[:, 8:], value, shape=(16, 1024), dtype=dtype, resident=False)

    def test_host_latest_storage_moves_nothing(self):
        # host-latest: the runtime lands the region in the user view; nothing may move, and the
        # data stays on the host (reading it back is a host copy, not a D2H)
        x = torch.randn(64, 1024).to(torch.bfloat16).to(DEVICE)
        before = _host_sync()
        x[:, 10:].fill_(4.0)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual((d[0], d[2]), (0, 0), f"host-latest fill moved data: {d}")
        moved, host = _d2h_bytes_of(x.cpu)
        self.assertEqual(moved, 0, "host-latest storage was pushed to the device by fill_")
        self.assertTrue(torch.all(host[:, 10:] == 4.0))

    def test_fresh_empty_storage(self):
        # an entry that was never materialised anywhere: fill_ is the first write
        x = torch.empty(32, 1024, dtype=torch.bfloat16, device=DEVICE)
        before = _host_sync()
        x.fill_(2.5)
        x[:, :100].fill_(-1.0)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual(d[0], 0, f"fill of a fresh storage read something back: {d}")
        host = x.cpu()
        self.assertTrue(torch.all(host[:, :100] == -1.0) and torch.all(host[:, 100:] == 2.5))

    def test_region_above_bulk_caps_is_split_and_correct(self):
        # more entries and more bytes than one bulk call may carry (RBLNHostBatch.cpp caps):
        # H2VBatch must split the submit
        x = _device_latest((32, 524288))
        ref = x.cpu()
        ref[:, :262144].fill_(1.0)
        before = _host_sync()
        x[:, :262144].fill_(1.0)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual(d[0], 0, f"large region fill read the storage back: {d}")
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_overlapping_unfold_view_stays_correct(self):
        # unfold windows overlap with positive strides; bulk destinations must be disjoint,
        # so this view takes the previous path
        x = _device_latest((64,))
        ref = x.cpu()
        ref.unfold(0, 4, 1).fill_(1.0)
        x.unfold(0, 4, 1).fill_(1.0)
        torch.rbln.synchronize()
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_too_many_runs_falls_back_correctly(self):
        # one-element runs above kMaxFillRuns: previous path
        x = _device_latest((8192, 64))
        ref = x.cpu()
        ref[:, 3].fill_(1.0)
        x[:, 3].fill_(1.0)
        torch.rbln.synchronize()
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_zero_full_allocation_still_marks_zeros(self):
        x = _device_latest((64, 1024))
        before = _host_sync()
        x.zero_()
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual((d[0], d[2]), (0, 0))
        self.assertTrue(torch.all(x.cpu() == 0))


if __name__ == "__main__":
    run_tests()
