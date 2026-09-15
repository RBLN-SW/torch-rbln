# Owner(s): ["module: PrivateUse1"]
"""In-place / fallback paths touch only the view's bytes of a device-resident storage.

Two paths used to move whole storages:
  - the boxed CPU fallback's input borrow syncs the whole vmem entry, so an op reading a
    prefix view of a device-latest tensor read back every byte of it;
  - a cpu->rbln copy_ into a non-contiguous view with a dtype/shape-mismatched cpu src
    staged the whole dst span through the host (read-modify-write).
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
    y = _identity(torch.randn(shape).to(dtype).to(DEVICE))  # compiled output: device-latest
    torch.rbln.synchronize()
    return y


@pytest.mark.test_set_ci
@needs_device
class TestFallbackRegionIO(TestCase):
    def test_boxed_fallback_reads_only_the_view(self):
        # argmax is an explicit CPU fallback; its input is a 4-row prefix of a 512-row storage
        x = _device_latest((512, 3072))
        ref = x.cpu()[:4].argmax(-1)
        before = _host_sync()
        got = x[:4].argmax(-1)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertLessEqual(d[1], 4 * 3072 * 2 + 4096, f"read back more than the 4-row view: {d}")
        _assert_data_on_device(self, x, "storage should stay device-latest")
        torch.testing.assert_close(got.cpu(), ref)

    def test_small_view_of_host_latest_storage_costs_only_the_view(self):
        # the size heuristic's worst case: the data is on the host, so the copy path is a host
        # memcpy of the view (no transfer) instead of a free borrow
        x = torch.randn(512, 3072).to(torch.bfloat16).to(DEVICE)
        ref = x.cpu()[:4].argmax(-1)
        before = _host_sync()
        got = x[:4].argmax(-1)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual((d[0], d[2]), (0, 0), f"host-latest view read moved data: {d}")
        torch.testing.assert_close(got.cpu(), ref)

    def test_full_view_keeps_borrow(self):
        # a view covering at least half the storage keeps the borrow fast path (one D2H, then synced)
        x = _device_latest((64, 1024))
        ref = x.cpu().argmax(-1)
        before = _host_sync()
        got = x.argmax(-1)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual(d[0], 1)
        torch.testing.assert_close(got.cpu(), ref)

    def test_strided_copy_from_mismatched_cpu_src_writes_only_runs(self):
        # every 4th row of a device-latest [256, 8192] bf16 tensor <- fp32 rows: the src is
        # converted on the CPU and the rows (above the strided run threshold) are written in place; the dst span is
        # neither read back nor rewritten as a whole
        x = _device_latest((256, 8192))
        ref = x.cpu()
        rows = torch.randn(64, 8192)  # fp32 -> bf16 conversion + strided dst
        ref[::4].copy_(rows)
        before = _host_sync()
        x[::4].copy_(rows)
        torch.rbln.synchronize()
        d = _delta(before, _host_sync())
        self.assertEqual(d[0], 0, f"strided dst copy read the dst back: {d}")
        self.assertLessEqual(d[3], 64 * 8192 * 2 + 65536, f"wrote more than the strided rows: {d}")
        _assert_data_on_device(self, x, "strided copy_ pulled the storage to the host")
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_strided_copy_profiler_attribution(self):
        # the src conversion counts as staging; the dst is not pulled to the host, so the
        # "non-contiguous rbln dst pulled to host" site must not move (BounceSite order:
        # 1 = cpu src staged, 2 = non-contiguous dst pulled)
        x = _device_latest((256, 8192))
        rows = torch.randn(64, 8192)
        before = _C._profiler_dump_bounces()
        x[::4].copy_(rows)
        torch.rbln.synchronize()
        after = _C._profiler_dump_bounces()
        self.assertEqual(after[1][0] - before[1][0], 1, "src conversion not attributed to staging")
        self.assertEqual(after[2][0] - before[2][0], 0, "dst reported as pulled to the host")

    def test_small_runs_still_correct(self):
        # below the strided run threshold the staged path remains; correctness only
        x = _device_latest((256, 4096))
        ref = x.cpu()
        rows = torch.randn(64, 4096)
        ref[::4].copy_(rows)
        x[::4].copy_(rows)
        torch.rbln.synchronize()
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
