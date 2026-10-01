# Owner(s): ["module: PrivateUse1"]
"""The CPU fallback and cpu->rbln copy_ move only the bytes of the view they touch.

- the boxed CPU fallback copies each input view to the host, not its whole storage;
- a cpu->rbln copy_ into a non-contiguous view with a dtype/shape-mismatched cpu src
  converts the src on the host and writes the view's runs in place, without reading the
  dst back.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln import _C
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
        out = fn()
        torch.rbln.synchronize()
    finally:
        _rt_timing_enable(False)
    calls = {name: cnt for name, (_ns, cnt) in zip(_RT_PRIMS, _read_rt_timing())}
    return calls["v2h"] + calls["v2h_multi"], out


def _on_device(shape, dtype=torch.bfloat16):
    return torch.randn(shape).to(dtype).to(DEVICE)


@pytest.mark.test_set_ci
@needs_device
class TestFallbackRegionIO(TestCase):
    def test_boxed_fallback_on_a_prefix_view(self):
        # argmax is an explicit CPU fallback; its input is a 4-row prefix of a 512-row storage
        x = _on_device((512, 3072))
        ref = x.cpu()[:4].argmax(-1)
        got = x[:4].argmax(-1)
        torch.testing.assert_close(got.cpu(), ref)

    def test_strided_copy_from_mismatched_cpu_src_does_not_read_the_dst(self):
        # every 4th row of a [256, 8192] bf16 tensor <- fp32 rows: the src is converted on the
        # CPU and the rows (above the strided run threshold) are written in place
        x = _on_device((256, 8192))
        ref = x.cpu()
        rows = torch.randn(64, 8192)  # fp32 -> bf16 conversion + strided dst
        ref[::4].copy_(rows)
        reads, _ = _device_reads(lambda: x[::4].copy_(rows))
        self.assertEqual(reads, 0, "strided dst copy read the dst back")
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)

    def test_strided_copy_profiler_attribution(self):
        # the src conversion counts as staging; the dst is not pulled to the host, so the
        # "non-contiguous rbln dst pulled to host" site must not move (BounceSite order:
        # 1 = cpu src staged, 2 = non-contiguous dst pulled)
        x = _on_device((256, 8192))
        rows = torch.randn(64, 8192)
        before = _C._profiler_dump_bounces()
        x[::4].copy_(rows)
        torch.rbln.synchronize()
        after = _C._profiler_dump_bounces()
        self.assertEqual(after[1][0] - before[1][0], 1, "src conversion not attributed to staging")
        self.assertEqual(after[2][0] - before[2][0], 0, "dst reported as pulled to the host")

    def test_small_runs_still_correct(self):
        # below the strided run threshold the staged path remains; correctness only
        x = _on_device((256, 4096))
        ref = x.cpu()
        rows = torch.randn(64, 4096)
        ref[::4].copy_(rows)
        x[::4].copy_(rows)
        torch.testing.assert_close(x.cpu(), ref, rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
