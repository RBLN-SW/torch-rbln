# Owner(s): ["module: PrivateUse1"]

"""
Test suite for torch.Event with the RBLN backend.

RBLN has a single in-order copy queue per device, so an event records the
device it was captured on and every wait drains that device's queue. This
gives `torch.Event` the CUDA usage pattern (record after a `non_blocking`
copy, synchronize before reading the host buffer) without per-event fences.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestEvent(TestCase):
    def test_event_default_device_is_rbln(self):
        event = torch.Event()
        self.assertEqual(event.device.type, "rbln")

    def test_query_before_record_is_true(self):
        self.assertTrue(torch.Event().query())

    def test_synchronize_before_record_is_noop(self):
        torch.Event().synchronize()

    def test_record_synchronize_d2h_pinned(self):
        # The vllm-rbln transfer_event pattern: non_blocking D2H into a pinned
        # buffer, record, synchronize, then read on the host.
        src = torch.randn(256, 256)
        dev = src.to("rbln")
        pinned = torch.empty_like(src, pin_memory=True)
        event = torch.Event()
        pinned.copy_(dev, non_blocking=True)
        event.record()
        event.synchronize()
        self.assertEqual(pinned, src)
        self.assertTrue(event.query())

    def test_record_synchronize_h2d_pinned(self):
        src = torch.randn(128, 128).pin_memory()
        dev = src.to("rbln", non_blocking=True)
        event = torch.Event()
        event.record()
        event.synchronize()
        self.assertEqual(dev.cpu(), src)

    def test_event_is_reusable(self):
        src = torch.arange(1024, dtype=torch.int64).reshape(-1, 1)
        dev = src.to("rbln")
        pinned = torch.empty_like(src, pin_memory=True)
        event = torch.Event()
        for _ in range(4):
            pinned.copy_(dev, non_blocking=True)
            event.record()
            event.synchronize()
            self.assertEqual(pinned.flatten().tolist(), list(range(1024)))

    def test_elapsed_time_unsupported(self):
        start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
        start.record()
        end.record()
        with self.assertRaisesRegex(RuntimeError, "elapsedTime"):
            start.elapsed_time(end)


if __name__ == "__main__":
    run_tests()
