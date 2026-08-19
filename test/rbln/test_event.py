# Owner(s): ["module: PrivateUse1"]

"""Test suite for torch.Event with the RBLN backend.

Covers the torch.Event usage pattern (record after a non_blocking copy, synchronize
before reading the host buffer), reuse, cross-stream and cross-device waits, and the
unsupported elapsed_time path.
"""

import unittest

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

    def test_rbln_event_namespace(self):
        # torch.rbln.Event mirrors torch.cuda.Event and binds to the rbln device.
        event = torch.rbln.Event()
        self.assertEqual(event.device.type, "rbln")
        self.assertTrue(event.query())  # never recorded -> complete

    def test_rbln_event_elapsed_time_unsupported(self):
        start, end = torch.rbln.Event(enable_timing=True), torch.rbln.Event(enable_timing=True)
        start.record()
        end.record()
        with self.assertRaisesRegex(RuntimeError, "elapsed_time"):
            start.elapsed_time(end)

    def test_wait_event_across_streams(self):
        # A consumer stream waits on an event recorded on a producer stream (a real
        # device-side fence), then a host sync makes the pinned D2H visible.
        src = torch.arange(512, dtype=torch.int32).reshape(-1, 1)
        dev = src.to("rbln")
        pinned = torch.empty_like(src, pin_memory=True)
        producer, consumer = torch.rbln.Stream(), torch.rbln.Stream()
        done = torch.rbln.Event()
        with torch.rbln.stream(producer):
            pinned.copy_(dev, non_blocking=True)
            done.record()
        consumer.wait_event(done)
        consumer.synchronize()
        self.assertEqual(pinned.flatten().tolist(), list(range(512)))

    @unittest.skipIf(torch.rbln.device_count() < 2, "needs >= 2 RBLN devices")
    def test_cross_device_event_wait_degrades_to_host_sync(self):
        # Cross-device waits are not supported and must degrade to a host-side
        # synchronize (correct, serializing) rather than erroring.
        src = torch.randn(128, 128)
        dev0 = src.to("rbln:0")
        pinned = torch.empty_like(src, pin_memory=True)
        _warm = torch.zeros(1, device="rbln:1")  # materialize device 1's context  # noqa: F841
        event = torch.rbln.Event()
        with torch.rbln.stream(torch.rbln.Stream(device=0)):
            pinned.copy_(dev0, non_blocking=True)
            event.record()
        stream1 = torch.rbln.Stream(device=1)
        stream1.wait_event(event)  # device 1 waits on device 0's event
        stream1.synchronize()
        self.assertEqual(pinned, src)


if __name__ == "__main__":
    run_tests()
