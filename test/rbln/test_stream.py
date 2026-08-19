# Owner(s): ["module: PrivateUse1"]

"""Test suite for torch.rbln streams (and the generic torch.Stream) on RBLN.

A stream is an ordered sequence of device work: work on the same stream runs in
order, work on different streams may run concurrently. These tests cover the
torch.cuda-parity surface —
Stream / current_stream / default_stream / set_stream / stream context manager /
query / synchronize / wait_event / wait_stream — plus the intentionally
unsupported bits (stream priorities are accepted but ignored).
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@pytest.mark.test_set_ci
class TestStream(TestCase):
    def tearDown(self):
        # Streams are thread-local process state; leave the default selected so tests
        # do not leak the current stream into one another.
        torch.rbln.set_stream(torch.rbln.default_stream())
        super().tearDown()

    def test_default_stream_id_is_zero(self):
        default = torch.rbln.default_stream()
        self.assertEqual(default.stream_id, 0)
        self.assertEqual(default.device.type, "rbln")

    def test_current_defaults_to_default(self):
        self.assertEqual(torch.rbln.current_stream(), torch.rbln.default_stream())

    def test_new_stream_is_nonzero_and_distinct(self):
        s1 = torch.rbln.Stream()
        s2 = torch.rbln.Stream()
        self.assertEqual(s1.device.type, "rbln")
        self.assertNotEqual(s1.stream_id, 0)
        self.assertNotEqual(s1, s2)

    def test_stream_creation_is_bounded_by_the_pool(self):
        # Streams come from a fixed per-device pool; past its size, instances reuse one.
        ids = {torch.rbln.Stream().stream_id for _ in range(96)}
        self.assertLessEqual(len(ids), 32)
        self.assertNotIn(0, ids)

    def test_generic_torch_stream_on_rbln(self):
        # torch.Stream must work with no torch.rbln pybind — purely via the guard impl.
        s = torch.Stream(device="rbln")
        self.assertEqual(s.device.type, "rbln")
        self.assertTrue(s.query())

    def test_set_stream_and_current(self):
        s = torch.rbln.Stream()
        prev = torch.rbln.current_stream()
        try:
            torch.rbln.set_stream(s)
            self.assertEqual(torch.rbln.current_stream(), s)
        finally:
            torch.rbln.set_stream(prev)
        self.assertEqual(torch.rbln.current_stream(), prev)

    def test_stream_context_manager(self):
        s = torch.rbln.Stream()
        default = torch.rbln.current_stream()
        with torch.rbln.stream(s):
            self.assertEqual(torch.rbln.current_stream(), s)
        self.assertEqual(torch.rbln.current_stream(), default)

    def test_stream_context_none_is_noop(self):
        default = torch.rbln.current_stream()
        with torch.rbln.stream(None):
            self.assertEqual(torch.rbln.current_stream(), default)

    def test_query_and_synchronize_idle(self):
        s = torch.rbln.Stream()
        self.assertTrue(s.query())
        s.synchronize()  # idle -> no crash
        self.assertTrue(s.query())

    def test_priority_accepted_but_ignored(self):
        s = torch.rbln.Stream(priority=-1)  # accepted; RBLN has no priorities
        self.assertEqual(s.device.type, "rbln")
        self.assertEqual(s.priority, 0)
        self.assertEqual(torch.rbln.Stream.priority_range(), (0, 0))

    def test_wait_stream_and_wait_event_do_not_error(self):
        s1, s2 = torch.rbln.Stream(), torch.rbln.Stream()
        s1.wait_stream(s2)  # record an event on s2, make s1 wait it
        e = torch.rbln.Event()
        e.record(s2)
        s1.wait_event(e)  # device-side fence
        s1.synchronize()
        s2.synchronize()

    def test_copy_on_stream_gated_by_event(self):
        # Non-blocking D2H on a non-default stream, gated by an event recorded on it.
        # Pinned host memory, so the event's seq wait makes the copy host-visible.
        src = torch.arange(1024, dtype=torch.int32).reshape(-1, 1)
        dev = src.to("rbln")
        pinned = torch.empty_like(src, pin_memory=True)
        s = torch.rbln.Stream()
        e = torch.rbln.Event()
        with torch.rbln.stream(s):
            self.assertEqual(torch.rbln.current_stream(), s)
            pinned.copy_(dev, non_blocking=True)
            e.record()
        e.synchronize()
        self.assertEqual(pinned.flatten().tolist(), list(range(1024)))

    def test_matmul_on_non_default_stream(self):
        # A compute op, not just a copy, honors the current stream through dispatch.
        a = torch.randn(64, 64).to("rbln")
        b = torch.randn(64, 64).to("rbln")
        s = torch.rbln.Stream()
        with torch.rbln.stream(s):
            self.assertEqual(torch.rbln.current_stream(), s)
            c = (a @ b).relu()
        torch.rbln.synchronize()
        expected = (a.cpu() @ b.cpu()).relu()
        self.assertTrue(torch.allclose(c.cpu(), expected, atol=1e-2, rtol=1e-2))

    def test_compute_gated_by_async_copy_event(self):
        # Copy<->compute ordering across streams: the event fence is what makes the
        # producer's still-in-flight copy safe for the consumer to matmul with.
        w_cpu = torch.randn(64, 64).pin_memory()
        x = torch.randn(8, 64).to("rbln")
        producer, consumer = torch.rbln.Stream(), torch.rbln.Stream()
        ready = torch.rbln.Event()
        with torch.rbln.stream(producer):
            w = w_cpu.to("rbln", non_blocking=True)
            ready.record()
        with torch.rbln.stream(consumer):
            consumer.wait_event(ready)  # fence: w must be fully copied before use
            y = x @ w
        torch.rbln.synchronize()
        self.assertTrue(torch.allclose(y.cpu(), x.cpu() @ w_cpu, atol=1e-2, rtol=1e-2))

    def test_stream_context_selects_the_stream_device(self):
        # Selecting a stream selects its device, so allocations land there.
        if torch.rbln.device_count() < 2:
            self.skipTest("needs >= 2 RBLN devices")
        torch.rbln.set_device(0)
        with torch.rbln.stream(torch.rbln.Stream(device=1)):
            self.assertEqual(torch.rbln.current_device(), 1)
            self.assertEqual(torch.zeros(4, device="rbln").device.index, 1)
        self.assertEqual(torch.rbln.current_device(), 0)

    def test_multi_device_independent_streams(self):
        # Independent work on per-device streams is correct and isolated.
        if torch.rbln.device_count() < 2:
            self.skipTest("needs >= 2 RBLN devices")
        a0, b0 = torch.randn(32, 32).to("rbln:0"), torch.randn(32, 32).to("rbln:0")
        a1, b1 = torch.randn(32, 32).to("rbln:1"), torch.randn(32, 32).to("rbln:1")
        s0, s1 = torch.rbln.Stream(device=0), torch.rbln.Stream(device=1)
        with torch.rbln.stream(s0):
            c0 = a0 @ b0
        with torch.rbln.stream(s1):
            c1 = a1 @ b1
        torch.rbln.synchronize(0)
        torch.rbln.synchronize(1)
        self.assertTrue(torch.allclose(c0.cpu(), a0.cpu() @ b0.cpu(), atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(c1.cpu(), a1.cpu() @ b1.cpu(), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    run_tests()
