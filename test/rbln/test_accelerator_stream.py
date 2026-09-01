# Owner(s): ["module: PrivateUse1"]

"""torch.accelerator stream/event conformance for the RBLN backend.

Ported from PyTorch's own accelerator test suite (pytorch v2.11.0
``test/test_accelerator.py``). These cases are device-agnostic — they drive the
generic ``torch.accelerator`` / ``torch.Stream`` / ``torch.Event`` API, which
routes to whichever backend is the current accelerator. On an RBLN host that is
``rbln``, so running them here exercises the RBLN DeviceGuard stream/event
implementation exactly the way upstream exercises CUDA/XPU/MPS.

Kept as close to the upstream bodies as possible so divergences are easy to spot.
"""

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TEST_ACCELERATOR, TEST_MULTIACCELERATOR, TestCase


@pytest.mark.test_set_ci
@pytest.mark.skipif(not TEST_ACCELERATOR, reason="no accelerator (rbln) detected")
class TestAcceleratorStream(TestCase):
    def test_generic_stream_behavior(self):
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(1000)
        b = torch.randn(1000)
        c = a + b
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream(), s2)
        a_acc = a.to(torch.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(torch.accelerator.current_accelerator(), non_blocking=True)
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.accelerator.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)

    def test_current_stream_query(self):
        s = torch.accelerator.current_stream()
        self.assertEqual(torch.accelerator.current_stream(s.device), s)
        self.assertEqual(torch.accelerator.current_stream(s.device.index), s)
        self.assertEqual(torch.accelerator.current_stream(str(s.device)), s)
        other_device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "doesn't match the current accelerator"):
            torch.accelerator.current_stream(other_device)

    def test_stream_context_manager(self):
        prev_stream = torch.accelerator.current_stream()
        with torch.Stream() as s:
            self.assertEqual(torch.accelerator.current_stream(), s)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    def test_stream_context_manager_reentrance(self):
        prev_stream = torch.accelerator.current_stream()
        s0 = torch.Stream()
        with s0, s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)
        s1 = torch.Stream()
        with s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
            with s1:
                self.assertEqual(torch.accelerator.current_stream(), s1)
                with s0:
                    self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    def test_generic_event_behavior(self):
        event1 = torch.Event(enable_timing=False)
        event2 = torch.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.Event(enable_timing=True)
        event2 = torch.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.Event()
        event2 = torch.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    @pytest.mark.skipif(not TEST_MULTIACCELERATOR, reason="only one accelerator detected")
    def test_generic_multi_device_behavior(self):
        orig_device = torch.accelerator.current_device_index()
        target_device = (orig_device + 1) % torch.accelerator.device_count()

        torch.accelerator.set_device_index(target_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.set_device_index(orig_device)
        self.assertEqual(orig_device, torch.accelerator.current_device_index())

        s1 = torch.Stream(target_device)
        torch.accelerator.set_stream(s1)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.synchronize(orig_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())

    @pytest.mark.skipif(not TEST_MULTIACCELERATOR, reason="only one accelerator detected")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        src_prev_stream = torch.accelerator.current_stream()
        dst_prev_stream = torch.accelerator.current_stream(dst_device)
        with torch.Stream(dst_device) as dst_stream:
            self.assertEqual(torch.accelerator.current_device_index(), dst_device)
            self.assertEqual(torch.accelerator.current_stream(), dst_stream)
            self.assertEqual(torch.accelerator.current_stream(src_device), src_prev_stream)
        self.assertEqual(torch.accelerator.current_device_index(), src_device)
        self.assertEqual(torch.accelerator.current_stream(), src_prev_stream)
        self.assertEqual(torch.accelerator.current_stream(dst_device), dst_prev_stream)


if __name__ == "__main__":
    run_tests()
