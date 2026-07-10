# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `aten::arange.start_out` implementation.

`arange_start_out_rbln` (RBLNTensorFactories.cpp) implements
arange(start, end, step, *, out=tensor) directly on the RBLN device:
* Computes the expected length as ceil((end - start) / step) and resizes
  `out` to match (mirrors PyTorch's structured meta function which sizes
  the output before dispatch on CPU/CUDA).
* Uses acquire_host_ptr_for_overwrite to skip the device→host sync
  (write-only) and fills out[i] = start + i*step on the host pointer.
* Commits via return_borrowed(updated=true) so the next device-side read
  picks up the new bytes lazily.

These tests verify the result matches CPU `torch.arange` across all
supported dtypes and across forward / reverse / fractional / single-step
configurations.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _arange_via_out(start, end, step, dtype, device, out_numel=0):
    """Call torch.arange with an explicit out= (triggers .start_out). ``out_numel``
    sizes the pre-allocated out: 0 = functional path; nonzero exercises resize."""
    out = torch.empty(out_numel, dtype=dtype, device=device)
    return torch.arange(start, end, step, out=out)


@pytest.mark.test_set_ci
class TestArangeStartOutRBLN(TestCase):
    """`aten::arange.start_out` on RBLN must match CPU."""

    def _check(self, start, end, step, dtype):
        ref = _arange_via_out(start, end, step, dtype, "cpu")
        out = _arange_via_out(start, end, step, dtype, "rbln")
        self.assertEqual(out.device.type, "rbln")
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        self.assertEqual(out.to("cpu"), ref)

    @parametrize(
        "start,end,step,dtype",
        [
            # Integer dtypes — forward / signed / strided.
            (0, 8, 1, torch.int64),
            (0, 128, 1, torch.int64),
            (5, 11, 2, torch.int64),
            (0, 16, 1, torch.int32),
            (-4, 4, 1, torch.int32),
            (0, 32, 4, torch.int16),
            (-3, 3, 1, torch.int8),
            (0, 16, 2, torch.uint8),
            # Float / double — forward, fractional step.
            (0.0, 5.0, 0.5, torch.float32),
            (-1.0, 1.0, 0.25, torch.float32),
            (0.0, 4.0, 1.0, torch.float64),
            # Reverse step (negative).
            (10, 0, -2, torch.int64),
            (5.0, 0.0, -0.5, torch.float32),
        ],
    )
    def test_matches_cpu(self, start, end, step, dtype):
        self._check(start, end, step, dtype)

    def test_empty_range_noop(self):
        # start == end → length 0
        out = _arange_via_out(3, 3, 1, torch.int64, "rbln")
        self.assertEqual(out.numel(), 0)

    def test_zero_step_raises(self):
        with self.assertRaises(RuntimeError):
            _arange_via_out(0, 5, 0, torch.int64, "rbln")

    def test_undersized_out_grows(self):
        """Undersized out= is grown in place by arange.start_out's structured
        kernel (rbln storage is resizable). Historically raised "not resizable"."""
        out = _arange_via_out(0, 8, 1, torch.int64, "rbln", out_numel=2)
        self.assertEqual(out.numel(), 8)
        self.assertEqual(out.to("cpu"), torch.arange(0, 8, 1, dtype=torch.int64))


instantiate_device_type_tests(TestArangeStartOutRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
