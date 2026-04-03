# Owner(s): ["module: PrivateUse1"]
"""
Tests that host (CPU fallback) ops run with float16 without casting to custom_float16,
so that precision is preserved and results match CPU float16 reference.

Previously, float16 -> custom_float16 casting for device execution caused precision loss
(e.g. div(..., rounding_mode='floor'/'trunc') giving wrong results). After the patch,
host ops keep float16 and produce correct results. These tests verify that behavior.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


# Values that most expose the precision loss when float16 is cast to custom_float16:
# - div(..., rounding_mode='floor'): 8.9922 -> 8, -6.0039 -> -7 (wrong if cast: 9, -6)
# - div(..., rounding_mode='trunc'): 8.9922 -> 8, -6.0039 -> -6 (wrong if cast)
DIV_SENSITIVE_POSITIVE = 8.9922
DIV_SENSITIVE_NEGATIVE = -6.0039


@pytest.mark.test_set_ci
class TestHostPrecisionCast(TestCase):
    """Verify host ops run in float16 and match CPU float16 results."""

    rbln_device = torch.device("rbln:0")

    def test_div_trunc_rounding_mode_float16(self):
        """
        div(..., rounding_mode='trunc') with float16.
        When this path uses host fallback, result must match CPU float16 (no custom_float16 cast).
        """
        a = torch.tensor(
            [DIV_SENSITIVE_POSITIVE, DIV_SENSITIVE_NEGATIVE, 7.5, -3.2],
            dtype=torch.float16,
            device=self.rbln_device,
        )
        b = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float16, device=self.rbln_device)
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        rbln_result = torch.div(a, b, rounding_mode="trunc")
        cpu_result = torch.div(a_cpu, b_cpu, rounding_mode="trunc")

        self.assertEqual(rbln_result.device, self.rbln_device)
        self.assertEqual(rbln_result.dtype, torch.float16)
        self.assertEqual(
            rbln_result.cpu(),
            cpu_result,
            msg="div(rounding_mode='trunc') should match CPU float16",
        )

    def test_div_floor_rounding_mode_float16(self):
        """
        div(..., rounding_mode='floor') with float16.
        When this path uses host fallback, result must match CPU float16 (no custom_float16 cast).
        """
        a = torch.tensor(
            [7.5, DIV_SENSITIVE_POSITIVE, DIV_SENSITIVE_NEGATIVE],
            dtype=torch.float16,
            device=self.rbln_device,
        )
        b = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float16, device=self.rbln_device)
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        rbln_result = torch.div(a, b, rounding_mode="floor")
        cpu_result = torch.div(a_cpu, b_cpu, rounding_mode="floor")

        self.assertEqual(rbln_result.device, self.rbln_device)
        self.assertEqual(rbln_result.dtype, torch.float16)
        self.assertEqual(
            rbln_result.cpu(),
            cpu_result,
            msg="div(rounding_mode='floor') should match CPU float16",
        )

    def test_div_by_one_float16_host_fallback(self):
        """
        div(..., 1) triggers CPU fallback (check_div_by_one_needs_fallback).
        Result must match CPU float16; no precision loss from host cast.
        """
        # Use values that are sensitive to precision (e.g. near boundary)
        x = torch.tensor(
            [DIV_SENSITIVE_POSITIVE, DIV_SENSITIVE_NEGATIVE, 1.5, 0.25],
            dtype=torch.float16,
            device=self.rbln_device,
        )
        one = torch.tensor(1.0, dtype=torch.float16, device=self.rbln_device)
        x_cpu = x.cpu()
        one_cpu = one.cpu()

        rbln_result = torch.div(x, one)
        cpu_result = torch.div(x_cpu, one_cpu)

        self.assertEqual(rbln_result.device, self.rbln_device)
        self.assertEqual(rbln_result.dtype, torch.float16)
        self.assertEqual(rbln_result.cpu(), cpu_result, msg="div by one (fallback) should match CPU float16")

    def test_mean_dim2_keepdim_float16(self):
        """
        torch.mean(x, dim=2, keepdim=True) with float16.
        When run on host (or with correct float16 path), result must match CPU float16.
        """
        # 3D shape so dim=2 is valid; include values that stress precision
        x = torch.tensor(
            [
                [[DIV_SENSITIVE_POSITIVE, 1.0], [2.0, 3.0], [4.0, 5.0]],
                [[DIV_SENSITIVE_NEGATIVE, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ],
            dtype=torch.float16,
            device=self.rbln_device,
        )
        x_cpu = x.cpu()

        rbln_result = torch.mean(x, dim=2, keepdim=True)
        cpu_result = torch.mean(x_cpu, dim=2, keepdim=True)

        self.assertEqual(rbln_result.device, self.rbln_device)
        self.assertEqual(rbln_result.dtype, torch.float16)
        self.assertEqual(
            rbln_result.cpu(),
            cpu_result,
            msg="mean(dim=2, keepdim=True) should match CPU float16",
            atol=1e-5,
            rtol=1e-5,
        )


instantiate_device_type_tests(TestHostPrecisionCast, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
