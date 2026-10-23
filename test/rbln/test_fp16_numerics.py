# Owner(s): ["module: PrivateUse1"]

"""float16 numerical accuracy on device, against an fp32 CPU reference.

A few seconds and no model download, so this is the first thing to run when a
model-level accuracy test moves: it says whether float16 arithmetic on the target
still lands where it used to, before anyone spends an hour on a model.

The bounds are empirical for the current target and are not a guarantee of any
particular precision -- they only have to be tight enough that a meaningful shift
shows up. Re-derive them (the failure message prints the measured value) if the
target's compute precision policy changes.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import DLFLOAT16_DIVISION_ACCURACY, to_dlfloat16

from torch_rbln._internal.ops_utils import dispatches


def _operands(numel, seed, lo=-4.0, hi=4.0):
    """Deterministic float16-exact operands, returned as fp32.

    Rounding to float16 up front keeps the fp32 reference and the device input on
    the same values, so the comparison measures the device and not this helper.
    """
    gen = torch.Generator().manual_seed(seed)
    raw = torch.rand(numel, generator=gen, dtype=torch.float32) * (hi - lo) + lo
    return raw.to(torch.float16).float()


def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


def _sum_last_dim(a):
    return a.sum(-1)


def _matmul(a, b):
    return a @ b


def _silu_mul(a, b):
    return torch.nn.functional.silu(a) * b


# (name, build operands, op, relative bound) -- bounds carry ~3x headroom over the
# measured deviation so ordinary build-to-build variation does not trip them.
CASES = [
    ("add", lambda: (_operands(4096, 1), _operands(4096, 2)), _add, 1.5e-2),
    ("mul", lambda: (_operands(4096, 3), _operands(4096, 4)), _mul, 1.5e-2),
    ("sum_last_dim", lambda: (_operands(64 * 64, 5).reshape(64, 64),), _sum_last_dim, 1e-3),
    (
        "matmul",
        lambda: (_operands(64 * 64, 6).reshape(64, 64), _operands(64 * 64, 7).reshape(64, 64)),
        _matmul,
        1.5e-2,
    ),
    ("silu_mul", lambda: (_operands(4096, 8), _operands(4096, 9)), _silu_mul, 2e-2),
]


class TestFp16Numerics(TestCase):
    @pytest.mark.test_set_ci
    @parametrize("name,build,op,bound", CASES)
    def test_fp16_matches_fp32_reference(self, device, name, build, op, bound):
        operands = build()
        reference = op(*operands)
        result = op(*[t.to(torch.float16).to(device) for t in operands]).float().cpu()

        deviation = (result - reference).abs().max().item()
        scale = max(reference.abs().max().item(), 1e-9)
        relative = deviation / scale

        # An exact match means the comparison is not measuring device float16 at all
        # -- the op ran somewhere else, or in a wider dtype. That passes an upper
        # bound silently, so reject it explicitly.
        self.assertGreater(
            deviation,
            0.0,
            msg=(
                f"{name}: float16 on {device} matched the fp32 reference exactly, which "
                "float16 cannot do here. The op is not running as float16 on the "
                "device, so this case is no longer measuring anything."
            ),
        )
        self.assertLessEqual(
            relative,
            bound,
            msg=(
                f"{name}: float16 on {device} deviates from the fp32 reference by "
                f"{deviation:.8g} ({relative:.3e} relative), past the {bound:.3e} bound. "
                "The target's float16 accuracy changed; re-check the model-level "
                "tolerances and re-derive this bound."
            ),
        )

    @pytest.mark.test_set_ci
    @dtypes(torch.float16, torch.float32)
    def test_values_reach_dlfloat16_as_to_dlfloat16_rounds(self, device, dtype):
        """ceil on the device sees every float16 up to 2048 as ``to_dlfloat16`` rounds it.

        A float16 keeps one mantissa bit more than dlfloat16, so every other value is a tie and the
        ceiling tells the tie rules apart: 1022.5 -> 1022 to even, -> 1023 away from zero."""
        if not dispatches(dtype):
            self.skipTest(f"{dtype} runs on the host under this float32 precision")
        magnitudes = torch.arange(0, 0x6801, dtype=torch.int16).view(torch.float16)
        x = torch.cat([magnitudes, -magnitudes])
        x = torch.cat([x, x[: -x.numel() % 64]]).reshape(-1, 64).to(dtype)
        result = torch.ceil(x.to(device)).cpu()
        expected = torch.ceil(to_dlfloat16(x)).to(dtype)
        mismatched = result != expected
        self.assertFalse(mismatched.any(), msg=f"first mismatches at {x[mismatched][:8].tolist()}")

    @pytest.mark.test_set_ci
    def test_division_stays_within_its_accuracy(self, device):
        """The quotient the device computes from the dlfloat16 operands, which float16 holds exactly."""
        gen = torch.Generator().manual_seed(10)
        x = (torch.randn(4096, 64, generator=gen) * 8).to(torch.float16)
        y = (torch.randn(4096, 64, generator=gen) + 0.1).to(torch.float16)
        result = torch.div(x.to(device), y.to(device)).cpu().double()
        exact = to_dlfloat16(x).double() / to_dlfloat16(y).double()
        finite = torch.isfinite(result) & (exact != 0)
        relative = ((result - exact).abs() / exact.abs())[finite].max().item()
        self.assertLessEqual(
            relative,
            DLFLOAT16_DIVISION_ACCURACY,
            msg=f"division is off by {relative / 2**-9:.2f} * 2**-9 relative, past DLFLOAT16_DIVISION_ACCURACY",
        )


class TestDlfloat16Emulation(TestCase):
    ulp = 2.0**-9

    def _assert_rounds(self, dtype, values, expected):
        values = torch.tensor(values, dtype=dtype)
        self.assertEqual(to_dlfloat16(values), torch.tensor(expected), atol=0, rtol=0)
        self.assertEqual(to_dlfloat16(-values), -torch.tensor(expected), atol=0, rtol=0)

    @pytest.mark.test_set_ci
    def test_float16_ties_go_to_even(self):
        self._assert_rounds(
            torch.float16, [1022.5, 1023.5, 1 + self.ulp / 2, 1 + 3 * self.ulp / 2], [1022, 1024, 1, 1 + 2 * self.ulp]
        )

    @pytest.mark.test_set_ci
    def test_float32_ties_go_away_from_zero(self):
        self._assert_rounds(
            torch.float32,
            [1022.5, 1023.5, 1 + self.ulp / 2, 1 + self.ulp / 4, 1 + 3 * self.ulp / 4, 2.0**-31],
            [1023, 1024, 1 + self.ulp, 1, 1 + self.ulp, 2.0**-31],
        )

    @pytest.mark.test_set_ci
    def test_bfloat16_is_exact(self):
        self._assert_rounds(torch.bfloat16, [1.5, 1.9921875], [1.5, 1.9921875])


instantiate_device_type_tests(TestFp16Numerics, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
