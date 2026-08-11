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
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


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


instantiate_device_type_tests(TestFp16Numerics, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
