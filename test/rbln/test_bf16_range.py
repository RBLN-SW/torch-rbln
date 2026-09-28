# Owner(s): ["module: PrivateUse1"]

"""bfloat16 at the edge of the format's range, on device.

A second and no model download. This is the canary for the model-level
``bfloat16 x attn_implementation="eager"`` cases that ``test/models/test_transformers.py``
skips on ATOM: it fails for the same reason, so run it before spending an hour on a model.

ATOM promotes bfloat16 to the device's custom float, whose exponent is narrower than
bfloat16's -- about 2**32 of range against 3.4e38. A value the promotion cannot represent
has to saturate; on ATOM it comes back NaN instead. HuggingFace builds its causal
mask out of ``torch.finfo(dtype).min``, so every masked position turns into NaN and takes
the attention output with it, with no error raised.

``strict=True`` is the point: once the promotion saturates, this passes unexpectedly, CI
reports it, and the model-level skip can go.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import xfail_atom


class TestBf16Range(TestCase):
    @pytest.mark.test_set_ci
    @xfail_atom("the bfloat16 -> custom float promotion returns NaN instead of saturating")
    def test_range_limit_survives_a_value_preserving_op(self, device):
        # `+ 0` cannot change the value, so whatever comes back is the promotion's doing.
        # (64, 64) rather than something tiny: small ops are scheduled on the host, which
        # does not promote and so passes either way.
        limit = torch.finfo(torch.bfloat16).min
        operand = torch.full((64, 64), limit, dtype=torch.bfloat16, device=device)

        result = (operand + torch.zeros_like(operand)).cpu()

        self.assertEqual(
            result,
            torch.full((64, 64), limit, dtype=torch.bfloat16),
            msg=(
                f"{limit} came back as {result.flatten()[0]} on {device}. A bfloat16 value "
                "the device compute type cannot hold must saturate, not become NaN."
            ),
        )


instantiate_device_type_tests(TestBf16Range, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
