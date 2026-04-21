# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 7: torch.compile graph mode with the RBLN backend.

Ported from walkthrough_guide/7.graph_mode.py.

Verifies that:
- A model can run in eager mode on RBLN after `.to("rbln", dtype=...)`.
- The same model can be compiled with `torch.compile(..., backend="rbln")`.
- Eager and graph outputs are numerically close (`torch.allclose`).
"""

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


# Tolerance for eager-vs-graph numerical comparison. Matches the original
# walkthrough script (atol=1e-4).
ATOL = 1e-4
RTOL = 1e-4


class _SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        return self.seq(x)


@pytest.mark.test_set_ci
class TestGraphMode(TestCase):
    """Walkthrough example 7: eager vs graph mode on RBLN."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_eager_forward_on_rbln(self, dtype):
        """Eager forward of the shared SmallNet should produce an RBLN tensor."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        x = torch.randn(4, 128, device=self.rbln_device, dtype=dtype)

        out = model(x)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_compiled_forward_on_rbln(self, dtype):
        """`torch.compile(..., backend='rbln')` should run without raising."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        x = torch.randn(4, 128, device=self.rbln_device, dtype=dtype)

        compiled_model = torch.compile(model, backend="rbln")
        out = compiled_model(x)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_eager_matches_graph(self, dtype):
        """Eager and graph outputs should agree within a small tolerance on CPU."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        x = torch.randn(4, 128, device=self.rbln_device, dtype=dtype)

        eager_output = model(x)

        compiled_model = torch.compile(model, backend="rbln")
        graph_output = compiled_model(x)

        self.assertEqual(eager_output.cpu(), graph_output.cpu(), atol=ATOL, rtol=RTOL)


instantiate_device_type_tests(TestGraphMode, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
