# Owner(s): ["module: PrivateUse1"]

"""
Walkthrough example 6: nn.Module on RBLN.

Ported from walkthrough_guide/6.nn_module_to_rbln.py.

Verifies that:
- `model.to("rbln", dtype=...)` moves parameters/buffers to RBLN.
- `model.parameters()` and `model.state_dict()` reflect the RBLN device.
- A forward pass with an RBLN input produces an RBLN output.
- `model.eval()` + `torch.inference_mode()` works with the same model.
"""

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


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
class TestNNModuleOnRBLN(TestCase):
    """Walkthrough example 6: nn.Module.to('rbln') and forward pass."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_parameters_moved_to_rbln(self, dtype):
        """All parameters should be on RBLN with the requested dtype."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        params = list(model.parameters())
        self.assertGreater(len(params), 0)
        for p in params:
            self.assertEqual(p.device.type, "rbln")
            self.assertEqual(p.dtype, dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_state_dict_on_rbln(self, dtype):
        """`state_dict()` tensors should live on RBLN after `.to('rbln', ...)`."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        state = model.state_dict()
        self.assertGreater(len(state), 0)
        for tensor in state.values():
            self.assertEqual(tensor.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_forward_returns_rbln_tensor(self, dtype):
        """A forward pass with an RBLN input should keep the output on RBLN."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        x = torch.randn(4, 128, device=self.rbln_device, dtype=dtype)

        out = model(x)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.device.type, "rbln")
        self.assertEqual(out.dtype, dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_eval_mode_with_inference_mode(self, dtype):
        """`model.eval()` + `torch.inference_mode()` should run on RBLN."""
        model = _SmallNet().to(self.rbln_device, dtype=dtype)
        x = torch.randn(4, 128, device=self.rbln_device, dtype=dtype)

        model.eval()
        with torch.inference_mode():
            out = model(x)

        self.assertEqual(out.shape, (4, 64))
        self.assertEqual(out.device.type, "rbln")


instantiate_device_type_tests(TestNNModuleOnRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
