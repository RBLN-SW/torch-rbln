# Owner(s): ["module: PrivateUse1"]

"""
Test suite for the RBLN-native `aten::_local_scalar_dense` implementation.

`_local_scalar_dense_rbln` (RBLNTensorFactories.cpp) replaces the default
fallback_rbln registration. It is the op behind `tensor.item()` and is
called whenever PyTorch needs a 1-element tensor's value as a Python
scalar. The device→host transfer is unavoidable (semantics require it on
the host) but bypassing cpu_fallback_rbln's schema cache + redispatch
overhead saves tens of microseconds per call — important on prefill where
vLLM's BasevLLMParameter dispatch hook may trigger thousands of these.

These tests verify the returned scalar matches the CPU reference across
all supported dtypes.
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@pytest.mark.test_set_ci
class TestLocalScalarDenseRBLN(TestCase):
    """`aten::_local_scalar_dense` via `tensor.item()` on RBLN must match CPU."""

    def _check_item(self, value, dtype):
        cpu_t = torch.tensor(value, dtype=dtype)
        rbln_t = cpu_t.to("rbln")
        self.assertEqual(rbln_t.item(), cpu_t.item())

    @parametrize(
        "value,dtype",
        [
            (-12345, torch.int64),
            (0, torch.int64),
            (2**31, torch.int64),
            (-7, torch.int32),
            (2**30, torch.int32),
            (123, torch.int16),
            (-456, torch.int16),
            (-3, torch.int8),
            (99, torch.int8),
            (200, torch.uint8),
            (0, torch.uint8),
            (True, torch.bool),
            (False, torch.bool),
            (3.14159, torch.float32),
            (-0.0, torch.float32),
            (2.718281828, torch.float64),
            (1.5, torch.float16),
            (0.5, torch.bfloat16),
        ],
    )
    def test_matches_cpu(self, value, dtype):
        self._check_item(value, dtype)

    def test_one_element_multi_dim(self):
        # A 1-element tensor with non-trivial shape — item() still works.
        t = torch.tensor([[[42]]], dtype=torch.int64, device="rbln")
        self.assertEqual(t.item(), 42)

    def test_multi_element_raises(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64, device="rbln")
        with self.assertRaises(RuntimeError):
            t.item()


instantiate_device_type_tests(TestLocalScalarDenseRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
