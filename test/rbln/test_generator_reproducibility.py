# Owner(s): ["module: PrivateUse1"]

"""Unit tests for the RBLN Generator initial implementation.

Coverage:
  1. Seed initialization — ``manual_seed`` correctly sets the generator's
     initial seed, which is observable through ``initial_seed``.

  2. State save and restore — the generator state can be saved and restored
     to reproduce the same random number sequence.

  3. Reproducibility — an RBLN generator initialized with the same seed as a
     CPU generator produces the same random sequence for the tested random
     operation.
"""

from __future__ import annotations

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401  (registers the rbln backend)


@pytest.mark.test_set_ci
class TestGeneratorBasic(TestCase):
    """Basic tests for the RBLN ``torch.Generator`` implementation."""

    def test_manual_seed(self):
        torch.manual_seed(42)

        self.assertEqual(torch_rbln._C.get_default_generator().initial_seed(), 42)

    def test_generator_set_seed(self):
        g = torch.Generator(device="rbln").manual_seed(4242)

        self.assertEqual(g.initial_seed(), 4242)

    def test_generator_get_and_set_state(self):
        g0 = torch.Generator(device="rbln").manual_seed(424242)
        initial_state = g0.get_state()
        x = torch.randint(1024, (10,), device="rbln", generator=g0)

        g1 = torch.Generator(device="rbln")
        g1.set_state(initial_state)
        y = torch.randint(1024, (10,), device="rbln", generator=g1)

        self.assertEqual(x.cpu(), y.cpu())

    def test_generator_reproducibility(self):
        rbln_g = torch.Generator(device="rbln").manual_seed(42424242)
        cpu_g = torch.Generator(device="cpu").manual_seed(42424242)

        rbln_x = torch.randint(1024, (10,), device="rbln", generator=rbln_g)
        cpu_x = torch.randint(1024, (10,), device="cpu", generator=cpu_g)

        self.assertEqual(rbln_x.cpu(), cpu_x)


instantiate_device_type_tests(TestGeneratorBasic, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
