# Owner(s): ["module: PrivateUse1"]

"""Regression test for FINE-621.

When the same nn.Module class is used to construct two model instances and
each one runs an eager forward followed by ``torch.compile(..., backend="rbln")``
in the same process, the second invocation's graph output returns wrong values.

Tracked as a rebel-compiler ``use_weight_sharing`` reuse-path bug; reproduces on
both rblnsw/dev and the personal dev branch, so it is independent of torch-rbln
changes.

Tests run in alphabetical order: ``test_a_*`` consumes the first invocation
(passes), and ``test_b_*`` then runs the second invocation (expected failure
until FINE-621 is fixed).
"""
import unittest

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests


DEVICE = "rbln:0"
DT = torch.float16


class _SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)

    def forward(self, x):
        return self.linear(x)


def _eager_matches_graph(testcase: TestCase) -> None:
    m = _SmallNet().to(DEVICE, dtype=DT)
    x = torch.randn(4, 128, device=DEVICE, dtype=DT)
    testcase.assertEqual(
        m(x).cpu(),
        torch.compile(m, backend="rbln")(x).cpu(),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.test_set_ci
class TestTorchCompileEagerGraphRepeat(TestCase):
    """eager + graph sequence repeated returns wrong output on second call (FINE-621)."""

    def test_a_first_invocation(self) -> None:
        _eager_matches_graph(self)

    @unittest.expectedFailure
    def test_b_second_invocation(self) -> None:
        # FINE-621: rebel-compiler's use_weight_sharing reuse path produces
        # wrong values for the second invocation's graph output. Remove the
        # ``@unittest.expectedFailure`` decorator once FINE-621 is fixed.
        _eager_matches_graph(self)


if __name__ == "__main__":
    run_tests()
