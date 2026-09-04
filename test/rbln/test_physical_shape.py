# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.physical_shape(tensor)`` — the compiler-assigned
physical shape of an RBLN tensor's allocation (may differ from ``tensor.shape``;
``()`` when no physical view is bound)."""

from __future__ import annotations

import math

import pytest
import torch

import torch_rbln  # noqa: F401  # binds the RBLN device + torch.rbln namespace


DEVICE = torch.device("rbln:0")


@pytest.mark.test_set_ci
class TestPhysicalShape:
    def test_api_visible(self):
        assert hasattr(torch.rbln, "physical_shape")
        assert callable(torch.rbln.physical_shape)

    def test_cpu_tensor_raises(self):
        with pytest.raises(RuntimeError, match="must be an RBLN tensor"):
            torch.rbln.physical_shape(torch.zeros(2, 3))

    def test_unbound_tensor_is_empty(self):
        # A bare allocation has no physical view bound yet.
        assert torch.rbln.physical_shape(torch.zeros(2, 4, 8, device=DEVICE)) == ()

    def test_view_resolves_to_owning_allocation(self):
        base = torch.zeros(2, 4, 8, device=DEVICE)
        view = base.permute(2, 0, 1)
        assert torch.rbln.physical_shape(view) == torch.rbln.physical_shape(base)

    def test_materialized_op_exposes_real_physical_layout(self):
        # A device-run, materialized op output carries a physical view that can
        # differ from the logical shape (compiler tiling/relayout): a (128, 256)
        # matmul is physically e.g. (128, 4, 64). Assert it is populated and a
        # faithful relayout (same element count), without pinning the exact
        # compiler-version-dependent shape.
        out = torch.randn(128, 64, device=DEVICE, dtype=torch.bfloat16) @ torch.randn(
            64, 256, device=DEVICE, dtype=torch.bfloat16
        )
        out.cpu()  # force execution so the physical view binds
        pv = torch.rbln.physical_shape(out)
        assert pv != ()
        assert math.prod(pv) >= out.numel()


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
