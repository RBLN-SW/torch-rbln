# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.set_device_layout_like(target, ref)``.

The op points ``target``'s device allocation at the same physical layout kind
(NT/WT) and dtype as a device-resident ``ref``, without copying data, so a
later ``target`` <-> ``ref`` D2D copy stays on the fast path. The binding hands
raw ``data_ptr()`` values to the runtime, so it validates the tensors first:
both must be RBLN tensors on the same device with the same dtype. (dtype must
match because the runtime derives the element count from ``target``'s byte size
divided by ``ref``'s element size — a differing dtype would re-type the alloc
out from under ``target``'s tensor metadata.)

Coverage: API visibility, a valid call, and the rejected-input paths.
"""

from __future__ import annotations

import pytest
import torch
import torch_rbln  # noqa: F401  # binds the RBLN device + torch.rbln namespace

DEVICE = torch.device("rbln:0")


@pytest.mark.test_set_ci
class TestSetDeviceLayoutLike:
    def test_api_visible(self):
        assert hasattr(torch.rbln, "set_device_layout_like")
        assert callable(torch.rbln.set_device_layout_like)

    def test_ref_without_physical_view_raises(self):
        """``ref`` must be device-resident (have a physical view); the runtime
        enforces this. Valid same-dtype/-device RBLN tensors whose ``ref`` has no
        physical view (a plain eager tensor) raise rather than silently
        mis-configure the layout. The successful end-to-end path — with a
        device-resident KV cache as ``ref`` — is exercised by the LMCache-RBLN
        device-tensor integration, which needs a compiled graph to materialize a
        physical view that a standalone unit test cannot set up."""
        ref = torch.ones((2, 4, 8), dtype=torch.float16, device=DEVICE) + 1
        target = torch.empty(2 * 4 * 8, dtype=torch.float16, device=DEVICE)
        with pytest.raises(RuntimeError, match="layout_like"):
            torch.rbln.set_device_layout_like(target, ref)

    def test_cpu_target_raises(self):
        ref = torch.ones((4,), dtype=torch.float16, device=DEVICE) + 1
        cpu_target = torch.empty((4,), dtype=torch.float16)
        with pytest.raises(RuntimeError, match="target must be an RBLN tensor"):
            torch.rbln.set_device_layout_like(cpu_target, ref)

    def test_cpu_ref_raises(self):
        target = torch.empty((4,), dtype=torch.float16, device=DEVICE)
        cpu_ref = torch.ones((4,), dtype=torch.float16)
        with pytest.raises(RuntimeError, match="ref must be an RBLN tensor"):
            torch.rbln.set_device_layout_like(target, cpu_ref)

    def test_dtype_mismatch_raises(self):
        ref = torch.ones((4,), dtype=torch.float16, device=DEVICE) + 1
        target = torch.empty((4,), dtype=torch.float32, device=DEVICE)
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.rbln.set_device_layout_like(target, ref)

    def test_offset_view_target_raises(self):
        """A storage_offset>0 view's data_ptr() is base+offset, which the runtime
        can't resolve — reject with a clear error instead of a cryptic vaddr
        failure."""
        base = torch.empty(64, dtype=torch.float16, device=DEVICE)
        view = base[8:24]
        assert view.storage_offset() != 0
        ref = torch.empty(16, dtype=torch.float16, device=DEVICE)
        with pytest.raises(RuntimeError, match="whole base"):
            torch.rbln.set_device_layout_like(view, ref)

    def test_narrowing_view_target_raises(self):
        """A storage_offset==0 view that doesn't span its storage (``base[:k]``):
        data_ptr() matches the base, but the runtime would size the layout from
        the whole allocation, not the view — reject it."""
        base = torch.empty(64, dtype=torch.float16, device=DEVICE)
        view = base[:16]
        assert view.storage_offset() == 0 and view.is_contiguous()
        ref = torch.empty(16, dtype=torch.float16, device=DEVICE)
        with pytest.raises(RuntimeError, match="whole base"):
            torch.rbln.set_device_layout_like(view, ref)

    def test_view_ref_raises(self):
        target = torch.empty(16, dtype=torch.float16, device=DEVICE)
        ref_view = torch.empty(64, dtype=torch.float16, device=DEVICE)[:16]
        with pytest.raises(RuntimeError, match="whole base"):
            torch.rbln.set_device_layout_like(target, ref_view)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
