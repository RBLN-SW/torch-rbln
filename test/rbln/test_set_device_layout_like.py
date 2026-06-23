# Owner(s): ["module: PrivateUse1"]

"""Tests for ``torch.rbln.set_device_layout_like(target, ref)``.

The op configures ``target``'s device allocation to match a device-resident
``ref``'s layout and dtype, without copying data, so a later ``target`` <->
``ref`` device-to-device copy stays on the fast path. The binding validates its
inputs first: both must be RBLN tensors, on the same device, with the same
dtype, and each a whole base allocation (not a view) — a dtype mismatch would
reinterpret ``target``'s buffer as a different dtype.

Coverage: API visibility, and the rejected-input paths.
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

    def test_ref_not_device_resident_raises(self):
        """``ref`` must be device-resident; this is enforced. Valid
        same-dtype/-device RBLN tensors whose ``ref`` is not device-resident (a
        plain eager tensor) raise rather than silently mis-configure the layout.
        The successful end-to-end path — with a device-resident KV cache as
        ``ref`` — is exercised by the LMCache-RBLN device-tensor integration,
        which a standalone unit test cannot easily set up."""
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
        """A storage_offset>0 view is not a whole base allocation and is rejected
        with a clear error."""
        base = torch.empty(64, dtype=torch.float16, device=DEVICE)
        view = base[8:24]
        assert view.storage_offset() != 0
        ref = torch.empty(16, dtype=torch.float16, device=DEVICE)
        with pytest.raises(RuntimeError, match="whole base"):
            torch.rbln.set_device_layout_like(view, ref)

    def test_narrowing_view_target_raises(self):
        """A storage_offset==0 view that doesn't span its storage (``base[:k]``)
        is still not a whole base allocation — reject it."""
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
