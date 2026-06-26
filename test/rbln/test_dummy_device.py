"""Host-backed dummy device contract (RBLN_DUMMY_DEVICE=1).

Dummy mode is an explicit opt-in that lets torch-rbln construct device tensors
and run memory transfers on host memory, so a model can be traced/compiled on a
host with no NPU. It is forced regardless of physical NPU presence.

RBLN_DUMMY_DEVICE must be set before ``import torch_rbln`` (the device-mapping
singleton reads it once at init), so each case runs in a fresh subprocess with
the env set.
"""

import subprocess
import sys
import textwrap

import pytest


def _run_with_dummy(snippet: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["RBLN_DUMMY_DEVICE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_reports_logical_device_but_no_physical():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        assert torch.rbln.device_count() >= 1, torch.rbln.device_count()
        assert torch.rbln.is_available() is True
        # physical count must not query the runtime in dummy mode; reports 0.
        assert torch.rbln.physical_device_count() == 0, torch.rbln.physical_device_count()
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_construct_device_tensor_with_value():
    # The exact pattern that fails on a no-NPU host without dummy mode:
    # torch.tensor(-inf, device='rbln:0') in vLLM's logits processor __init__.
    proc = _run_with_dummy(
        """
        import math, torch, torch_rbln
        t = torch.tensor(-math.inf, device="rbln:0")
        assert t.device.type == "rbln"
        back = t.cpu().item()
        assert math.isinf(back) and back < 0, back
        # device='rbln' (no index) resolves to the current device
        t2 = torch.tensor([1.0, 2.0, 3.0], device="rbln")
        assert t2.cpu().tolist() == [1.0, 2.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_factories_and_scalar_readback():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        assert torch.zeros(4, device="rbln:0").cpu().tolist() == [0.0, 0.0, 0.0, 0.0]
        assert torch.full((2, 2), 7.0, device="rbln:0").cpu().tolist() == [[7.0, 7.0], [7.0, 7.0]]
        assert torch.arange(0, 5, device="rbln:0").cpu().tolist() == [0, 1, 2, 3, 4]
        assert torch.tensor(42, device="rbln:0").item() == 42
        x = torch.full((2,), 3.0, device="rbln:0")
        assert x.clone().cpu().tolist() == [3.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_set_device_and_context_do_not_raise():
    # With device_count() >= 1 the count-guards in torch_rbln.device pass exactly
    # like a real device, so no dummy-specific Python branch is needed.
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        torch.rbln.set_device(0)
        assert torch.rbln.current_device() == 0
        with torch.rbln.device(0):
            _ = torch.tensor(1.0, device="rbln")
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_count_honors_device_map_group_count():
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        # 3 logical-device groups -> 3 dummy logical devices.
        assert torch.rbln.device_count() == 3, torch.rbln.device_count()
        torch.tensor(1.0, device="rbln:2")  # highest index must be usable
        print("OK")
        """,
        env_extra={"RBLN_DEVICE_MAP": "[0],[1],[2]"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_map_preserves_tp_shape():
    # RBLN_DEVICE_MAP group sizes must survive so torch.compile's auto TP sizing
    # still works (topology keeps non-empty physical-id lists in dummy mode).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        from torch_rbln._internal.rsd_utils import (
            auto_determine_tensor_parallel_size,
            get_physical_device_ids,
        )
        assert get_physical_device_ids(0) == [0, 1], get_physical_device_ids(0)
        assert auto_determine_tensor_parallel_size(0) == 2
        assert auto_determine_tensor_parallel_size(1) == 2
        print("OK")
        """,
        env_extra={"RBLN_DEVICE_MAP": "[0,1],[2,3]"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


# NOTE: overlap/self-copy safety of the dummy v2v path (memmove, not memcpy) is
# covered in test/cpp/core/RBLNDummyDeviceTest.cpp::V2VHandlesOverlap — PyTorch's
# copy_ guards against aliasing storage before dispatch, so the overlap cannot be
# driven from the Python layer.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
