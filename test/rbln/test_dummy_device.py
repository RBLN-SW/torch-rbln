# Owner(s): ["module: PrivateUse1"]

"""Host-backed dummy device contract (RBLN_DUMMY_DEVICE=1).

Dummy mode is an explicit opt-in that lets torch-rbln construct device tensors
and run memory transfers on host memory, so a model can be traced/compiled on a
host with no NPU. It is forced regardless of physical NPU presence.

RBLN_DUMMY_DEVICE must be set before ``import torch_rbln`` (the device-mapping
singleton reads it once at init), so each case runs in a fresh subprocess with
the env set.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _clean_env() -> dict:
    """A hermetic env copy for subprocess tests.

    Strips external state these tests must not inherit:
      - TORCH_RBLN_DIAGNOSE: another test module (test_find_and_load_tvm_library)
        sets it at import (it disables backend init) and never restores it, so it
        leaks into the pytest process and then into ``os.environ`` here.
      - RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE: a parent shell/CI mapping would
        change the dummy logical device count; tests assume the default of 1 and
        set their own map via ``env_extra`` when they need one.
    """
    env = dict(os.environ)
    for key in ("TORCH_RBLN_DIAGNOSE", "RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE"):
        env.pop(key, None)
    return env


def _run_with_dummy(snippet: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = _clean_env()
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
        # default count is 1 (no RBLN_DEVICE_MAP set)
        assert torch.rbln.device_count() == 1, torch.rbln.device_count()
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
            auto_determine_num_devices,
            get_physical_device_ids,
        )
        assert get_physical_device_ids(0) == [0, 1], get_physical_device_ids(0)
        assert auto_determine_num_devices(0) == 2
        assert auto_determine_num_devices(1) == 2
        print("OK")
        """,
        env_extra={"RBLN_DEVICE_MAP": "[0,1],[2,3]"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_boolean_spellings():
    # RBLN_DUMMY_DEVICE is a boolean flag (shared with the rebel runtime); the
    # logical count comes from RBLN_DEVICE_MAP (default 1), not this value. Cover
    # the full truthy/falsy spellings the parser recognizes.
    for truthy in ("1", "true", "t", "on", "yes", "y"):
        proc = _run_with_dummy(
            "import torch, torch_rbln; "
            "assert torch.rbln.is_dummy_device() is True; "
            "assert torch.rbln.device_count() == 1; print('OK')",
            env_extra={"RBLN_DUMMY_DEVICE": truthy},
        )
        _assert_ok(proc)
        assert "OK" in proc.stdout

    for falsy in ("0", "false", "f", "off", "no", "n"):
        env = _clean_env()
        env["RBLN_DUMMY_DEVICE"] = falsy
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_rbln; assert torch.rbln.is_dummy_device() is False; print('OK')",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        _assert_ok(proc)
        assert "OK" in proc.stdout


@pytest.mark.parametrize("invalid", ["4", "2", "maybe"])
def test_invalid_value_fails_fast(invalid):
    # RBLN_DUMMY_DEVICE is boolean, NOT an integer device count. A non-boolean
    # value is rejected loudly by the rebel runtime at startup (process aborts on
    # import) rather than silently disabling dummy mode — pin that contract.
    env = _clean_env()
    env["RBLN_DUMMY_DEVICE"] = invalid
    proc = subprocess.run(
        [sys.executable, "-c", "import torch, torch_rbln"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "boolean" in (proc.stdout + proc.stderr).lower()


def test_is_dummy_device_api():
    proc = _run_with_dummy("import torch, torch_rbln; assert torch.rbln.is_dummy_device() is True; print('OK')")
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_is_dummy_device_false_without_env():
    env = _clean_env()
    env.pop("RBLN_DUMMY_DEVICE", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import torch, torch_rbln; assert torch.rbln.is_dummy_device() is False; print('OK')"],
        env=env,
        capture_output=True,
        text=True,
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_device_map_rejects_duplicate_physical_id():
    # Duplicate ids are rejectable without hardware; dummy must reject like real.
    proc = _run_with_dummy(
        "import torch, torch_rbln; torch.rbln.device_count()",
        env_extra={"RBLN_DEVICE_MAP": "[0],[0]"},
    )
    assert proc.returncode != 0
    assert "more than one logical device" in (proc.stdout + proc.stderr).lower()


def test_device_map_rejects_invalid_group_size():
    # TP=3 is not an allowed group size; dummy must reject it like real hardware
    # so a config that compiles under dummy also runs on an NPU.
    proc = _run_with_dummy(
        "import torch, torch_rbln; torch.rbln.device_count()",
        env_extra={"RBLN_DEVICE_MAP": "[0,1,2]"},
    )
    assert proc.returncode != 0
    assert "valid sizes" in (proc.stdout + proc.stderr).lower()


def test_torch_compile_tracing_smoke():
    # Dummy mode's purpose is the no-NPU compile path. A custom backend that does
    # not lower to the device confirms dynamo can fakify and trace over dummy rbln
    # tensors (full rebel compilation is execution-triggered and validated via the
    # vLLM VLLM_RBLN_COMPILE_ONLY flow, which writes artifacts without executing).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        def backend(gm, example_inputs):
            return gm.forward  # eager replay; ops fall back to CPU in dummy mode
        f = torch.compile(lambda x: x * 2 + 1, backend=backend, fullgraph=True)
        x = torch.ones(4, device="rbln:0")
        assert f(x).cpu().tolist() == [3.0, 3.0, 3.0, 3.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_set_device_layout_like_is_noop():
    # Runtime-free in dummy mode: same-device whole allocations -> no-op (no NPU).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        target = torch.empty(4, device="rbln:0")
        ref = torch.empty(4, device="rbln:0")
        torch.rbln.set_device_layout_like(target, ref)
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_offload_is_noop():
    # torch.rbln.offload() must not touch the runtime in dummy mode.
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        with torch.rbln.offload():
            t = torch.tensor([1.0, 2.0], device="rbln:0")
        assert t.cpu().tolist() == [1.0, 2.0]
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_npus_per_device_sets_single_group_tp():
    # RBLN_NPUS_PER_DEVICE=N (no RBLN_DEVICE_MAP) -> one logical device of size N (TP=N).
    proc = _run_with_dummy(
        """
        import torch, torch_rbln
        from torch_rbln._internal.rsd_utils import auto_determine_num_devices, get_physical_device_ids
        assert torch.rbln.device_count() == 1, torch.rbln.device_count()
        assert get_physical_device_ids(0) == [0, 1, 2, 3], get_physical_device_ids(0)
        assert auto_determine_num_devices(0) == 4
        torch.tensor(1.0, device="rbln:0")  # device of size N is usable
        print("OK")
        """,
        env_extra={"RBLN_NPUS_PER_DEVICE": "4"},
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


def test_torch_compile_compile_only_produces_artifact():
    # End-to-end integrity: on a no-NPU host, dummy device + the (pinned) compiler
    # compile a model and write a .rbln artifact via compile-only. Execution still
    # needs an NPU, so the call CPU-falls-back or raises; the artifact is the
    # contract under test.
    proc = _run_with_dummy(
        """
        import glob, os, tempfile
        import torch, torch_rbln
        cache = tempfile.mkdtemp(prefix="rbln_dummy_compile_")

        class Net(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) * 2.0 + 1.0

        m = Net().eval().to("rbln:0")
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="rbln:0")
        try:
            torch.compile(
                m,
                backend="rbln",
                dynamic=False,
                options={"mode": ["compile_only"], "cache_dir": cache, "npu": "RBLN-CA25"},
            )(x)
        except Exception:
            pass
        arts = glob.glob(os.path.join(cache, "*.rbln"))
        assert len(arts) == 1 and os.path.getsize(arts[0]) > 0, arts
        print("OK")
        """
    )
    _assert_ok(proc)
    assert "OK" in proc.stdout


# NOTE: overlap/self-copy safety of the dummy v2v path (memmove, not memcpy) is
# covered in test/cpp/core/RBLNDummyDeviceTest.cpp::V2VHandlesOverlap — PyTorch's
# copy_ guards against aliasing storage before dispatch, so the overlap cannot be
# driven from the Python layer.


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
