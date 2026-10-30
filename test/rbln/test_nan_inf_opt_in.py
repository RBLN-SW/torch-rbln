# Owner(s): ["module: PrivateUse1"]

from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import run_tests

from torch_rbln import _C
from torch_rbln._internal import ops_utils


@pytest.mark.parametrize("setting", [None, "OFF", "ON"])
def test_python_scan_requires_opt_in(monkeypatch, setting):
    monkeypatch.delenv("TORCH_RBLN_DEPLOY", raising=False)
    monkeypatch.delenv("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", raising=False)
    if setting is None:
        monkeypatch.delenv("TORCH_RBLN_CHECK_NAN_INF", raising=False)
    else:
        monkeypatch.setenv("TORCH_RBLN_CHECK_NAN_INF", setting)
    x = torch.full((64,), float("nan"), dtype=torch.float16)
    with (
        mock.patch.object(ops_utils, "has_invalid_tensor", return_value=True) as scan,
        mock.patch.object(ops_utils, "to_cpu", wraps=ops_utils.to_cpu) as host_copy,
    ):
        fallback = ops_utils.is_cpu_fallback_cases((x,))
    assert fallback == (setting == "ON"), (setting, fallback)
    assert scan.call_count == int(setting == "ON"), (setting, scan.call_count)
    assert host_copy.called == (setting == "ON"), (setting, host_copy.call_count)


@pytest.mark.serial
@pytest.mark.npu(count=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_opt_in_is_live_before_cold_and_warm_dispatch(monkeypatch, dtype):
    monkeypatch.delenv("TORCH_RBLN_DEPLOY", raising=False)
    monkeypatch.delenv("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", raising=False)
    monkeypatch.delenv("TORCH_RBLN_CHECK_NAN_INF", raising=False)
    x_cpu = torch.full((64,), float("nan"), dtype=dtype)
    y_cpu = torch.full((64,), 2.0, dtype=dtype)
    x, y = x_cpu.to("rbln"), y_cpu.to("rbln")
    _C._warmcache_clear()
    _C._dispatch_shim_diag_reset()
    _C._dispatch_fallback_reasons_reset()

    # A NaN input makes an accidentally enabled scanner observable as a fallback.
    torch.add(x, y)
    torch.rbln.synchronize()
    assert _C._warmcache_size() > 0, "default-off cold dispatch did not install a device entry"
    assert _C._dispatch_fallback_reasons()[1] == 0, "default-off dispatch scanned NaN inputs"

    monkeypatch.setenv("TORCH_RBLN_CHECK_NAN_INF", "ON")
    hits_before = _C._dispatch_shim_diag_dump()[2]
    result = torch.add(x, y)
    torch.testing.assert_close(result.cpu(), torch.add(x_cpu, y_cpu), equal_nan=True)
    assert _C._dispatch_fallback_reasons()[1] == 1, "opt-in did not take the NaN/Inf fallback"
    assert _C._dispatch_shim_diag_dump()[2] == hits_before, "warm hit bypassed the opt-in scan"

    monkeypatch.delenv("TORCH_RBLN_CHECK_NAN_INF")
    torch.add(x, y)
    torch.rbln.synchronize()
    assert _C._dispatch_shim_diag_dump()[2] > hits_before, "disabling the scan did not restore the warm hit"
    assert _C._dispatch_fallback_reasons()[1] == 1, "disabled scan still caused a fallback"


if __name__ == "__main__":
    run_tests()
