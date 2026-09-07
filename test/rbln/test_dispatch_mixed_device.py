# Owner(s): ["module: PrivateUse1"]
"""Mixed CPU/rbln operands with an extended dispatch catalog (TORCH_RBLN_DISPATCH_DTYPES).

Before the catalog could be extended, int operands never reached the compile path: the C++
boxed CPU fallback took them and placed the result on the first tensor argument's device.
vllm-rbln's speculative-decoding glue relies on that (a CPU cumsum minus an rbln count).
With int32 in the catalog the same expression must neither raise nor change device, and
stock dtypes must keep the result device they had.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401


_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")

_SCENARIO = r"""
import torch, torch_rbln
mask_cpu = torch.tensor([False], dtype=torch.bool)
backup_cpu = torch.tensor([5], dtype=torch.int32)
valid = torch.tensor([2], dtype=torch.int32, device="rbln")
ids = torch.tensor([[7, -1, 9]], dtype=torch.int32, device="rbln")
mask64_cpu = torch.zeros(64, dtype=torch.bool)
a = torch.ones(64, dtype=torch.bfloat16, device="rbln")
outs = [
    torch.where(mask_cpu, torch.zeros_like(valid), valid),  # int32, cpu first
    torch.where(valid > 0, valid, backup_cpu),  # int32, rbln first
    torch.where(mask_cpu, backup_cpu, valid),  # int32, cpu first
    (ids != -1).sum(dim=1).to(torch.int32),  # int32, all rbln
    torch.where(mask64_cpu, a, a * 2),  # bf16 (stock) with a cpu condition
]
torch.rbln.synchronize()
print("RESULT", [(o.device.type, o.cpu().flatten().tolist()) for o in outs])
"""


def _run(**env):
    """Run the scenario in a fresh interpreter: the Python dtype policy is an import-time snapshot."""
    full_env = dict(os.environ)
    full_env.pop("TORCH_RBLN_DISPATCH_DTYPES", None)
    full_env.pop("TORCH_RBLN_DISPATCH_STRICT", None)
    full_env.update(env)
    proc = subprocess.run([sys.executable, "-c", _SCENARIO], env=full_env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    return ast.literal_eval(lines[-1][len("RESULT ") :])


@pytest.mark.test_set_ci
@needs_device
class TestMixedDeviceOperands(TestCase):
    def test_extended_catalog_keeps_stock_device_semantics(self):
        stock = _run()
        self.assertEqual([d for d, _ in stock], ["cpu", "rbln", "cpu", "rbln", "rbln"])
        self.assertEqual([v for _, v in stock[:4]], [[2], [2], [2], [2]])
        for dtypes in ("int32,int16", "bool", "int32,bool"):
            with self.subTest(dtypes=dtypes):
                self.assertEqual(_run(TORCH_RBLN_DISPATCH_DTYPES=dtypes), stock)


if __name__ == "__main__":
    run_tests()
