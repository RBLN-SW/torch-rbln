# Owner(s): ["module: PrivateUse1"]
"""Unaligned last dims fall back to the CPU by default; TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=align
sends them to the device instead. Results must match the CPU either way."""

from __future__ import annotations

import os

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401


_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")


_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import compile_cache
from torch_rbln._internal.ops_utils import cpu_fallback_counts
a = (torch.arange(12).reshape(4, 3) * 3).to(torch.bfloat16).to("rbln")
b = torch.ones(4, 3).to(torch.bfloat16).to("rbln")
want = a.cpu() + b.cpu()
cpu_fallback_counts(reset=True)
_C._dispatch_fallback_by_op_reset()
compiled_before = len(compile_cache._compiled_op_cache)
got = a + b
torch.rbln.synchronize()
assert torch.equal(got.cpu(), want), (got.cpu(), want)
# Device-path evidence beyond the Python counter: the C++ boxed fallback (which also
# returns rbln results without touching the Python counter) recorded nothing, and the
# compile cache gained the op.
evidence = {
    "cxx_fallbacks": _C._dispatch_fallback_by_op(),
    "compiled": len(compile_cache._compiled_op_cache) - compiled_before,
}
print("RESULT", cpu_fallback_counts().get("aten::add", 0), got.device.type, evidence)
"""


def _run_scenario(**env) -> tuple[int, str, dict]:
    """Run the unaligned bf16 add in a fresh interpreter with the env set, so the result does not
    depend on what earlier tests imported or patched in this process."""
    import ast
    import subprocess
    import sys

    full_env = dict(os.environ)
    full_env.pop("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", None)
    full_env.update(env)
    proc = subprocess.run([sys.executable, "-c", _SCENARIO], env=full_env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    _, count, device, evidence = lines[-1].split(" ", 3)
    return int(count), device, ast.literal_eval(evidence)


@needs_device
class TestAlignFallback(TestCase):
    def test_unaligned_bf16_falls_back_by_default(self):
        count, _, _ = _run_scenario()
        self.assertGreater(count, 0)

    def test_disabled_align_fallback_takes_device_path(self):
        count, device, evidence = _run_scenario(TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK="align")
        self.assertEqual(count, 0)
        self.assertEqual(device, "rbln")
        self.assertEqual(evidence["cxx_fallbacks"], [])
        self.assertGreater(evidence["compiled"], 0)


if __name__ == "__main__":
    run_tests()
