# Owner(s): ["module: PrivateUse1"]
"""Unaligned last dims fall back to the CPU by default, except for ops the compiler
pads itself; TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=align sends them to the device
instead. Results must match the CPU either way.
"""

from __future__ import annotations

import pytest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase

from test.utils import run_snippet


_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import compile_cache
a = (torch.arange(12).reshape(4, 3) * 3).to(torch.bfloat16).to("rbln")
b = torch.ones(4, 3).to(torch.bfloat16).to("rbln")
want = torch.logical_not(a.cpu())
_C._dispatch_fallback_by_op_reset()
compiled_before = len(compile_cache._compiled_op_cache)
got = torch.logical_not(a)
torch.rbln.synchronize()
assert torch.equal(got.cpu(), want), (got.cpu(), want)
# Which ops fell back (every fallback path), and whether the compile cache gained an op.
evidence = {
    "fallbacks": dict(_C._dispatch_fallback_by_op()),
    "compiled": len(compile_cache._compiled_op_cache) - compiled_before,
}
print("RESULT", (got.device.type, evidence))
"""


# Disable the alignment fallback, let a warm-cache entry serve an unaligned add, then
# remove the setting: the cached entry must not keep the call on the device. The
# exemption list is emptied so add still meets the alignment gate.
_RESTORE_SCENARIO = r"""
import os, torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import ops_utils
ops_utils._ALIGN_EXEMPT_OPS = frozenset()
a = torch.full((4, 3), 2.0, dtype=torch.bfloat16).to("rbln")
b = torch.full((4, 3), 3.0, dtype=torch.bfloat16).to("rbln")
out = torch.empty_like(a)
_C._warmcache_clear()
os.environ["TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK"] = "align"
torch.add(a, b, out=out)
hits = _C._dispatch_shim_diag_dump()[2]
torch.add(a, b, out=out)
torch.rbln.synchronize()
warm_hit = _C._dispatch_shim_diag_dump()[2] > hits
os.environ.pop("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK")
_C._dispatch_fallback_by_op_reset()
torch.add(a, b, out=out)
torch.rbln.synchronize()
print("RESULT", (warm_hit, dict(_C._dispatch_fallback_by_op()), out.cpu().tolist()))
"""


_EXEMPT_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
cpu = {
    "a": torch.randn(5, 100) * 0.1,
    "b": torch.rand(5, 100) + 0.5,
    "w": torch.randn(100, 100) * 0.1,
    "bias": torch.randn(100) * 0.1,
    "p": torch.rand(5, 100) + 0.1,
}
cpu["a3"], cpu["w3"] = cpu["a"].expand(2, 5, 100), cpu["w"].expand(2, 100, 100)
dev = {k: v.to(torch.bfloat16).contiguous().to("rbln") for k, v in cpu.items()}
fn = lambda a, b, w, bias, a3, w3, p: EXPR
_C._dispatch_fallback_by_op_reset()
got = fn(**dev)
torch.rbln.synchronize()
want = fn(**{k: v.to(torch.bfloat16).float() for k, v in cpu.items()})
if want.dtype == torch.bool:
    close = torch.equal(got.cpu(), want)
else:
    close = torch.allclose(got.cpu().float(), want, atol=5e-2, rtol=5e-2)
print("RESULT", (got.device.type, dict(_C._dispatch_fallback_by_op()), close))
"""


def _run_scenario(**env) -> tuple[str, dict]:
    """Run the unaligned bf16 logical_not in a fresh interpreter with the env set, so
    the result does not depend on what earlier tests imported or patched in this
    process.
    """
    return run_snippet(_SCENARIO, env=env, unset=("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK",))


@pytest.mark.npu(count=1)
class TestAlignFallback(TestCase):
    def test_unaligned_bf16_falls_back_by_default(self):
        _, evidence = _run_scenario()
        self.assertGreater(evidence["fallbacks"].get("aten::logical_not.out", 0), 0)

    def test_disabled_align_fallback_takes_device_path(self):
        device, evidence = _run_scenario(TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK="align")
        self.assertEqual(device, "rbln")
        self.assertEqual(evidence["fallbacks"], {})
        self.assertGreater(evidence["compiled"], 0)

    def test_restoring_the_default_reaches_a_cached_entry(self):
        warm_hit, fallbacks, values = run_snippet(_RESTORE_SCENARIO, unset=("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK",))
        self.assertTrue(warm_hit, "no warm hit while align was disabled; the case is not exercised")
        self.assertGreater(fallbacks.get("aten::add.out", 0), 0, "the cached entry bypassed the alignment fallback")
        self.assertEqual(values, [[5.0] * 3] * 4)

    @parametrize(
        "expr",
        [
            "a + b",
            "a - b",
            "a * b",
            "a / b",
            "torch.pow(a, 2)",
            "torch.neg(a)",
            "torch.abs(a)",
            "torch.ceil(a)",
            "torch.floor(a)",
            "torch.clamp(a, -0.1, 0.1)",
            "torch.log(p)",
            "torch.rsqrt(p)",
            "torch.sigmoid(a)",
            "torch.nn.functional.silu(a)",
            "torch.eq(a, b)",
            "torch.ne(a, b)",
            "torch.lt(a, b)",
            "torch.le(a, b)",
            "torch.gt(a, b)",
            "torch.ge(a, b)",
            "torch.maximum(a, b)",
            "torch.minimum(a, b)",
            "torch.max(a)",
            "torch.min(a)",
            "torch.mean(a)",
            "torch.mm(a, w)",
            "torch.bmm(a3, w3)",
            "torch.addmm(bias, a, w)",
            "torch.nn.functional.linear(a, w, bias)",
            "torch.softmax(a, -1)",
        ],
        name_fn=lambda e: e.split("(")[0].split(".")[-1]
        if "(" in e
        else {"+": "add", "-": "sub", "*": "mul", "/": "div"}[e[2]],
    )
    def test_unaligned_exempt_op_stays_on_device_by_default(self, expr):
        device, fallbacks, close = run_snippet(
            _EXEMPT_SCENARIO.replace("EXPR", expr), unset=("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK",)
        )
        self.assertEqual(device, "rbln")
        self.assertEqual(fallbacks, {})
        self.assertTrue(close)


instantiate_parametrized_tests(TestAlignFallback)


if __name__ == "__main__":
    run_tests()
