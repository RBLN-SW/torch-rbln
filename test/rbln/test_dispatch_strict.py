# Owner(s): ["module: PrivateUse1"]
"""TORCH_RBLN_DISPATCH_STRICT: listed dtypes skip the alignment performance fallbacks and take the
device path as first-class ops; results must match the CPU and the op must not be counted as a
CPU fallback."""

from __future__ import annotations

import os

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln import _C


DEVICE = torch.device("rbln:0")
_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")


_SCENARIO = r"""
import os, sys, torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import compile_cache
from torch_rbln._internal.ops_utils import cpu_fallback_counts
dtype = getattr(torch, sys.argv[1])
a = (torch.arange(12).reshape(4, 3) * 3).to(dtype).to("rbln")
b = torch.ones(4, 3).to(dtype).to("rbln")
want = a.cpu() + b.cpu()
cpu_fallback_counts(reset=True)
_C._dispatch_fallback_by_op_reset()
compiled_before = len(compile_cache._compiled_op_cache)
got = a + b
torch.rbln.synchronize()
assert torch.equal(got.cpu(), want), (got.cpu(), want)
extra = {}
if dtype == torch.int32:
    x = torch.randint(0, 9, (4, 3), dtype=torch.int32, device="rbln")
    xc = x.cpu()
    checks = {
        "mul": (x * x, xc * xc),
        "eq": (x == 3, xc == 3),
        "where": (torch.where(x > 4, x, x - 1), torch.where(xc > 4, xc, xc - 1)),
    }
    torch.rbln.synchronize()
    extra = {k: bool(torch.equal(r.cpu(), ref)) for k, (r, ref) in checks.items()}
# Device-path evidence beyond the Python counter: the C++ boxed fallback (which also
# returns rbln results without touching the Python counter) recorded nothing, and the
# compile cache gained the op.
evidence = {
    "cxx_fallbacks": _C._dispatch_fallback_by_op(),
    "compiled": len(compile_cache._compiled_op_cache) - compiled_before,
}
print("RESULT", cpu_fallback_counts().get("aten::add", 0), got.device.type, (extra, evidence))
"""

_NAN_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
x = torch.arange(64, dtype=torch.float32).reshape(1, 64).to("rbln")
for _ in range(3):  # finite inputs: compile, then warm-cache hits
    y = x + 1
torch.rbln.synchronize()
bad = x.cpu()
bad[0, 5] = float("nan")
bad[0, 7] = float("inf")
bad = bad.to("rbln")
before = _C._dispatch_fallback_reasons()
z = bad + 1
torch.rbln.synchronize()
after = _C._dispatch_fallback_reasons()
zc = z.cpu()
print("RESULT", after[1] - before[1], bool(torch.isnan(zc[0, 5])), bool(torch.isinf(zc[0, 7])))
"""


def _run_scenario(dtype_name: str, **env) -> tuple[int, str, dict]:
    """Run the unaligned-add scenario in a fresh interpreter with the env set before import.

    The Python compile path snapshots the dtype policy at import (SupportedDtypes), which
    is also how the knob is meant to be used; a subprocess keeps this test independent of
    what earlier tests imported or patched in this process."""
    import ast
    import subprocess
    import sys

    full_env = dict(os.environ)
    full_env.pop("TORCH_RBLN_DISPATCH_STRICT", None)
    full_env.pop("TORCH_RBLN_DISPATCH_DTYPES", None)
    full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", _SCENARIO, dtype_name], env=full_env, capture_output=True, text=True, timeout=600
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    _, count, device, extra = lines[-1].split(" ", 3)
    extra, evidence = ast.literal_eval(extra)
    return int(count), device, extra, evidence


def _run_nan_scenario(**env) -> tuple[int, bool, bool]:
    import ast
    import subprocess
    import sys

    full_env = dict(os.environ)
    full_env.pop("TORCH_RBLN_DISPATCH_STRICT", None)
    full_env.pop("TORCH_RBLN_DISPATCH_DTYPES", None)
    full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", _NAN_SCENARIO], env=full_env, capture_output=True, text=True, timeout=600
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    _, hits, is_nan, is_inf = lines[-1].split(" ")
    return int(hits), ast.literal_eval(is_nan), ast.literal_eval(is_inf)


@pytest.mark.test_set_ci
class TestStrictCatalog(TestCase):
    def test_env_parsing(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TORCH_RBLN_DISPATCH_STRICT", raising=False)
            self.assertEqual(_C._dispatch_strict_dtypes(), ())
            mp.setenv("TORCH_RBLN_DISPATCH_STRICT", "int32, float32")
            self.assertEqual(set(_C._dispatch_strict_dtypes()), {torch.int32, torch.float32})
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "int16")
            mp.setenv("TORCH_RBLN_DISPATCH_STRICT", "all")
            self.assertEqual(set(_C._dispatch_strict_dtypes()), {torch.float16, torch.bfloat16, torch.int16})
            mp.setenv("TORCH_RBLN_DISPATCH_STRICT", "int32, float3")
            with self.assertRaisesRegex(RuntimeError, "unknown dtype name 'float3'"):
                _C._dispatch_strict_dtypes()
            mp.setenv("TORCH_RBLN_DISPATCH_STRICT", "int32")
            self.assertEqual(_C._dispatch_strict_dtypes(), (torch.int32,))  # the bad value left no partial state


@pytest.mark.test_set_ci
@needs_device
class TestStrictDispatch(TestCase):
    def test_default_policy_keeps_unaligned_bf16_on_host(self):
        count, _, _, _ = _run_scenario("bfloat16")
        self.assertGreater(count, 0)

    def test_strict_bf16_takes_device_path(self):
        count, device, _, evidence = _run_scenario("bfloat16", TORCH_RBLN_DISPATCH_STRICT="bfloat16")
        self.assertEqual(count, 0)
        self.assertEqual(device, "rbln")
        self.assertEqual(evidence["cxx_fallbacks"], [])
        self.assertGreater(evidence["compiled"], 0)

    def test_strict_int32_with_extended_catalog(self):
        # int32 is outside the default catalog: DISPATCH_DTYPES admits it, STRICT keeps it on the
        # device even at [4, 3]; a few more glue-shaped int32 ops must stay correct too.
        count, device, extra, evidence = _run_scenario(
            "int32", TORCH_RBLN_DISPATCH_DTYPES="int32", TORCH_RBLN_DISPATCH_STRICT="int32"
        )
        self.assertEqual(count, 0)
        self.assertEqual(device, "rbln")
        self.assertEqual(evidence["cxx_fallbacks"], [], "the C++ boxed fallback ran an int32 op")
        self.assertGreater(evidence["compiled"], 0, "no int32 op was compiled for the device")
        self.assertEqual(extra, {"mul": True, "eq": True, "where": True})

    def test_extended_float32_nan_inf_is_still_caught_on_warm_cache(self):
        # float32 admitted by DISPATCH_DTYPES: after finite calls warmed the cache, a NaN/Inf
        # input must still be caught by the C++ scan (reason 1) and land on the host path,
        # where NaN/Inf propagate as on the CPU.
        hits, is_nan, is_inf = _run_nan_scenario(TORCH_RBLN_DISPATCH_DTYPES="float32")
        self.assertGreaterEqual(hits, 1, "float32 NaN/Inf input was not caught by the shim scan")
        self.assertTrue(is_nan and is_inf)


if __name__ == "__main__":
    run_tests()
