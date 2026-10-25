# Owner(s): ["module: PrivateUse1"]
"""TORCH_RBLN_DISPATCH_DTYPES: parsing of the extra-dtype list, and an admitted dtype (int32)
taking the device path with correct results."""

from __future__ import annotations

import os

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401
from torch_rbln import _C


_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")


_INT32_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import compile_cache
from torch_rbln._internal.ops_utils import cpu_fallback_counts
a = (torch.arange(12).reshape(4, 3) * 3).to(torch.int32).to("rbln")
b = torch.ones(4, 3).to(torch.int32).to("rbln")
want = a.cpu() + b.cpu()
cpu_fallback_counts(reset=True)
_C._dispatch_fallback_by_op_reset()
compiled_before = len(compile_cache._compiled_op_cache)
got = a + b
torch.rbln.synchronize()
assert torch.equal(got.cpu(), want), (got.cpu(), want)
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

_BAD_ENV_SCENARIO = r"""
import os, torch, torch_rbln
a = torch.arange(64, dtype=torch.int32).reshape(1, 64).to("rbln")
b = a + 1  # int32 admitted: compiles
torch.rbln.synchronize()
os.environ["TORCH_RBLN_DISPATCH_DTYPES"] = "int32,bad_dtype"  # the shim re-reads it on the next op
try:
    c = a + 1
    torch.rbln.synchronize()
    print("RESULT no_error")
except RuntimeError as e:
    print("RESULT", "bad_dtype" in str(e))
"""


def _run_int32_scenario(**env) -> tuple[int, str, dict, dict]:
    """Run the int32 scenario in a fresh interpreter with the env set before import.

    The Python compile path snapshots the dtype policy at import (SupportedDtypes), which
    is also how the knob is meant to be used; a subprocess keeps this test independent of
    what earlier tests imported or patched in this process."""
    import ast
    import subprocess
    import sys

    full_env = dict(os.environ)
    full_env.pop("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", None)
    full_env.pop("TORCH_RBLN_DISPATCH_DTYPES", None)
    full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", _INT32_SCENARIO], env=full_env, capture_output=True, text=True, timeout=600
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-2000:]
    _, count, device, extra = lines[-1].split(" ", 3)
    extra, evidence = ast.literal_eval(extra)
    return int(count), device, extra, evidence


class TestDispatchDtypesEnv(TestCase):
    def test_env_parsing(self):
        stock = (torch.float16, torch.bfloat16, torch.float32)
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TORCH_RBLN_DISPATCH_DTYPES", raising=False)
            self.assertEqual(_C._dispatch_dtypes(), stock)
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "int32, Float64, fp32")
            self.assertEqual(_C._dispatch_dtypes(), stock + (torch.int32, torch.float64))
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "int32, float3")
            with self.assertRaisesRegex(RuntimeError, "unknown dtype name 'float3'"):
                _C._dispatch_dtypes()
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "all")
            with self.assertRaisesRegex(RuntimeError, "unknown dtype name 'all'"):
                _C._dispatch_dtypes()
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "int32")
            self.assertEqual(_C._dispatch_dtypes(), stock + (torch.int32,))  # the bad value left no partial state


@needs_device
class TestExtendedCatalog(TestCase):
    def test_int32_takes_device_path(self):
        # int32 is outside the default catalog: DISPATCH_DTYPES admits it. The alignment fallback
        # is disabled so these glue-shaped [4, 3] ops stay on the device; they must stay correct.
        count, device, extra, evidence = _run_int32_scenario(
            TORCH_RBLN_DISPATCH_DTYPES="int32", TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK="align"
        )
        self.assertEqual(count, 0)
        self.assertEqual(device, "rbln")
        self.assertEqual(evidence["cxx_fallbacks"], [], "the C++ boxed fallback ran an int32 op")
        self.assertGreater(evidence["compiled"], 0, "no int32 op was compiled for the device")
        self.assertEqual(extra, {"mul": True, "eq": True, "where": True})

    def test_bad_dtype_name_after_import_raises_instead_of_terminating(self):
        # the live parser throws on an unknown name; the getters it runs under must not be
        # noexcept, or the throw terminates the process instead of reaching Python
        import subprocess
        import sys

        env = dict(os.environ, TORCH_RBLN_DISPATCH_DTYPES="int32")
        proc = subprocess.run(
            [sys.executable, "-c", _BAD_ENV_SCENARIO], env=env, capture_output=True, text=True, timeout=600
        )
        self.assertEqual(proc.returncode, 0, f"process died instead of raising: {proc.stderr[-1500:]}")
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertEqual(lines[-1], "RESULT True", proc.stdout[-500:])


if __name__ == "__main__":
    run_tests()
