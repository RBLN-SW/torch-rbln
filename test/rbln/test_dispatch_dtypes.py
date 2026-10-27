# Owner(s): ["module: PrivateUse1"]
"""TORCH_RBLN_DISPATCH_DTYPES: parsing of the extra-dtype list at import, and an
admitted dtype (int32) taking the device path with correct results.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln
from test.utils import run_snippet
from torch_rbln._internal.ops_utils import _dispatch_extension_from_env


_have_device = torch_rbln.device.device_count() > 0 and os.environ.get("RBLN_DUMMY_DEVICE") != "1"
needs_device = pytest.mark.skipif(not _have_device, reason="needs an RBLN device")


_INT32_SCENARIO = r"""
import torch, torch_rbln
from torch_rbln import _C
from torch_rbln._internal import compile_cache
a = (torch.arange(12).reshape(4, 3) * 3).to(torch.int32).to("rbln")
b = torch.ones(4, 3).to(torch.int32).to("rbln")
want = a.cpu() + b.cpu()
_C._dispatch_fallback_by_op_reset()
compiled_before = len(compile_cache._compiled_op_cache)
got = a + b
torch.rbln.synchronize()
assert torch.equal(got.cpu(), want), (got.cpu(), want)
x = torch.randint(0, 9, (4, 3), dtype=torch.int32).to("rbln")  # RNG itself runs on the host
xc = x.cpu()
checks = {
    "mul": (x * x, xc * xc),
    "eq": (x == 3, xc == 3),
    "where": (torch.where(x > 4, x, x - 1), torch.where(xc > 4, xc, xc - 1)),
}
torch.rbln.synchronize()
extra = {k: bool(torch.equal(r.cpu(), ref)) for k, (r, ref) in checks.items()}
# Which ops fell back (every fallback path), and whether the compile cache gained an op.
evidence = {
    "fallbacks": dict(_C._dispatch_fallback_by_op()),
    "compiled": len(compile_cache._compiled_op_cache) - compiled_before,
}
print("RESULT", (got.device.type, extra, evidence))
"""


def _run_int32_scenario(**env) -> tuple[str, dict, dict]:
    """Run the int32 scenario in a fresh interpreter with the env set before import.

    The Python compile path snapshots the dtype policy at import (SupportedDtypes),
    which is also how the knob is meant to be used; a subprocess keeps this test
    independent of what earlier tests imported or patched in this process.
    """
    return run_snippet(
        _INT32_SCENARIO, env=env, unset=("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK", "TORCH_RBLN_DISPATCH_DTYPES")
    )


class TestDispatchDtypesEnv(TestCase):
    def test_env_parsing(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TORCH_RBLN_DISPATCH_DTYPES", raising=False)
            self.assertEqual(_dispatch_extension_from_env(), ())
            # torch names and torch aliases; catalog dtypes and repeats are skipped
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", " int32, float64,float32, int, long ,")
            self.assertEqual(_dispatch_extension_from_env(), (torch.int32, torch.float64, torch.int64))
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "int32, float3")
            with self.assertRaisesRegex(ValueError, "'float3' is not a torch dtype name"):
                _dispatch_extension_from_env()
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "add")  # a torch attribute that is not a dtype
            with self.assertRaisesRegex(ValueError, "'add' is not a torch dtype name"):
                _dispatch_extension_from_env()
            mp.setenv("TORCH_RBLN_DISPATCH_DTYPES", "complex64")
            with self.assertRaisesRegex(ValueError, "torch.complex64 cannot be admitted"):
                _dispatch_extension_from_env()

    def test_bad_dtype_name_fails_at_import(self):
        env = dict(os.environ, TORCH_RBLN_DISPATCH_DTYPES="int32,bad_dtype")
        proc = subprocess.run(
            [sys.executable, "-c", "import torch_rbln"],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("TORCH_RBLN_DISPATCH_DTYPES: 'bad_dtype' is not a torch dtype name", proc.stderr)


@needs_device
class TestExtendedCatalog(TestCase):
    def test_int32_takes_device_path(self):
        # int32 is outside the default catalog: DISPATCH_DTYPES admits it. The alignment
        # fallback is disabled so these glue-shaped [4, 3] ops stay on the device; they
        # must stay correct.
        device, extra, evidence = _run_int32_scenario(
            TORCH_RBLN_DISPATCH_DTYPES="int32", TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK="align"
        )
        self.assertEqual(device, "rbln")
        # where's bool condition next to its data is a mixed-dtype call, which the
        # Python gate sends to the CPU for every dtype; only its result is checked.
        fallbacks = {op: n for op, n in evidence["fallbacks"].items() if op != "aten::where"}
        self.assertEqual(fallbacks, {}, "an int32 op fell back to the CPU")
        self.assertGreater(evidence["compiled"], 0, "no int32 op was compiled for the device")
        self.assertEqual(extra, {"mul": True, "eq": True, "where": True})


if __name__ == "__main__":
    run_tests()
