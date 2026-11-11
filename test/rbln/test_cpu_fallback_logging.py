# Owner(s): ["module: PrivateUse1"]

"""Every CPU fallback must leave a log line and record its reason.

``AGENTS.md``: a run where a fallback fired is a failed run, and the way to tell
is to grep the log. That only holds if each fallback logs. Two groups did not:

  * the ops the C++ dispatch shim covers (``add``, ``mul``, the reductions, …).
    The shim's own pre-check decides dtype / NaN-Inf / all-scalar fallbacks and
    called ``cpu_fallback_rbln`` directly, skipping the log the
    ``fallback_rbln`` handler made for an unsupported op. No log at any level.
  * the three ``get_fallback_reason`` branches in the SDPA kernel, whose
    siblings on the same path do log.

``TORCH_RBLN_LOG_LEVEL`` is latched in a function-local static on first use, so
each scenario runs in its own process with the level already set.

The second half: a fallback is counted under the reason that decided it, which
is what the per-test fallback report and ``torch.rbln.explain()`` read. One
scenario drives each fallback path once and checks the reason it records.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import run_snippet
from torch_rbln import _C
from torch_rbln._internal.log_utils import rbln_record_cpu_fallback
from torch_rbln._internal.ops_utils import _ALL_FALLBACK_CASES


# Deselected by rebel_compiler's CI (`-m "not torch_rbln_only"`); see the marker in
# pyproject. Whether a fallback logs and which reason it records is torch-rbln's own
# logic: the fallbacks here are decided before anything is compiled.
pytestmark = pytest.mark.torch_rbln_only


_PROJECT_ROOT = str(Path(__file__).absolute().parents[2])
_FALLBACK_LOG = "op ran on CPU instead of RBLN"


@pytest.mark.npu(count=1)
class TestCpuFallbackIsLogged(TestCase):
    """A fallback the shim decides in C++ logs the same line the Python path logs."""

    def _run(self, body: str, log_level: str = "INFO") -> subprocess.CompletedProcess:
        # Assembled at column zero rather than dedented from a template: ``body`` is
        # multi-line, and its lines after the first share no indent for dedent to strip.
        script = "\n".join(
            [
                "import os, sys",
                f"sys.path.insert(0, {_PROJECT_ROOT!r})",
                f'os.environ["TORCH_RBLN_LOG_LEVEL"] = {log_level!r}',
                "import torch, torch_rbln",
                textwrap.dedent(body).strip(),
                'print("BODY_OK")',
            ]
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _assert_ran(self, p: subprocess.CompletedProcess) -> str:
        # A probe that died before the body says nothing about logging, and an absence
        # assertion would pass on its empty output.
        self.assertIn(
            "BODY_OK",
            p.stdout,
            f"the probe did not finish\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}",
        )
        # Both streams: an absence assertion against stdout alone would not see a line
        # that went to stderr.
        return p.stdout + p.stderr

    def test_shim_op_dtype_fallback_is_logged(self):
        """fp32 is outside the dispatch dtype catalog, so ``add`` falls back in the
        shim's pre-check, before Python is reached. Fails without the fix: the
        values are right and nothing is logged.
        """
        p = self._run(
            """
            a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="rbln")
            b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device="rbln")
            assert (a + b).cpu().tolist() == [5.0, 7.0, 9.0]
            """
        )
        out = self._assert_ran(p)
        self.assertIn(_FALLBACK_LOG, out, f"an fp32 add fell back to CPU with no log\n{out}")
        self.assertIn("aten::add", out, f"the log did not name the op that fell back\n{out}")

    def test_fallback_log_stays_off_by_default(self):
        """The line is INFO, so the default (WARNING) keeps it out of a user's console.
        Pins the level: raising it to WARNING would make every fp32 workload noisy.
        """
        p = self._run(
            """
            a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="rbln")
            assert (a + a).cpu().tolist() == [2.0, 4.0, 6.0]
            """,
            log_level="WARNING",
        )
        out = self._assert_ran(p)
        self.assertNotIn(_FALLBACK_LOG, out, f"the fallback line escaped a WARNING-level run\n{out}")

    def test_sdpa_inner_fallback_is_logged(self):
        """The dtype gate in ``sdpa_overrideable`` passes for float16, so the kernel
        reaches the per-step ``get_fallback_reason`` checks, which a NaN in the inputs
        trips. Those two branches return into a CPU helper; without the fix neither says
        so.
        """
        p = self._run(
            """
            q = torch.randn(1, 2, 64, 64, dtype=torch.float16, device="rbln")
            q[0, 0, 0, 0] = float("nan")
            torch.nn.functional.scaled_dot_product_attention(q, q, q).cpu()
            """
        )
        out = self._assert_ran(p)
        self.assertIn("sdpa (attn_weights)", out, f"the attention-weights fallback said nothing\n{out}")
        self.assertIn("sdpa (output)", out, f"the output fallback said nothing\n{out}")

    def test_sdpa_backward_fallback_is_logged(self):
        """Same for the backward: a NaN in the incoming gradient trips its own check."""
        p = self._run(
            """
            shape = (1, 2, 64, 64)
            q = torch.randn(*shape, dtype=torch.float16, device="rbln", requires_grad=True)
            k = torch.randn(*shape, dtype=torch.float16, device="rbln", requires_grad=True)
            v = torch.randn(*shape, dtype=torch.float16, device="rbln", requires_grad=True)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            grad = torch.randn_like(out)
            grad[0, 0, 0, 0] = float("nan")
            out.backward(grad)
            """
        )
        out = self._assert_ran(p)
        self.assertIn("sdpa_backward", out, f"the backward fallback said nothing\n{out}")


_FALLBACK_REASON_SCENARIO = r"""
import torch, torch_rbln
from torch._dynamo.backends import registry
from torch_rbln._internal.monkey_patches import _register_rbln_backend
offset = torch.ones(65, dtype=torch.bfloat16).to("rbln")[1:]
unaligned = torch.ones(4, 3, dtype=torch.bfloat16).to("rbln")
scalar = torch.tensor(2.0, dtype=torch.bfloat16).to("rbln")
aligned = torch.ones(64, dtype=torch.bfloat16).to("rbln")

def compile_error_retry():
    _register_rbln_backend()
    def fail(gm, example_inputs, **kwargs):
        raise RuntimeError("forced compile failure")
    registry._COMPILER_FNS["rbln"] = fail
    torch.neg(aligned)

cases = {
    "wrapper_check": lambda: offset + offset,
    "device_entry_check": lambda: unaligned + unaligned,
    "shim_precheck": lambda: scalar * scalar,
    "registered_op": lambda: aligned.flip(0),
    "unregistered_op": lambda: torch.lgamma(aligned),
    # Last: it replaces the rbln compiler for the rest of the process.
    "compile_error_retry": compile_error_retry,
}
reasons = {}
for name, case in cases.items():
    with torch.rbln.explain() as p:
        case()
    reasons[name] = p.dump()["cpu_fallback_reasons_by_op"]
print("RESULT", reasons)
"""


@pytest.mark.npu(count=1)
class TestCpuFallbackRecordsItsReason(TestCase):
    """Every fallback path records the reason that decided it, under the op."""

    def test_each_fallback_path_records_its_reason(self):
        # A fresh interpreter gets the default fallback settings: some are read at
        # import, and this suite's conftest disables the compile_error fallback.
        reasons = run_snippet(
            _FALLBACK_REASON_SCENARIO,
            unset=(
                "TORCH_RBLN_DISABLE_FALLBACK",
                "TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK",
                "TORCH_RBLN_DISPATCH_DTYPES",
            ),
        )
        self.assertEqual(
            reasons,
            {
                "wrapper_check": {"aten::add.out": {"storage_offset": 1}},
                "device_entry_check": {"aten::add.out": {"align": 1}},
                "shim_precheck": {"aten::mul.out": {"scalar": 1}},
                "registered_op": {"aten::flip": {"registered": 1}},
                "unregistered_op": {"aten::lgamma.out": {"unsupported_op": 1}},
                "compile_error_retry": {"aten::neg.out": {"compile_error": 1}},
            },
        )


class TestCpuFallbackReasonVocabulary(TestCase):
    """The reason names are one list, ``c10::rbln::CpuFallbackReason``. The Python
    fallback paths pass a name and the binding parses it, so the list is also what the
    `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` cases and their documentation must match.
    """

    def test_unknown_reason_is_rejected_at_the_call(self):
        # A typo must not count a fallback under a name nothing else recognizes.
        with self.assertRaisesRegex(ValueError, "unknown CPU fallback reason 'typo'"):
            rbln_record_cpu_fallback("aten::add", "typo")

    def test_every_disable_case_is_a_reason(self):
        names = set(_C._cpu_fallback_reason_names())
        self.assertEqual(_ALL_FALLBACK_CASES - names, set(), "a disable case the record cannot count under")

    def test_configuration_doc_lists_the_disable_cases(self):
        # docs/CONFIGURATION.md is the user's copy of the case list; the rows of its
        # table are the cases, plus `all`.
        doc = (Path(_PROJECT_ROOT) / "docs" / "CONFIGURATION.md").read_text()
        section = doc.split("comma-separated list** of fallback case names", 1)[1].split("\n## ", 1)[0]
        documented = {line.split("`")[1] for line in section.splitlines() if line.startswith("| `")}
        self.assertEqual(documented - {"all"}, set(_ALL_FALLBACK_CASES))


if __name__ == "__main__":
    run_tests()
