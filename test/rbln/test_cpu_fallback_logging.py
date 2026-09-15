# Owner(s): ["module: PrivateUse1"]

"""Every CPU fallback must leave a log line.

``AGENTS.md``: a run where a fallback fired is a failed run, and the way to tell
is to grep the log. That only holds if each fallback logs. Two groups did not:

  * the ops the C++ dispatch shim covers (``add``, ``mul``, the reductions, …).
    The shim's own pre-check decides dtype / NaN-Inf / all-scalar fallbacks and
    called ``cpu_fallback_rbln`` directly, skipping the ``log_cpu_fallback`` the
    ``fallback_rbln`` handler makes for an unsupported op. No log at any level.
  * the three ``is_cpu_fallback_cases`` branches in the SDPA kernel, whose
    siblings on the same path do log.

``TORCH_RBLN_LOG_LEVEL`` is latched in a function-local static on first use, so
each scenario runs in its own process with the level already set.
"""

import os
import subprocess
import sys
import textwrap

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FALLBACK_LOG = "op ran on CPU instead of RBLN"


@pytest.mark.test_set_ci
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
                "import torch, torch_rbln  # noqa: F401",
                textwrap.dedent(body).strip(),
                'print("BODY_OK")',
            ]
        )
        return subprocess.run(
            [sys.executable, "-c", script],
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
        values are right and nothing is logged."""
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
        Pins the level: raising it to WARNING would make every fp32 workload noisy."""
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
        """The dtype gate in ``sdpa_overrideable`` passes for float16, so the kernel reaches
        the per-step ``is_cpu_fallback_cases`` checks, which a NaN in the inputs trips. Those
        two branches return into a CPU helper; without the fix neither says so."""
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


if __name__ == "__main__":
    run_tests()
