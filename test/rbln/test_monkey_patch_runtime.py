# Owner(s): ["module: PrivateUse1"]

"""Tests for ``patch_dynamo_runtime`` in ``torch_rbln._internal.monkey_patches``.

Verifies idempotence, deploy-mode validity skip, and that the patched runtime
preserves end-to-end correctness vs CPU reference.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch_rbln  # noqa: F401  -- registers backend, applies patches
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_rbln._internal import monkey_patches as mp


@pytest.mark.test_set_ci
class TestPatchDynamoRuntime(TestCase):
    """Patch is applied at import-time. We verify the resulting state."""

    def test_patch_marker_set(self):
        self.assertTrue(mp._dynamo_runtime_patched)

    def test_patch_idempotent(self):
        # Calling apply_all_patches again should not raise or rebind.
        mp.apply_all_patches()
        mp.apply_all_patches()
        self.assertTrue(mp._dynamo_runtime_patched)

    def test_patched_run_attached(self):
        from rebel.sync_runtime import DynamoRuntime
        # The patched function name we wrote is ``patched_run``.
        self.assertEqual(DynamoRuntime.run.__name__, "patched_run")


@pytest.mark.test_set_ci
class TestPatchedRunCorrectness(TestCase):
    """End-to-end: simple eager add via the patched runtime should match CPU."""

    rbln_device = torch.device("rbln:0")

    def test_add_matches_cpu(self):
        a = torch.full((1024,), 1.0, dtype=torch.float16, device=self.rbln_device)
        b = torch.full((1024,), 2.0, dtype=torch.float16, device=self.rbln_device)
        r = (a + b).cpu()
        self.assertEqual(r.tolist(), [3.0] * 1024)

    def test_compiled_module_matches_cpu(self):
        class Add(nn.Module):
            def forward(self, x, y):
                return x + y

        a = torch.full((1024,), 1.5, dtype=torch.float16, device=self.rbln_device)
        b = torch.full((1024,), 2.5, dtype=torch.float16, device=self.rbln_device)
        fn = torch.compile(Add().eval(), backend="rbln", dynamic=False, options={"tensor_parallel_size": 1})
        r = fn(a, b).cpu()
        self.assertEqual(r.tolist(), [4.0] * 1024)


@pytest.mark.test_set_ci
class TestSkipValidityToggle(TestCase):
    """``TORCH_RBLN_SKIP_RUNTIME_VALIDITY=1`` should not change correctness.

    The patched ``run()`` checks the env var on every call, so toggling it
    mid-session must not affect functional behavior — only validation cost.
    """

    rbln_device = torch.device("rbln:0")

    def _run_once(self):
        a = torch.full((128,), 1.0, dtype=torch.float16, device=self.rbln_device)
        b = torch.full((128,), 2.0, dtype=torch.float16, device=self.rbln_device)
        return (a + b).cpu()

    def test_validity_skip_correct(self):
        prior = os.environ.get("TORCH_RBLN_SKIP_RUNTIME_VALIDITY")
        try:
            os.environ["TORCH_RBLN_SKIP_RUNTIME_VALIDITY"] = "1"
            r1 = self._run_once()
            os.environ.pop("TORCH_RBLN_SKIP_RUNTIME_VALIDITY", None)
            r2 = self._run_once()
            self.assertEqual(r1.tolist(), r2.tolist())
            self.assertEqual(r1.tolist(), [3.0] * 128)
        finally:
            if prior is None:
                os.environ.pop("TORCH_RBLN_SKIP_RUNTIME_VALIDITY", None)
            else:
                os.environ["TORCH_RBLN_SKIP_RUNTIME_VALIDITY"] = prior


if __name__ == "__main__":
    run_tests()
