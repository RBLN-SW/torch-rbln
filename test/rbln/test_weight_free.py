# Owner(s): ["module: PrivateUse1"]

"""Regression tests for weight-free (``RBLN_WEIGHT_FREE``) compilation."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.test_set_ci
def test_weight_free_host_const_param_clean_teardown():
    """A weight-free graph with host const params (a ``Linear`` bias) must exit
    cleanly: the borrowed host weight must not be double-freed at teardown. Run in a
    subprocess and checked via exit code (the crash happened at shutdown).
    """
    script = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import torch_rbln  # noqa: F401

        dev = torch.device("rbln:0")
        model = nn.Linear(64, 32).to(dev, dtype=torch.float16)  # bias => host const param
        x = torch.randn(4, 64, device=dev, dtype=torch.float16)
        out = torch.compile(model, backend="rbln")(x)
        assert tuple(out.shape) == (4, 32)
        """
    )
    env = {**os.environ, "RBLN_WEIGHT_FREE": "on", "RBLN_DEVICES": "0"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"weight-free teardown crashed (returncode={result.returncode}).\nSTDERR (tail):\n{result.stderr[-3000:]}"
    )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
