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


# Probe the *default* selection logic in a fresh interpreter: importing torch_rbln
# runs torch_backends_entry_point(), which sets RBLN_WEIGHT_FREE when unset. A
# subprocess is required because the entry point is once-guarded, so the unset path
# can only be exercised by a fresh process. Markers are unique strings so log lines
# (which may themselves contain "RBLN_WEIGHT_FREE=...") cannot pollute parsing.
_DEFAULT_PROBE = textwrap.dedent(
    """
    import os
    import torch_rbln  # noqa: F401  -- torch_backends_entry_point() runs on import

    print("__WF_ENV__=" + os.environ.get("RBLN_WEIGHT_FREE", "<unset>"))
    try:
        import rebel

        if hasattr(rebel, "get_weight_free_mode"):
            print("__WF_MODE__=" + str(rebel.get_weight_free_mode()))
    except Exception:
        pass
    """
)


def _probe_weight_free_default(env_override):
    """Run the probe with RBLN_WEIGHT_FREE forced to ``env_override`` (or removed when
    it is not provided), returning the parsed ``__WF_ENV__`` / ``__WF_MODE__`` values."""
    env = {k: v for k, v in os.environ.items() if k != "RBLN_WEIGHT_FREE"}
    env.update(env_override)
    result = subprocess.run(
        [sys.executable, "-c", _DEFAULT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"weight-free default probe failed (returncode={result.returncode}).\nSTDERR (tail):\n{result.stderr[-2000:]}"
    )
    out = {}
    for line in result.stdout.splitlines():
        for key in ("__WF_ENV__", "__WF_MODE__"):
            prefix = key + "="
            if line.startswith(prefix):
                out[key] = line[len(prefix) :]
    return out


@pytest.mark.test_set_ci
def test_weight_free_default_on_when_unset():
    """With RBLN_WEIGHT_FREE unset, importing torch_rbln must default it to ``on``
    (the torch-like behavior). The optional MODE assertion guards that rebel actually
    observes the value the entry point set -- i.e. it is read lazily at compile time,
    not snapshotted before our default is applied. This breaks in CI (not in prod) if
    a future rebel build starts caching the value earlier."""
    out = _probe_weight_free_default({})
    assert out.get("__WF_ENV__") == "on"
    if "__WF_MODE__" in out:  # only on rebel builds exposing the query API
        assert out["__WF_MODE__"] == "on"


@pytest.mark.test_set_ci
def test_weight_free_user_off_wins():
    """An explicit ``RBLN_WEIGHT_FREE=off`` must win over the default."""
    out = _probe_weight_free_default({"RBLN_WEIGHT_FREE": "off"})
    assert out.get("__WF_ENV__") == "off"
    if "__WF_MODE__" in out:
        assert out["__WF_MODE__"] == "off"


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
