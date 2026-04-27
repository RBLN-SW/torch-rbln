# Owner(s): ["module: PrivateUse1"]
"""
End-to-end vllm-rbln LLM tests on the native vLLM model path.

Similar in spirit to ``test_optimum_llm.py`` but exercises the native vLLM
model path (``VLLM_RBLN_USE_VLLM_MODEL=1``) on a small matrix of
representative vllm-rbln models as a pre-screen for downstream CI.

Models are restricted to those that run on <=4 NPUs. Sampling is greedy
(``temperature=0``); the first few generated tokens are compared against
hard-coded expected strings.

Environment requirements
------------------------
* ``vllm-rbln`` installed on ``origin/device_tensor_rebased`` (or descendant).
* ``vllm_rbln`` / ``vllm`` importable — the test skips cleanly otherwise.

Matrix
------
* Default: graph mode (``enforce_eager=False``) — the primary compile path.
* One case: eager mode (``enforce_eager=True``) — sanity-check the eager
  execution path.
"""

import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import torch

from test.utils import run_in_isolated_process


# NOTE: do NOT import ``torch.testing._internal.common_utils`` at module level.
# Spawned children re-import this module, and that import path perturbs rebel
# runtime state into ``SYS_ERROR -14`` on NPU submit. See PR #10533.


pytest.importorskip(
    "vllm_rbln",
    reason="vllm-rbln is not installed; install vllm-rbln to run vLLM LLM tests",
)


# Prepended to PYTHONPATH so EngineCore's fresh interpreter can resolve the
# TP>1 qualified-name ``worker_cls``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# TP>1 only — skips stock ``_init_device_env`` which rewrites ``RBLN_DEVICES``
# at runtime and trips rebel's mutation check. See ``_vllm_rbln_worker_patch.py``.
_PATCHED_WORKER_CLS = "test.models._vllm_rbln_worker_patch.PatchedRBLNWorker"


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmModelConfig:
    """One model entry in the vLLM LLM test matrix."""

    model_id: str
    family: str
    # Kept small to fit CI NPU memory — vllm's KV-cache block budget scales
    # with this in ``determine_available_memory``.
    max_model_len: int = 2 * 1024
    block_size: int = 1024
    max_num_batched_tokens: int = 128
    max_num_seqs: int = 1
    trust_remote_code: bool = False
    # Highest TP we will ever attempt for this model. Kept <=4 NPUs.
    max_npus: int = 4
    extra_env: dict[str, str] = field(default_factory=dict)


# Three representative decoder-only LLMs that are already exercised by
# vllm-rbln's own ``test_basic_models_correctness.py`` /
# ``test_model_coverage_single.py`` matrix, picked to cover distinct
# architectures (Llama, Qwen2, Qwen3) while keeping the 1-NPU / <=4-NPU budget.
MODEL_CONFIGS: dict[str, VllmModelConfig] = {
    "llama_3_2_1b": VllmModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
    ),
    "qwen2_5_1_5b": VllmModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen2",
    ),
    "qwen3_0_6b": VllmModelConfig(
        model_id="Qwen/Qwen3-0.6B",
        family="qwen3",
    ),
}


PROMPT = "The capital of France is"
MAX_TOKENS = 5


# Greedy-decode (temperature=0) expected outputs. Key: (model, tp, mode).
# ``None`` falls back to non-empty + shape checks. Captured on RBLN-CA25 with
# rebel-compiler dev329 + vllm-rbln device_tensor_rebased; drift = regression.
EXPECTED_TEXT: dict[tuple[str, int, str], Optional[str]] = {
    ("llama_3_2_1b", 1, "graph"): " Paris. The Eiff",
    ("qwen2_5_1_5b", 1, "graph"): " Paris. The capital of",
    ("qwen3_0_6b", 1, "graph"): " Paris. The capital of",
    ("llama_3_2_1b", 1, "eager"): " Paris. The Eiff",
    ("llama_3_2_1b", 2, "graph"): " Paris. The Eiff",
}


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _vllm_generate_worker(
    model_key: str,
    tp_size: int,
    enforce_eager: bool,
    expected_text: Optional[str],
    prompt: str,
    max_tokens: int,
) -> None:
    """Run a single greedy vLLM generation in a spawned subprocess.

    ``RBLN_DEVICES`` / ``RBLN_NPUS_PER_DEVICE`` must be set in the *parent*
    (``_run_case``) before spawn: rebel's ``librbln-thunk.so`` snapshots them
    at ``import torch``, and spawn re-imports this module before reaching this
    body. Same pattern as ``test_optimum_llm.py::_run_test_case``.
    """
    from vllm import LLM, SamplingParams

    cfg = MODEL_CONFIGS[model_key]
    tc = unittest.TestCase()

    # Always TP=1 on the vLLM engine; RSD fan-out is driven by
    # ``VLLM_RBLN_TP_SIZE`` (set in ``_run_case``), keeping the test
    # single-process and off the RCCL multi-worker path.
    # ``gpu_memory_utilization=0.5`` caps KV-cache blocks for tight CI NPU mem.
    llm_kwargs: dict = dict(
        model=cfg.model_id,
        dtype="float16",
        max_model_len=cfg.max_model_len,
        block_size=cfg.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        tensor_parallel_size=1,
        trust_remote_code=cfg.trust_remote_code,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.5,
    )
    # TP>1 only — see ``_PATCHED_WORKER_CLS`` above.
    if tp_size > 1:
        llm_kwargs["worker_cls"] = _PATCHED_WORKER_CLS

    llm = LLM(**llm_kwargs)

    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens))
    tc.assertEqual(len(outputs), 1, "expected a single RequestOutput")

    gen = outputs[0].outputs[0]
    gen_text = gen.text
    gen_ids = list(gen.token_ids)

    mode = "eager" if enforce_eager else "graph"
    print(f"[vllm_llm_test] model={model_key} tp={tp_size} mode={mode} text={gen_text!r} ids={gen_ids}")

    tc.assertGreater(len(gen_text), 0, "generated text should not be empty")
    tc.assertEqual(len(gen_ids), max_tokens, "greedy decode should fill max_tokens")
    tc.assertTrue(all(isinstance(i, int) and i >= 0 for i in gen_ids))

    if expected_text is not None:
        tc.assertEqual(
            gen_text,
            expected_text,
            (
                f"vLLM RBLN output mismatch for {model_key} tp{tp_size} {mode}. "
                f"Expected: {expected_text!r}, Got: {gen_text!r}"
            ),
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _skip_if_not_enough_npus(tp_size: int) -> None:
    if tp_size <= 1:
        return
    n_phys = torch.rbln.physical_device_count()
    if n_phys < tp_size:
        pytest.skip(f"Requires at least {tp_size} physical devices, found {n_phys}")


def _run_case(model_key: str, tp_size: int, enforce_eager: bool) -> None:
    cfg = MODEL_CONFIGS[model_key]
    if tp_size > cfg.max_npus:
        pytest.skip(f"{model_key} matrix entry restricted to <={cfg.max_npus} NPUs, got tp={tp_size}")
    _skip_if_not_enough_npus(tp_size)

    mode = "eager" if enforce_eager else "graph"
    expected_text = EXPECTED_TEXT.get((model_key, tp_size, mode))

    with pytest.MonkeyPatch.context() as mp:
        # Set RBLN_DEVICES (and RBLN_NPUS_PER_DEVICE for RSD) in the parent
        # before spawn — librbln-thunk snapshots them at ``import torch``,
        # which fires when ``multiprocessing.spawn`` re-imports this module.
        # Setting them inside the child is too late.
        mp.setenv("RBLN_DEVICES", ",".join(str(i) for i in range(tp_size)))
        if tp_size > 1:
            mp.setenv("RBLN_NPUS_PER_DEVICE", str(tp_size))

        # Repo root on PYTHONPATH for EngineCore + triton kernel-compile
        # subprocesses (fresh interpreters): (1) resolve the TP>1 qualified
        # ``worker_cls``, and (2) prefer this checkout's ``torch_rbln`` over
        # any sibling editable install in site-packages.
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            mp.setenv(
                "PYTHONPATH",
                f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}",
            )
        else:
            mp.setenv("PYTHONPATH", str(_REPO_ROOT))

        # Native vLLM model path config — matches the vllm-rbln team's
        # ``test_llama_batch.py`` reference on ``device_tensor_rebased``.
        mp.setenv("VLLM_USE_V1", "1")
        mp.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        mp.setenv("VLLM_RBLN_SAMPLER", "1")
        mp.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
        mp.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        mp.setenv("RBLN_KERNEL_MODE", "triton")
        mp.setenv("RBLN_PV_OPT", "1")
        mp.setenv("TORCH_RBLN_DISABLE_FALLBACK", "compile_error")
        # RSD fan-out (1 worker → ``tp_size`` physical NPUs merged into one
        # logical device). vllm engine stays at ``tensor_parallel_size=1``.
        mp.setenv("VLLM_RBLN_TP_SIZE", str(tp_size))
        if enforce_eager:
            # Eager workarounds: keep sampler + attention backend off the
            # compile path so they stay consistent with the uncompiled model.
            # Without these: RBLN sampler hits weight-reuse assert
            # (OpInvalidWeightSharingError) and FA emits wrong logits.
            # Revisit when rebel tolerates shape-only recompiles under
            # ``use_weight_sharing``.
            mp.setenv("VLLM_RBLN_SAMPLER", "0")
            mp.setenv("VLLM_RBLN_COMPILE_MODEL", "0")
        for key, val in cfg.extra_env.items():
            mp.setenv(key, val)

        run_in_isolated_process(
            _vllm_generate_worker,
            model_key,
            tp_size,
            enforce_eager,
            expected_text,
            PROMPT,
            MAX_TOKENS,
        )


# ---------------------------------------------------------------------------
# Tests — graph mode is the default (enforce_eager=False); eager has one case.
# ---------------------------------------------------------------------------


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.parametrize("model_key", list(MODEL_CONFIGS.keys()))
def test_vllm_llm_graph_tp1(enable_deploy_mode, model_key):
    """Graph mode (torch.compile) TP=1 — primary device-tensor validation."""
    _run_case(model_key=model_key, tp_size=1, enforce_eager=False)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.parametrize("model_key", ["llama_3_2_1b"])
def test_vllm_llm_graph_tp2(enable_deploy_mode, model_key):
    """Graph mode RSD=2 — 1 vLLM worker over 2 physical NPUs via
    ``VLLM_RBLN_TP_SIZE=2``. Skipped if <2 NPUs. Does not exercise the
    RCCL multi-worker collective path (separate axis)."""
    _run_case(model_key=model_key, tp_size=2, enforce_eager=False)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.parametrize("model_key", ["llama_3_2_1b"])
def test_vllm_llm_eager_tp1(enable_deploy_mode, model_key):
    """Eager mode TP=1 — sanity check non-compile path. Greedy decode should
    match graph-mode TP1 (eager workaround envs set in ``_run_case``)."""
    _run_case(model_key=model_key, tp_size=1, enforce_eager=True)


if __name__ == "__main__":
    # See module-level NOTE for why we don't use ``common_utils.run_tests``.
    raise SystemExit("Run this module via pytest: pytest test/models/test_vllm_llm.py")
