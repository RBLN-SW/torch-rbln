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

The test runs with ``VLLM_RBLN_USE_DEVICE_TENSOR`` *unset* (i.e. the default
legacy KV cache path). The opt-in ``VLLM_RBLN_USE_DEVICE_TENSOR=1`` flow
introduced by the ``a1d4d86`` commit is currently unstable (rebel runtime
reports ``Buffer not found for DramTensor`` and submit fails with
``SYS_ERROR -14``); it is excluded from this pre-screen until that
end-to-end flow lands as a supported configuration upstream. The remaining
env matches the vllm-rbln ``test_llama_batch.py`` reference configuration
announced to run green on ``device_tensor_rebased``.

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


# NOTE: intentionally do NOT `from torch.testing._internal.common_utils import
# run_tests` at module level. Importing ``common_utils`` pulls in enough of
# torch's test-internal machinery to perturb the rebel runtime state of any
# child process that later re-imports this module; the effect is that
# ``multiprocessing.spawn`` children dispatched via ``run_in_isolated_process``
# end up failing rebel's NPU submit with ``SYS_ERROR -14 Bad address``, while
# the exact same child works fine otherwise. Details:
# https://github.com/rbln-sw/torch-rbln/pull/10533 and internal notes.


pytest.importorskip(
    "vllm_rbln",
    reason="vllm-rbln is not installed; install vllm-rbln to run vLLM LLM tests",
)


# Repo root: test/models/test_vllm_llm.py → parents[2] is the repo root.
# EngineCore is spawned via ``multiprocessing.spawn`` from inside vLLM, so its
# Python interpreter starts with a fresh ``sys.path`` (no implicit CWD). For
# the qualified-name ``worker_cls`` below to resolve in that subprocess we
# prepend the repo root to ``PYTHONPATH`` in ``_run_case``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Qualified path of the ``RBLNWorker`` subclass that skips the
# ``_init_device_env`` re-assignment. See
# ``test/models/_vllm_rbln_worker_patch.py`` for the full rationale.
_PATCHED_WORKER_CLS = "test.models._vllm_rbln_worker_patch.PatchedRBLNWorker"


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmModelConfig:
    """One model entry in the vLLM LLM test matrix."""

    model_id: str
    family: str
    # Kept small on purpose: the test only greedy-decodes ``MAX_TOKENS=5``
    # tokens from a short prompt, so we don't need a long context window
    # and larger values just inflate the KV-cache block budget that vllm
    # auto-computes in ``determine_available_memory`` — on tighter CI NPU
    # memory that can blow up as ``RuntimeError: Not enough memory for
    # N blocks of KV cache``.
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


# Hard-coded greedy (temperature=0) expected outputs from the CPU reference
# run. Each value is the exact ``generated_text`` (including leading space)
# returned by ``llm.generate``. If an entry is ``None`` the test falls back
# to non-empty + shape checks only.
#
# Key: (model_key, tp_size, mode) where mode in {"graph", "eager"}.
#
# Captured from a green run on RBLN-CA25 with the default fallback stack
# (torch-rbln + rebel-compiler dev329 + vllm-rbln device_tensor_rebased).
# Any future drift against these strings is a real correctness regression.
EXPECTED_TEXT: dict[tuple[str, int, str], Optional[str]] = {
    ("llama_3_2_1b", 1, "graph"): " Paris. The Eiff",
    ("qwen2_5_1_5b", 1, "graph"): " Paris. The capital of",
    ("qwen3_0_6b", 1, "graph"): " Paris. The capital of",
    # Eager mode uses VLLM_RBLN_SAMPLER=0 + VLLM_RBLN_COMPILE_MODEL=0 (see
    # _run_case); output then matches the graph-mode greedy decode exactly.
    ("llama_3_2_1b", 1, "eager"): " Paris. The Eiff",
    # TP2 via RSD (VLLM_RBLN_TP_SIZE=2, single vLLM worker over 2 physical
    # NPUs). Greedy decode matches TP1 exactly.
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
    """Run a single greedy vLLM generation on RBLN device tensors.

    Executed in a spawned subprocess so vllm-rbln / RBLN singleton state is
    isolated from the parent test runner.

    ``RBLN_DEVICES`` / ``RBLN_NPUS_PER_DEVICE`` are intentionally NOT set
    here. They must be exported in the *parent* pytest worker (see
    ``_run_case``) so this spawned child inherits them before its own
    ``import torch`` runs. Rationale: torch-rbln registers a
    ``torch.backends`` entry point (``torch_backends_entry_point``) that
    autoloads at ``import torch`` time and pulls in rebel's
    ``librbln-thunk.so``; the thunk snapshots ``RBLN_DEVICES`` at library
    load. Spawn re-imports this module before invoking the target function,
    so by the time control reaches this function body rebel has already
    recorded its initial value. Any mutation performed here (or later in
    vllm-rbln's stock ``RBLNWorker._init_device_env``) would trip rebel's
    consistency check ("RBLN_DEVICES environment variable changed at
    runtime. Initial value: , Current value: 0"). See the matching pattern
    used by ``test_optimum_llm.py::_run_test_case``.
    """
    from vllm import LLM, SamplingParams

    cfg = MODEL_CONFIGS[model_key]
    tc = unittest.TestCase()

    # NOTE: we always instantiate the vLLM engine with
    # ``tensor_parallel_size=1`` and use ``VLLM_RBLN_TP_SIZE`` (set in
    # ``_run_case``) to drive RSD fan-out across physical NPUs. That
    # keeps the test on a single worker process, exercising the "1 logical
    # device = ``tp_size`` physical NPUs" path (the setup we actually ship
    # for RSD) and avoids the RCCL multi-worker collective path, which is
    # a separate axis we do not care about here.
    #
    # ``worker_cls`` must be a qualified module path; vLLM's
    # ``ParallelConfig.worker_cls`` is typed ``str`` and ``worker_base.py``
    # rejects non-string values ("passing worker_cls is no longer
    # supported"). ``_run_case`` puts the repo root on ``PYTHONPATH`` so
    # EngineCore can import the named module.
    # ``gpu_memory_utilization=0.5`` caps the NPU memory share vllm uses
    # for KV cache, which in turn caps the number of blocks auto-computed
    # by ``determine_available_memory``. Default is 0.9; CI NPUs with
    # smaller HBM budgets otherwise fail with "Not enough memory for N
    # blocks of KV cache". The short prompt + ``MAX_TOKENS=5`` decode
    # only needs a couple of blocks, so 0.5 is plenty.
    llm = LLM(
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
        worker_cls=_PATCHED_WORKER_CLS,
    )

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
        # ``RBLN_DEVICES`` (and, for RSD, ``RBLN_NPUS_PER_DEVICE``) MUST be
        # exported here in the parent, *before* ``run_in_isolated_process``
        # spawns the worker. Both the spawned ``_vllm_generate_worker``
        # child and the EngineCore subprocess it later forks inherit this
        # env at process-start, so rebel's ``librbln-thunk.so`` — loaded
        # via the ``torch.backends`` entry point that fires on ``import
        # torch`` — snapshots the correct value once and stays consistent
        # through ``determine_available_memory``. Setting them inside the
        # child is too late: ``multiprocessing.spawn`` re-imports this
        # module (and therefore runs ``import torch``) before calling
        # ``_vllm_generate_worker``. Same pattern as
        # ``test_optimum_llm.py::_run_test_case``.
        mp.setenv("RBLN_DEVICES", ",".join(str(i) for i in range(tp_size)))
        if tp_size > 1:
            mp.setenv("RBLN_NPUS_PER_DEVICE", str(tp_size))

        # Ensure ``test.models._vllm_rbln_worker_patch`` is importable in the
        # EngineCore subprocess. EngineCore is spawned by vLLM's
        # ``multiprocessing.spawn`` path, which starts a fresh Python
        # interpreter whose ``sys.path`` does not implicitly include the
        # repo root. ``PYTHONPATH`` is the reliable way to inject it and it
        # propagates through ``multiprocessing.spawn`` → ``EngineCore``.
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            mp.setenv(
                "PYTHONPATH",
                f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}",
            )
        else:
            mp.setenv("PYTHONPATH", str(_REPO_ROOT))

        # Env matches the vllm-rbln ``test_llama_batch.py`` reference
        # announced to run green on ``device_tensor_rebased``. The opt-in
        # ``VLLM_RBLN_USE_DEVICE_TENSOR=1`` flow is intentionally NOT set:
        # it is unstable on this branch (rebel runtime trips "Buffer not
        # found for DramTensor" -> SYS_ERROR -14 at submit). We exercise
        # the native vLLM model path over the legacy KV-cache path, which
        # is the configuration the vllm-rbln team currently tests.
        mp.setenv("VLLM_USE_V1", "1")
        mp.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        mp.setenv("VLLM_RBLN_SAMPLER", "1")
        mp.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
        mp.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        mp.setenv("RBLN_KERNEL_MODE", "triton")
        mp.setenv("RBLN_PV_OPT", "1")
        # torch-rbln: surface compile errors rather than silently fall back.
        mp.setenv("TORCH_RBLN_DISABLE_FALLBACK", "compile_error")
        # RSD fan-out — not vllm worker-level TP. ``VLLM_RBLN_TP_SIZE`` is
        # the axis that tells vllm-rbln to claim ``tp_size`` physical NPUs
        # for a single worker and merge them into one logical device; the
        # worker then sets ``RBLN_NPUS_PER_DEVICE`` on its own (see
        # ``vllm_rbln/v1/worker/rbln_worker.py::_init_device_env``). This
        # is the axis we want to validate end-to-end — it keeps the test
        # on a single process and skips the RCCL multi-worker collective
        # path entirely. The vLLM engine is instantiated with
        # ``tensor_parallel_size=1`` in ``_vllm_generate_worker`` to match.
        mp.setenv("VLLM_RBLN_TP_SIZE", str(tp_size))
        if enforce_eager:
            # Eager mode + the default RBLN sampler (``VLLM_RBLN_SAMPLER=1``)
            # trips rebel's weight-reuse assertion at
            # RTOSAWeightReusabilityCheck. Root cause chain: the main model
            # is not torch.compile'd in eager mode, so every forward sees
            # dynamic input shapes; the RBLN topk/topp sampler is still
            # torch.compile'd on each shape and re-registers under the same
            # module name. rebel's use_weight_sharing path then flips the
            # second compile into ``weight_mode="reuse"`` and aborts with
            # ``OpInvalidWeightSharingError: no key found on reuse mode``.
            # Falling back to vllm's native (CPU) sampler removes the
            # repeated rbln compile entirely, so eager mode exercises the
            # rest of the path end-to-end. Revisit once rebel tolerates
            # shape-only recompiles under ``use_weight_sharing``.
            mp.setenv("VLLM_RBLN_SAMPLER", "0")
            # vllm-rbln's flash-attention backend has multiple branches
            # gated on ``VLLM_RBLN_COMPILE_MODEL`` (attention metadata
            # layout, KV cache wiring, slot-mapping). With
            # ``enforce_eager=True`` the main model is not torch.compile'd,
            # but ``VLLM_RBLN_COMPILE_MODEL`` defaults to True, so the
            # attention backend keeps taking the compile-mode code paths —
            # the resulting mismatch causes the eager forward to emit wrong
            # logits and the sampler to produce gibberish tokens. Aligning
            # ``VLLM_RBLN_COMPILE_MODEL=0`` with eager execution makes the
            # attention path consistent with the uncompiled model.
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
    """Graph mode with RSD=2 — 1 vLLM worker fans a single logical device
    out across 2 physical NPUs via ``VLLM_RBLN_TP_SIZE=2``. Skipped if
    fewer than 2 physical NPUs are available. This does **not** exercise
    the RCCL multi-worker collective path (that is a separate axis)."""
    _run_case(model_key=model_key, tp_size=2, enforce_eager=False)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.parametrize("model_key", ["llama_3_2_1b"])
def test_vllm_llm_eager_tp1(enable_deploy_mode, model_key):
    """Eager mode TP=1 — sanity check non-compile execution path.

    Uses ``VLLM_RBLN_SAMPLER=0`` + ``VLLM_RBLN_COMPILE_MODEL=0`` (set in
    ``_run_case``) so the attention backend and sampler stay consistent
    with the uncompiled main model; see ``_run_case`` for the rationale.
    """
    _run_case(model_key=model_key, tp_size=1, enforce_eager=True)


if __name__ == "__main__":
    # NOTE: invoked via pytest, not ``python -m test.models.test_vllm_llm``.
    # The canonical entry is ``pytest test/models/test_vllm_llm.py``; see the
    # module docstring for the reason we do not import
    # ``torch.testing._internal.common_utils.run_tests`` here.
    raise SystemExit("Run this module via pytest: pytest test/models/test_vllm_llm.py")
