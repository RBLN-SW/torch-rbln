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
* ``vllm_rbln`` / ``vllm`` importable.

Matrix
------
* Default: graph mode (``enforce_eager=False``) — the primary compile path.
* One case: eager mode (``enforce_eager=True``) — sanity-check the eager
  execution path.
"""

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import torch
import vllm_rbln  # noqa: F401

from test.utils import is_rebel_device, run_in_isolated_process, SUPPORTED_DTYPES


# NOTE: do NOT import ``torch.testing._internal.common_utils`` at module level.
# Spawned children re-import this module, and that import path perturbs rebel
# runtime state into ``SYS_ERROR -14`` on NPU submit. See PR #10533.


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


# Greedy-decode (temperature=0) expected outputs. Key: (model, tp, mode, dtype).
# ``None`` falls back to non-empty + shape checks. Captured on RBLN-CA25 with
# rebel-compiler dev329 + vllm-rbln device_tensor_rebased; drift = regression.
EXPECTED_TEXT: dict[tuple[str, int, str, torch.dtype], Optional[str]] = {
    ("llama_3_2_1b", 1, "graph", torch.float16): " Paris. The Eiff",
    ("llama_3_2_1b", 1, "graph", torch.bfloat16): " Paris. The Eiff",
    ("qwen2_5_1_5b", 1, "graph", torch.float16): " Paris. The capital of",
    ("qwen2_5_1_5b", 1, "graph", torch.bfloat16): " Paris. The capital of",
    ("qwen3_0_6b", 1, "graph", torch.float16): " Paris. The capital of",
    ("qwen3_0_6b", 1, "graph", torch.bfloat16): " Paris. The capital of",
    ("llama_3_2_1b", 1, "eager", torch.float16): " Paris. The Eiff",
    ("llama_3_2_1b", 1, "eager", torch.bfloat16): " Paris. The Eiff",
    ("llama_3_2_1b", 2, "graph", torch.float16): " Paris. The Eiff",
    ("llama_3_2_1b", 2, "graph", torch.bfloat16): " Paris. The Eiff",
}


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _vllm_generate_worker(
    model_key: str,
    tp_size: int,
    enforce_eager: bool,
    dtype: torch.dtype,
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

    # Always TP=1 on the vLLM engine; RSD fan-out is driven by
    # ``VLLM_RBLN_TP_SIZE`` (set in ``_run_case``), keeping the test
    # single-process and off the RCCL multi-worker path.
    # ``gpu_memory_utilization`` sizes the KV-cache pool. REBEL has 140 GB DRAM, so
    # the ATOM-tuned 0.5 (= 70 GB) massively over-allocates it (~66 GiB for a 1B
    # model); at that size eager strided KV writes hit deep vmem addresses the v2v
    # engine rejects, then OOM. 0.1 (~10 GiB) is plenty here; ATOM (16 GB) keeps 0.5.
    llm_kwargs: dict = dict(
        model=cfg.model_id,
        # vLLM expects the dtype as a string name (e.g., "float16", "bfloat16"), not a torch.dtype.
        dtype=str(dtype).removeprefix("torch."),
        max_model_len=cfg.max_model_len,
        block_size=cfg.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        tensor_parallel_size=1,
        trust_remote_code=cfg.trust_remote_code,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.1 if is_rebel_device() else 0.5,
    )
    # TP>1 only — see ``_PATCHED_WORKER_CLS`` above.
    if tp_size > 1:
        llm_kwargs["worker_cls"] = _PATCHED_WORKER_CLS

    llm = LLM(**llm_kwargs)

    try:
        outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=max_tokens))
        assert len(outputs) == 1, "expected a single RequestOutput"

        gen = outputs[0].outputs[0]
        gen_text = gen.text
        gen_ids = list(gen.token_ids)
    finally:
        # Deterministically tear down the vLLM EngineCore subprocess *before*
        # this worker exits. vLLM V1 runs EngineCore in its own spawned process
        # and relies on a ``weakref.finalize`` for cleanup. If the worker exits
        # via an exception (e.g. the output-mismatch assert below), multiprocessing's
        # atexit ``_exit_function`` joins the still-running EngineCore before that
        # finalizer fires and deadlocks — the AssertionError only surfaces after a
        # manual SIGINT. Shutting down here keeps failures observable.
        llm.llm_engine.engine_core.shutdown()

    mode = "eager" if enforce_eager else "graph"
    print(f"[vllm_llm_test] model={model_key} tp={tp_size} mode={mode} dtype={dtype} text={gen_text!r} ids={gen_ids}")

    assert len(gen_text) > 0, "generated text should not be empty"
    assert len(gen_ids) == max_tokens, "greedy decode should fill max_tokens"
    assert all(isinstance(i, int) and i >= 0 for i in gen_ids)

    if expected_text is not None:
        assert gen_text == expected_text, (
            f"vLLM RBLN output mismatch for {model_key} tp{tp_size} {mode} {dtype}. "
            f"Expected: {expected_text!r}, Got: {gen_text!r}"
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _skip_if_not_enough_npus(tp_size: int) -> None:
    if tp_size <= 1:
        return
    if is_rebel_device():
        pytest.skip("TP>1 is not supported on the REBEL (CR) lineup yet (single device = quad chiplet)")
    n_phys = torch.rbln.physical_device_count()
    if n_phys < tp_size:
        pytest.skip(f"Requires at least {tp_size} physical devices, found {n_phys}")


def _run_case(model_key: str, tp_size: int, enforce_eager: bool, dtype: torch.dtype) -> None:
    cfg = MODEL_CONFIGS[model_key]
    if tp_size > cfg.max_npus:
        pytest.skip(f"{model_key} matrix entry restricted to <={cfg.max_npus} NPUs, got tp={tp_size}")
    _skip_if_not_enough_npus(tp_size)

    mode = "eager" if enforce_eager else "graph"
    expected_text = EXPECTED_TEXT.get((model_key, tp_size, mode, dtype))

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
        # REBEL graph compile: triton custom-kernel mode hits a rebel-compiler
        # bug (RTOSA ExpandContribCustomOps has no bf16 branch), so use the
        # string-kernel path + device tensors. ATOM and eager keep triton.
        rebel_graph = is_rebel_device() and not enforce_eager
        if rebel_graph:
            mp.delenv("RBLN_KERNEL_MODE", raising=False)
            mp.setenv("VLLM_RBLN_USE_DEVICE_TENSOR", "1")
        else:
            mp.setenv("RBLN_KERNEL_MODE", "triton")
        mp.setenv("RBLN_PV_OPT", "1")
        # RSD fan-out (1 worker → ``tp_size`` physical NPUs merged into one
        # logical device). vllm engine stays at ``tensor_parallel_size=1``.
        mp.setenv("VLLM_RBLN_TP_SIZE", str(tp_size))
        if enforce_eager:
            # Eager mode bypasses torch.compile, so ops must dispatch to a real
            # device='rbln' instead of the compile-backend fake-CPU tensors of
            # the default vLLM model path. vllm-rbln's platform validation now
            # rejects ``enforce_eager=True`` without this. See platform.py.
            mp.setenv("VLLM_RBLN_USE_DEVICE_TENSOR", "1")
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
            dtype,
            expected_text,
            PROMPT,
            MAX_TOKENS,
        )


# ---------------------------------------------------------------------------
# Tests — graph mode is the default (enforce_eager=False); eager has one case.
# ---------------------------------------------------------------------------


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("model_key", list(MODEL_CONFIGS.keys()))
def test_vllm_llm_graph_tp1(model_key, dtype):
    """Graph mode (torch.compile) TP=1 — primary device-tensor validation."""
    _run_case(model_key=model_key, tp_size=1, enforce_eager=False, dtype=dtype)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("model_key", ["llama_3_2_1b"])
def test_vllm_llm_graph_tp2(model_key, dtype):
    """Graph mode RSD=2 — 1 vLLM worker over 2 physical NPUs via
    ``VLLM_RBLN_TP_SIZE=2``. Skipped if <2 NPUs. Does not exercise the
    RCCL multi-worker collective path (separate axis)."""
    _run_case(model_key=model_key, tp_size=2, enforce_eager=False, dtype=dtype)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("model_key", ["llama_3_2_1b"])
def test_vllm_llm_eager_tp1(model_key, dtype):
    """Eager mode TP=1 — sanity check non-compile path. Greedy decode should
    match graph-mode TP1 (eager workaround envs set in ``_run_case``)."""
    _run_case(model_key=model_key, tp_size=1, enforce_eager=True, dtype=dtype)


# ---------------------------------------------------------------------------
# No-NPU compile-only (RBLN_DUMMY_DEVICE)
# ---------------------------------------------------------------------------

# vLLM's native path (VLLM_RBLN_USE_VLLM_MODEL=1) under VLLM_RBLN_COMPILE_ONLY=1
# builds each graph and writes its .rbln artifact to the compile cache without
# executing -- the runtime is constructed on a dummy device, so no NPU is needed.
# This is the reason RBLN_DUMMY_DEVICE exists: compile a servable model on a host
# with no hardware. Run in a subprocess with the dummy env set before ``import
# torch`` (rebel snapshots RBLN_* at import). Dummy mode host-backs the device
# (physical count stays 0), so no real NPU is used even on a machine that has one.
_COMPILE_ONLY_DUMMY_WORKER = """
import glob, os
import torch_rbln  # noqa: F401  (registers the dummy PrivateUse1 device)
import torch

assert torch.rbln.is_dummy_device() is True, "dummy mode is not active"
assert torch.rbln.physical_device_count() == 0, torch.rbln.physical_device_count()

from vllm import LLM

# compile_only builds artifacts during engine init; no generate() -> no execution.
LLM(
    model=os.environ["RBLN_TEST_MODEL_ID"],
    dtype="float16",
    max_model_len=2048,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
    tensor_parallel_size=1,
    enforce_eager=False,
    gpu_memory_utilization=0.1,
)
arts = glob.glob(os.path.join(os.environ["VLLM_CACHE_ROOT"], "rbln", "**", "*.rbln"), recursive=True)
assert arts, "compile-only produced no .rbln artifacts"
print(f"OK artifacts={len(arts)}")
"""


@pytest.mark.test_set_ci
@pytest.mark.single_worker
def test_vllm_compile_only_no_npu_via_dummy(tmp_path):
    """vLLM native-path compile-only on a no-NPU host (RBLN_DUMMY_DEVICE).

    The point of dummy mode: vLLM can build a servable model's .rbln artifacts
    with no NPU present. Needs no real device (forced onto a non-existent one),
    so it runs anywhere vllm-rbln is importable, including a CPU-only host.
    """
    env = dict(os.environ)
    # rebel snapshots RBLN_* at import. Drop any inherited device mapping so vLLM
    # sets its own (user 0 -> system 0); a stale RBLN_DEVICE_MAP conflicts with
    # that and makes device_count() fail. Dummy mode host-backs the device
    # regardless (physical count stays 0), so no real NPU is used.
    for key in ("RBLN_DEVICES", "RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE"):
        env.pop(key, None)
    env.update(
        RBLN_DUMMY_DEVICE="1",
        RBLN_TARGET_SOC="RBLN-CA25",
        VLLM_RBLN_USE_VLLM_MODEL="1",
        VLLM_RBLN_COMPILE_ONLY="1",
        VLLM_CACHE_ROOT=str(tmp_path / "vllm_cache"),
        RBLN_TEST_MODEL_ID=MODEL_CONFIGS["qwen3_0_6b"].model_id,
    )
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_COMPILE_ONLY_DUMMY_WORKER)],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert proc.returncode == 0, f"compile-only worker failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK artifacts=" in proc.stdout, proc.stdout


if __name__ == "__main__":
    # See module-level NOTE for why we don't use ``common_utils.run_tests``.
    raise SystemExit("Run this module via pytest: pytest test/models/test_vllm_llm.py")
