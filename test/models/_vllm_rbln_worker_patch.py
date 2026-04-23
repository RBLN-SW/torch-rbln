"""Patched vllm-rbln ``RBLNWorker`` used by ``test_vllm_llm``.

This module is intentionally kept as a separate file (and imported *only*
by the EngineCore subprocess via a qualified-name ``worker_cls``) so that:

* vLLM's ``ParallelConfig.worker_cls`` field (``str``) accepts the value —
  the cloudpickled-class path was removed upstream; ``worker_base.py``
  now explicitly rejects non-string worker_cls values
  ("passing worker_cls is no longer supported").
* Collection of ``test_vllm_llm`` in the parent pytest worker does *not*
  eagerly import ``vllm_rbln.v1.worker.rbln_worker`` (which would pull in
  ``vllm`` + ``rebel`` at module load time and muddle the parent process
  rebel state).

``_vllm_generate_worker`` pre-sets ``RBLN_DEVICES`` (and, for RSD,
``RBLN_NPUS_PER_DEVICE``) *before* any rebel-loading import, and then
names this module in ``worker_cls=...``. The EngineCore subprocess
inherits that env, imports this module, and uses the patched worker —
whose only change is to skip the env-var re-assignment that stock
vllm-rbln performs in ``_init_device_env``.
"""

from vllm_rbln.v1.worker.rbln_worker import RBLNWorker


class PatchedRBLNWorker(RBLNWorker):
    """RBLNWorker that trusts the pre-set ``RBLN_DEVICES`` env var.

    Stock vllm-rbln's ``_init_device_env`` re-assigns ``RBLN_DEVICES``
    from ``VLLM_RBLN_TP_SIZE`` + ``local_rank`` inside
    ``RBLNWorker.__init__``. That assignment runs *after* rebel's
    ``librbln-thunk.so`` has snapshotted the environment at library-load
    time, so the first ``rebel.get_npu_name`` call (from
    ``determine_available_memory``) trips rebel's consistency check:

        RBLNRuntimeError: RBLN_DEVICES environment variable changed at
        runtime. Initial value: , Current value: 0

    The test driver pre-computes the final expanded value (``"0"`` for
    TP1, ``"0,1"`` for TP2) and exports ``RBLN_DEVICES`` *before* the
    first rebel-loading import, so re-running the mapping here is both
    redundant and actively harmful. Override it to a no-op.
    """

    def _init_device_env(self) -> None:  # type: ignore[override]
        return
