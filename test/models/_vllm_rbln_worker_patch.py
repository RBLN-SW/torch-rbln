"""Patched vllm-rbln ``RBLNWorker`` used by ``test_vllm_llm`` for TP>1.

Used only by the TP>1 (RSD) test cases; TP=1 uses the stock worker via
``ParallelConfig.worker_cls="auto"`` because stock vllm-rbln rewrites
``RBLN_DEVICES`` to the same value (``"0"`` -> ``"0"``) in that config and
rebel does not flag a same-string assignment as a runtime mutation.

For TP>1 the stock worker either fails an assertion (when ``RBLN_DEVICES``
is preset in physical-device format, ``"0,1"``) or rewrites
``RBLN_DEVICES`` from logical to physical format (``"0"`` -> ``"0,1"``) —
which rebel's ``librbln-thunk.so`` flags as a runtime mutation against the
value snapshotted at libthunk-load time, raising

    RBLNRuntimeError: RBLN_DEVICES environment variable changed at runtime.
    Initial value: 0, Current value: 0,1

The test driver pre-computes the expanded physical value (``"0,1"`` for
TP2), exports ``RBLN_DEVICES`` *before* any rebel-loading import, and uses
this subclass to skip the in-worker re-assignment so the env stays
consistent through ``determine_available_memory``.

This module is intentionally kept as a separate file (and imported only by
the EngineCore subprocess via a qualified-name ``worker_cls``) so that:

* vLLM's ``ParallelConfig.worker_cls`` field (``str``) accepts the value —
  the cloudpickled-class path was removed upstream; ``worker_base.py`` now
  explicitly rejects non-string ``worker_cls`` values.
* Pytest collection of ``test_vllm_llm`` in the parent worker does *not*
  eagerly import ``vllm_rbln.v1.worker.rbln_worker`` (which would pull in
  ``vllm`` + ``rebel`` at module load time and muddle the parent process's
  rebel state).
"""

from vllm_rbln.v1.worker.rbln_worker import RBLNWorker


class PatchedRBLNWorker(RBLNWorker):
    """RBLNWorker that trusts the pre-set ``RBLN_DEVICES`` env var."""

    def _init_device_env(self) -> None:  # type: ignore[override]
        return
