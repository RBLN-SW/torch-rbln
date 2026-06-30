# Owner(s): ["module: PrivateUse1"]
"""Regression test for eager-mode async buffer lifetime.

In eager mode every op runs as its own device submission, and device buffers are
constantly allocated, freed, and reallocated between ops. This test guards against a
class of failure where a device buffer is freed or reallocated while asynchronous device
work that still references it is in flight — which corrupts buffer state and aborts the
runtime on a later op.

The workload is a decode-like eager loop (per-layer q/k/v/o projections + matmul attention
over a persistent KV-style cache, with ``empty_cache()`` churn between steps), run in a
freshly spawned process. On a correct runtime it completes; if a buffer is recycled out
from under in-flight async work, the worker process aborts and the test fails. The race is
timing-dependent, so the worker retries the workload a few times to fail reliably when the
bug is present.

Requirements (provided by fixtures in ``test/conftest.py``):
  * ``TORCH_RBLN_DEPLOY=ON`` (``enable_deploy_mode``) — required to exercise this path.
  * ``TORCH_RBLN_DISABLE_FALLBACK`` includes ``compile_error`` (``disable_compile_error_fallback``,
    autouse) — so the runtime error surfaces instead of falling back to CPU.

The full model-level path is already covered by
``test/models/test_optimum_llm.py::TestLlamaEagerPRIVATEUSE1``; this is a fast, model-free
supplement. It runs in a spawned process because the failure aborts the worker.
"""

import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import requires_logical_devices, run_in_isolated_process


def _eager_matmul_decode_worker(
    attempts: int = 8,
    steps: int = 250,
    layers: int = 8,
    hidden: int = 2048,
    cache_len: int = 128,
    dtype_str: str = "float16",
) -> None:
    """Decode-like eager loop (q/k/v/o linears + matmul attention over a persistent KV
    cache + ``empty_cache`` churn) that stresses eager buffer alloc/free/reallocate.

    Module-level (picklable) so it can be the target of ``mp.spawn``. The spawned process
    inherits ``TORCH_RBLN_DEPLOY=ON`` from the parent (``enable_deploy_mode``). On a correct
    runtime it returns; if a buffer is recycled under in-flight async work it aborts.
    """
    import torch

    import torch_rbln

    dt = torch.float16 if dtype_str == "float16" else torch.bfloat16
    dev = torch.device("rbln", 0)
    torch.manual_seed(0)

    q_proj = [torch.nn.Linear(hidden, hidden, bias=False).to(dev, dtype=dt) for _ in range(layers)]
    k_proj = [torch.nn.Linear(hidden, hidden, bias=False).to(dev, dtype=dt) for _ in range(layers)]
    v_proj = [torch.nn.Linear(hidden, hidden, bias=False).to(dev, dtype=dt) for _ in range(layers)]
    o_proj = [torch.nn.Linear(hidden, hidden, bias=False).to(dev, dtype=dt) for _ in range(layers)]
    k_cache = [torch.zeros(cache_len, hidden, dtype=dt, device=dev) for _ in range(layers)]
    v_cache = [torch.zeros(cache_len, hidden, dtype=dt, device=dev) for _ in range(layers)]

    for _ in range(attempts):
        for step in range(steps):
            pos = step % cache_len
            h = torch.randn(1, hidden, dtype=dt, device=dev)
            for layer in range(layers):
                q = q_proj[layer](h)
                k = k_proj[layer](h)
                v = v_proj[layer](h)
                k_cache[layer][pos] = k[0]  # scatter into KV cache
                v_cache[layer][pos] = v[0]
                scores = torch.matmul(q, k_cache[layer][: pos + 1].transpose(0, 1))  # qk^T
                probs = torch.softmax(scores.float(), dim=-1).to(dt)
                ctx = torch.matmul(probs, v_cache[layer][: pos + 1])  # attn @ v
                h = o_proj[layer](ctx) + h
            del h
            torch_rbln.memory.empty_cache(dev)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
class TestEagerAsyncBufferRegression(TestCase):
    """Eager-mode async buffer-lifetime regression."""

    @requires_logical_devices(1)
    def test_eager_decode_loop_no_async_buffer_crash(self):
        """A decode-like eager loop must not crash the runtime.

        Fails if a device buffer is freed/reallocated while async work referencing it is
        still in flight; passes on a runtime that keeps buffer lifetime consistent.
        """
        run_in_isolated_process(_eager_matmul_decode_worker)


if __name__ == "__main__":
    run_tests()
