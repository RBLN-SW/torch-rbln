# Owner(s): ["module: PrivateUse1"]
"""The paged-attention custom ops inside a larger compiled graph.

``test_custom_kernel.py`` drives ``rbln_custom_ops.paged_attn_*`` through eager dispatch,
which compiles each op on its own. An inference engine's graph mode is different: the op
sits inside one program together with the projections around it, and the KV caches are
program inputs the op mutates in place. This test builds that shape -- q/k/v projection ->
paged attention -> output projection -- as one ``torch.compile`` program and checks it
against the eager dispatch of the same module: same output, same cache contents afterwards,
for the prefill kernel (batch 1) and the decode kernel (batch 2).
"""

import math

import pytest
import rebel  # noqa: F401  -- defines the rbln_custom_ops schemas
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 64
MAX_SEQ = 512  # one block: block_size == MAX_SEQ, block_table maps every batch row to its own block
HIDDEN = NUM_KV_HEADS * NUM_Q_HEADS * HEAD_DIM


class PagedAttentionBlock(torch.nn.Module):
    """Projections around a paged-attention kernel, the way a decoder layer wraps it."""

    def __init__(self, phase: str):
        super().__init__()
        torch.manual_seed(0)
        self.phase = phase
        self.q = torch.nn.Linear(HIDDEN, NUM_KV_HEADS * NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.kv = torch.nn.Linear(HIDDEN, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(NUM_KV_HEADS * NUM_Q_HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x, mask, k_cache, v_cache, seq, block_table):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 3, 1, 4)
        k, v = self.kv(x).view(b, t, 2, NUM_KV_HEADS, 1, HEAD_DIM).permute(2, 0, 3, 4, 1, 5)
        scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM))  # constant: the kernel takes scale as a tensor
        op = (
            torch.ops.rbln_custom_ops.paged_attn_prefill
            if self.phase == "prefill"
            else torch.ops.rbln_custom_ops.paged_attn_decode
        )
        a = op(q, k, v, mask, k_cache, v_cache, seq, scale, block_table, MAX_SEQ)
        return self.o(a.permute(0, 3, 1, 2, 4).reshape(b, t, -1))


def _inputs(phase: str, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(1)
    if phase == "prefill":
        b, t, ctx = 1, 64, 0
    else:
        b, t, ctx = 2, 1, 64
    x = torch.randn(b, t, HIDDEN, dtype=dtype)
    k_cache = torch.zeros(b, NUM_KV_HEADS, 1, MAX_SEQ, HEAD_DIM, dtype=dtype)
    v_cache = torch.zeros(b, NUM_KV_HEADS, 1, MAX_SEQ, HEAD_DIM, dtype=dtype)
    if ctx:  # decode: a processed prompt already sits in the cache
        k_cache[..., :ctx, :] = torch.randn(b, NUM_KV_HEADS, 1, ctx, HEAD_DIM, dtype=dtype)
        v_cache[..., :ctx, :] = torch.randn(b, NUM_KV_HEADS, 1, ctx, HEAD_DIM, dtype=dtype)
    mask = torch.zeros(b, 1, 1, t, MAX_SEQ, dtype=dtype)
    if phase == "prefill":
        mask[0, 0, 0] = torch.ones(t, MAX_SEQ, dtype=dtype).tril()
    else:
        mask[..., : ctx + 1] = 1.0
    seq = torch.full((b, 1), ctx, dtype=torch.int32)
    block_table = (
        torch.arange(b, dtype=torch.int16).view(b, 1) if phase == "decode" else torch.zeros(b, dtype=torch.int16)
    )
    return tuple(a.to(device) for a in (x, mask, k_cache, v_cache, seq, block_table))


@pytest.mark.test_set_ci
# Compile-heavy; keep the RBLN compile cache across tests (skip per-test dynamo reset).
@pytest.mark.no_dynamo_reset
class TestPagedAttentionInGraph(TestCase):
    rbln_device = torch.device("rbln:0")

    def _check(self, phase: str, dtype: torch.dtype):
        atol = 2e-2 if dtype is torch.float16 else 8e-2
        module = PagedAttentionBlock(phase).eval().to(device=self.rbln_device, dtype=dtype)

        eager_in = _inputs(phase, dtype, self.rbln_device)
        graph_in = _inputs(phase, dtype, self.rbln_device)
        compiled = torch.compile(module, backend="rbln", dynamic=False)
        with torch.no_grad():
            expected = module(*eager_in)  # eager dispatch: the op compiled on its own
            out = compiled(*graph_in)  # one program: projections + op + projection
            out2 = compiled(*_inputs(phase, dtype, self.rbln_device))  # warm program, fresh caches

        torch.testing.assert_close(out.float().cpu(), expected.float().cpu(), atol=atol, rtol=0.0)
        torch.testing.assert_close(out2.float().cpu(), expected.float().cpu(), atol=atol, rtol=0.0)
        # The op mutates the caches it was handed; through the program, that write must land
        # in the caller's tensors just as it does in eager.
        for name, i in (("k_cache", 2), ("v_cache", 3)):
            torch.testing.assert_close(
                graph_in[i].float().cpu(), eager_in[i].float().cpu(), atol=atol, rtol=0.0, msg=f"{name} after {phase}"
            )
            self.assertGreater(graph_in[i].float().cpu().abs().sum().item(), 0.0, f"{name} untouched by {phase}")

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_prefill_in_graph(self, dtype):
        self._check("prefill", dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_decode_in_graph(self, dtype):
        self._check("decode", dtype)


instantiate_device_type_tests(TestPagedAttentionInGraph, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
