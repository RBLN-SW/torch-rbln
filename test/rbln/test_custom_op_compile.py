# Owner(s): ["module: PrivateUse1"]
"""The attention kernel an engine calls, inside a larger compiled graph.

``test_custom_kernel.py`` compiles each ``rbln_custom_ops`` kernel on its own and asserts the
wiring contract. An engine's graph mode is different: the op sits in one program with the
projections around it, and the KV cache is a program input it mutates in place. This test
builds that shape -- q/kv projection -> attention -> output projection -- as one
``torch.compile`` program.

The kernel is ``flash_causal_attention_naive_{prefill,decode}``, the one vllm-rbln's flash
backend selects for paged serving (``VLLM_RBLN_FLASH_CAUSAL_ATTN``, the default).

Both references are the op's own plain-torch body, the one rebel_compiler registers. On the CPU
in fp32 it is an independent definition of what the kernel computes. Dispatched eagerly on a
device tensor it is that same body -- torch-rbln registers no PrivateUse1 kernel for this
family -- carrying the device's 16-bit formats without the compiler seeing the program, which
holds the compiled run to a tighter bound than fp32 does.
"""

import math

import pytest
import rebel  # noqa: F401  -- defines the rbln_custom_ops schemas
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


NUM_KV_HEADS, NUM_Q_GROUPS, HEAD_DIM = 2, 4, 64
PARTITION, NUM_PARTITIONS = 128, 2  # paged: the block is shorter than the context it spans
HIDDEN = NUM_KV_HEADS * NUM_Q_GROUPS * HEAD_DIM

# Both phases reach past the first block: prefill continues from the block boundary, decode
# spans both blocks and appends into the second. With an empty cache in one block, a kernel
# that lost the block table or the position would still pass.
PREFILL_CACHED, PREFILL_TOKENS = PARTITION, 96
DECODE_CACHED = PARTITION + 32


class AttentionBlock(torch.nn.Module):
    """Projections around the attention kernel, the way a decoder layer wraps it."""

    def __init__(self, phase: str):
        super().__init__()
        torch.manual_seed(0)
        self.phase = phase
        self.q = torch.nn.Linear(HIDDEN, NUM_KV_HEADS * NUM_Q_GROUPS * HEAD_DIM, bias=False)
        self.kv = torch.nn.Linear(HIDDEN, 2 * NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(NUM_KV_HEADS * NUM_Q_GROUPS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x, kv_cache, seq_idx, block_tables):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, NUM_KV_HEADS, NUM_Q_GROUPS, HEAD_DIM).permute(0, 2, 3, 1, 4)
        k, v = self.kv(x).view(b, t, 2, NUM_KV_HEADS, 1, HEAD_DIM).permute(2, 0, 3, 4, 1, 5)
        scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM))  # constant: the kernel takes scale as a tensor
        op = (
            torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill
            if self.phase == "prefill"
            else torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode
        )
        # The last operand is slot_mapping, which the kernel does not read; the engine feeds it
        # `scale` again rather than building a tensor for it.
        a = op(q, k, v, kv_cache, scale, seq_idx, block_tables, scale)
        return self.o(a.permute(0, 3, 1, 2, 4).reshape(b, t, -1))


def _inputs(phase: str, dtype: torch.dtype, device: torch.device | str):
    """Engine-shaped operands. Prefill is single-batch; decode runs a batch of sequences."""
    torch.manual_seed(1)
    b, t, cached = (1, PREFILL_TOKENS, PREFILL_CACHED) if phase == "prefill" else (2, 1, DECODE_CACHED)
    # Drawn in fp32 and cast: randn() in a 16-bit dtype is a different RNG sequence, which
    # would hand the reference and the device runs different inputs.
    x = torch.randn(b, t, HIDDEN).to(dtype)
    # Each sequence owns NUM_PARTITIONS blocks; its prompt fills them in order. The block axis
    # comes first, ahead of the K/V pair -- the layout the kernel documents.
    kv_cache = torch.zeros(b * NUM_PARTITIONS, 2, NUM_KV_HEADS, 1, PARTITION, HEAD_DIM, dtype=dtype)
    for i in range(b):
        left = cached
        for p in range(NUM_PARTITIONS):
            n = min(left, PARTITION)
            if n <= 0:
                break
            kv_cache[i * NUM_PARTITIONS + p, :, :, :, :n, :] = torch.randn(2, NUM_KV_HEADS, 1, n, HEAD_DIM).to(dtype)
            left -= n
    # seq_idx is [batch, 1], the position of the first query token -- what the engine passes
    # (``positions[query_start_loc]``). The kernel derives the per-block offsets from it.
    seq_idx = torch.full((b, 1), cached, dtype=torch.int32)
    block_tables = (
        torch.arange(NUM_PARTITIONS, dtype=torch.int16)  # prefill: 1-D, the one sequence's blocks
        if phase == "prefill"
        else torch.stack(  # decode: [batch, num_partitions]
            [torch.arange(i * NUM_PARTITIONS, (i + 1) * NUM_PARTITIONS, dtype=torch.int16) for i in range(b)]
        )
    )
    return [a.to(device) for a in (x, kv_cache, seq_idx, block_tables)]


@pytest.mark.test_set_ci
# Compile-heavy; keep the RBLN compile cache across tests (skip per-test dynamo reset).
@pytest.mark.no_dynamo_reset
class TestAttentionInGraph(TestCase):
    rbln_device = torch.device("rbln:0")

    def _check(self, phase: str, dtype: torch.dtype):
        with torch.no_grad():
            reference_in = _inputs(phase, torch.float32, "cpu")
            reference = AttentionBlock(phase).eval()(*reference_in)  # the op's own torch body, in fp32

            module = AttentionBlock(phase).eval().to(device=self.rbln_device, dtype=dtype)
            compiled = torch.compile(module, backend="rbln", dynamic=False)
            eager_in, graph_in = _inputs(phase, dtype, self.rbln_device), _inputs(phase, dtype, self.rbln_device)
            eager = module(*eager_in)  # eager dispatch: the op compiled on its own
            out = compiled(*graph_in)  # one program: projections + op + projection
            warm = compiled(*_inputs(phase, dtype, self.rbln_device))  # warm program, fresh cache

        # Tolerances are fractions of the reference's scale: against fp32 each device run
        # carries the device's 16-bit formats, while the two device runs differ only in how much
        # of the program the compiler saw. A wrong cache read misses by the scale itself.
        scale = float(reference.abs().max())
        device_tol, pair_tol = 0.1 * scale, 0.05 * scale

        def close(a, b, tol, msg):
            torch.testing.assert_close(a.float().cpu(), b.float().cpu(), atol=tol, rtol=0.0, msg=msg)

        close(eager, reference, device_tol, f"{phase}: eager dispatch vs the fp32 CPU reference")
        close(out, reference, device_tol, f"{phase}: compiled program vs the fp32 CPU reference")
        close(warm, reference, device_tol, f"{phase}: warm compiled program vs the fp32 CPU reference")
        close(out, eager, pair_tol, f"{phase}: compiled program vs the same body dispatched eagerly")

        # The op mutates the cache it was handed; through the program, that write must land in
        # the caller's tensor, in the block the block table names, as it does on the CPU.
        graph_cache, reference_cache = graph_in[1], reference_in[1]
        cache_tol = 0.05 * float(reference_cache.abs().max())
        close(graph_cache, reference_cache, cache_tol, f"kv_cache after {phase}")
        self.assertGreater(graph_cache.float().cpu().abs().sum().item(), 0.0, f"kv_cache untouched by {phase}")

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_causal_attention_prefill_in_graph(self, dtype):
        self._check("prefill", dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_causal_attention_decode_in_graph(self, dtype):
        self._check("decode", dtype)


instantiate_device_type_tests(TestAttentionInGraph, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
