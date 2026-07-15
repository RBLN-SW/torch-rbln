# Owner(s): ["module: PrivateUse1"]

"""User-level wiring/contract tests for the rbln_custom_ops attention kernels.

The ops are defined and numerically verified in rebel_compiler; torch-rbln owns
only the PrivateUse1 eager dispatch path. Each test drives the op on the RBLN
device and checks the contract — shape/dtype/device, declared in-place mutation,
and a finite/non-degenerate result — not numerical correctness.
"""

import math

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
# Compile-heavy; keep the RBLN compile cache across tests (skip per-test dynamo reset).
@pytest.mark.no_dynamo_reset
class TestCustomKernelRBLN(TestCase):
    rbln_device = torch.device("rbln:0")

    # ------------------------------------------------------------------
    # Shared contract assertions
    # ------------------------------------------------------------------
    def _assert_wellformed(self, out, reference, dtype):
        """Output metadata contract + a coarse non-degenerate sanity check."""
        self.assertEqual(out.shape, reference.shape)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.device.type, "rbln")

        host = out.cpu().to(torch.float32)
        self.assertTrue(torch.isfinite(host).all(), "output contains NaN/Inf")
        self.assertGreater(host.abs().sum().item(), 0.0, "output is identically zero (degenerate)")

    def _assert_mutated(self, before_cpu, after_cpu, name):
        """The op declares `mutates_args`; confirm the write actually landed."""
        self.assertFalse(torch.equal(before_cpu, after_cpu), f"{name} was not mutated in place")

    # ------------------------------------------------------------------
    # paged_attn
    # ------------------------------------------------------------------
    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_prefill_wiring(self, dtype):
        # Prefill kernel is specialized for batch size 1.
        batch_size = 1
        max_seq_length = 8192
        seq_len = 256
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        q = torch.randn([batch_size, num_kv_heads, num_q_heads, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        k = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        v = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        k_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype).to(self.rbln_device)
        v_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype).to(self.rbln_device)

        base_mask = torch.ones([batch_size, seq_len, max_seq_length], dtype=dtype).tril()
        mask = base_mask.view([batch_size, 1, 1, seq_len, max_seq_length]).to(self.rbln_device)

        seq = torch.tensor([[0]] * batch_size, dtype=torch.int32).to(self.rbln_device)
        scale = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = 8192
        block_table = torch.tensor([0] * batch_size, dtype=torch.int16).to(self.rbln_device)

        k_cache_before = k_cache.cpu().clone()
        v_cache_before = v_cache.cpu().clone()
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_attn_prefill(
                q, k, v, mask, k_cache, v_cache, seq, scale, block_table, block_size
            )

        self._assert_wellformed(out, q, dtype)
        self._assert_mutated(k_cache_before, k_cache.cpu(), "kcache")
        self._assert_mutated(v_cache_before, v_cache.cpu(), "vcache")

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_decode_wiring(self, dtype):
        # Batch size 2 exercises the multi-batch decode path.
        batch_size = 2
        max_seq_length = 8192
        seq_len = 256  # existing context length already in the cache
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        q = torch.randn([batch_size, num_kv_heads, num_q_heads, 1, head_dim], dtype=dtype).to(self.rbln_device)
        k = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype).to(self.rbln_device)
        v = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype).to(self.rbln_device)

        # Pre-populate the cache up to `seq_len` to simulate a processed prompt.
        k_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        k_cache[:, :, :, :seq_len, :] = torch.randn(slice_shape, dtype=dtype)
        v_cache[:, :, :, :seq_len, :] = torch.randn(slice_shape, dtype=dtype)
        k_cache = k_cache.to(self.rbln_device)
        v_cache = v_cache.to(self.rbln_device)

        # Mask marks all valid tokens (existing context + the new token).
        mask = torch.zeros([batch_size, 1, 1, 1, max_seq_length], dtype=dtype)
        mask[:, 0, 0, 0, : seq_len + 1] = 1.0
        mask = mask.to(self.rbln_device)

        scale = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = max_seq_length
        seq = torch.tensor([[seq_len], [seq_len]], dtype=torch.int32).to(self.rbln_device)
        block_table = torch.tensor([[0], [1]], dtype=torch.int16).to(self.rbln_device)

        k_cache_before = k_cache.cpu().clone()
        v_cache_before = v_cache.cpu().clone()
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_attn_decode(
                q, k, v, mask, k_cache, v_cache, seq, scale, block_table, block_size
            )

        self._assert_wellformed(out, q, dtype)
        self._assert_mutated(k_cache_before, k_cache.cpu(), "kcache")
        self._assert_mutated(v_cache_before, v_cache.cpu(), "vcache")

    # ------------------------------------------------------------------
    # paged_causal_attn
    # ------------------------------------------------------------------
    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_prefill_wiring(self, dtype):
        batch_size = 1
        max_seq_length = 8192
        seq_len = 256
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        q = torch.randn([batch_size, num_kv_heads, num_q_heads, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        k = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        v = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        k_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype).to(self.rbln_device)
        v_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype).to(self.rbln_device)

        seq = torch.tensor([[0]] * batch_size, dtype=torch.int32).to(self.rbln_device)
        scale = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = 8192
        block_table = torch.tensor([0] * batch_size, dtype=torch.int16).to(self.rbln_device)
        is_bidirectional = False  # always causal here; optional mask omitted

        k_cache_before = k_cache.cpu().clone()
        v_cache_before = v_cache.cpu().clone()
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
                q, k, v, k_cache, v_cache, seq, scale, block_table, block_size, is_bidirectional
            )

        self._assert_wellformed(out, q, dtype)
        self._assert_mutated(k_cache_before, k_cache.cpu(), "kcache")
        self._assert_mutated(v_cache_before, v_cache.cpu(), "vcache")

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_decode_wiring(self, dtype):
        batch_size = 2
        max_seq_length = 8192
        seq_len = 256
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        q = torch.randn([batch_size, num_kv_heads, num_q_heads, 1, head_dim], dtype=dtype).to(self.rbln_device)
        k = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype).to(self.rbln_device)
        v = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype).to(self.rbln_device)

        k_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        k_cache[:, :, :, :seq_len, :] = torch.randn(slice_shape, dtype=dtype)
        v_cache[:, :, :, :seq_len, :] = torch.randn(slice_shape, dtype=dtype)
        k_cache = k_cache.to(self.rbln_device)
        v_cache = v_cache.to(self.rbln_device)

        scale = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = max_seq_length
        seq = torch.tensor([[seq_len], [seq_len]], dtype=torch.int32).to(self.rbln_device)
        block_table = torch.tensor([[0], [1]], dtype=torch.int16).to(self.rbln_device)

        k_cache_before = k_cache.cpu().clone()
        v_cache_before = v_cache.cpu().clone()
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_causal_attn_decode(
                q, k, v, k_cache, v_cache, seq, scale, block_table, block_size
            )

        self._assert_wellformed(out, q, dtype)
        self._assert_mutated(k_cache_before, k_cache.cpu(), "kcache")
        self._assert_mutated(v_cache_before, v_cache.cpu(), "vcache")

    # ------------------------------------------------------------------
    # flash_attention_naive (unified KV cache: axis 0 stacks [K, V])
    # ------------------------------------------------------------------
    def _run_flash_attention_naive(self, dtype, *, seq_len: int, op_name: str) -> None:
        """Shared body for flash_attention_naive prefill/decode wiring tests."""
        num_kv_heads = 8  # H
        num_q_groups = 4  # G
        head_dim = 64  # D — must be a multiple of 64 for the kernel
        num_blocks = 1  # B
        partition_size = 256  # P

        is_decode = "decode" in op_name
        # Prefill writes the full prompt from slot 0; decode appends one token to
        # a partially-filled cache. The mask covers the valid region (history + new).
        prefilled = 32 if is_decode else 0
        valid_len = prefilled + seq_len

        q = torch.randn([1, num_kv_heads, num_q_groups, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        k = torch.randn([1, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        v = torch.randn([1, num_kv_heads, 1, seq_len, head_dim], dtype=dtype).to(self.rbln_device)
        kv_cache = torch.randn([2, num_blocks, num_kv_heads, 1, partition_size, head_dim], dtype=dtype).to(
            self.rbln_device
        )

        mask = torch.zeros([1, 1, 1, seq_len, partition_size], dtype=dtype)
        mask[:, :, :, :, :valid_len] = 1
        mask = mask.to(self.rbln_device)

        scale = torch.tensor(1.0 / math.sqrt(head_dim))
        seq_idx = torch.tensor([[prefilled]], dtype=torch.int32).to(self.rbln_device)
        # prefill: [num_partitions], decode: [batch, num_partitions]
        block_tables_cpu = torch.tensor([[0]], dtype=torch.int16) if is_decode else torch.tensor([0], dtype=torch.int16)
        block_tables = block_tables_cpu.to(self.rbln_device)
        slot_mapping = torch.arange(seq_len, dtype=torch.int32).to(self.rbln_device)

        op = getattr(torch.ops.rbln_custom_ops, op_name)
        kv_cache_before = kv_cache.cpu().clone()
        with torch.no_grad():
            out = op(q, k, v, kv_cache, mask, scale, seq_idx, block_tables, slot_mapping)

        self._assert_wellformed(out, q, dtype)
        self._assert_mutated(kv_cache_before, kv_cache.cpu(), "kv_cache")

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_prefill_wiring(self, dtype):
        self.skipTest("pending flash_attention_naive single-sourcing in rebel_compiler")
        self._run_flash_attention_naive(dtype, seq_len=256, op_name="flash_attention_naive_prefill")

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_decode_wiring(self, dtype):
        self.skipTest("pending flash_attention_naive single-sourcing in rebel_compiler")
        self._run_flash_attention_naive(dtype, seq_len=1, op_name="flash_attention_naive_decode")


instantiate_device_type_tests(TestCustomKernelRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
