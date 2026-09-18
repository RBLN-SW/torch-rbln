# Owner(s): ["module: PrivateUse1"]

"""The rbln_custom_ops attention kernels through eager dispatch, against a CPU fp32 reference.

Each op has a closed form -- append k/v at the position ``seq`` names, attend over the valid part
of that block, matmul with the values -- so the same computation on the CPU in fp32 is an
independent oracle. It is written here rather than borrowed: rebel_compiler registers these four
with a stub body (``torch.empty_like(q)``), so there is nothing to call.

The KV cache the op mutates is compared the same way, which pins *where* the write landed.

See ``test_custom_op_compile.py`` for the same family inside a larger compiled program.
"""

import math

import pytest
import rebel  # noqa: F401  -- defines the rbln_custom_ops schemas these tests call
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


MAX_SEQ_LENGTH = 8192  # one block; block_size == the whole cache
SEQ_LEN = 256  # prompt length, and the decode phase's existing context
NUM_Q_GROUPS, NUM_KV_HEADS, HEAD_DIM = 4, 8, 64
SCALE = torch.tensor(1.0 / math.sqrt(HEAD_DIM))

# Fractions of the reference's own scale, as in test_custom_op_compile.py. The device's 16-bit
# compute formats move the result by well under a percent; reading the wrong part of the cache,
# or writing the new token to the wrong slot, moves it by its own scale.
OUTPUT_TOL_FRACTION = 0.1
CACHE_TOL_FRACTION = 0.05


def _close(actual, reference, tol, what):
    """assert_close, naming the comparison above its diff summary (a plain ``msg=`` replaces it)."""
    torch.testing.assert_close(
        actual.float().cpu(),
        reference,
        atol=tol,
        rtol=0.0,
        msg=lambda default: f"{what}\n{default}",
    )


def _attend(q, kcache, vcache, additive_mask):
    """softmax(q @ k^T * scale + mask) @ v, all fp32, on one block's cache."""
    weights = torch.matmul(q * SCALE.float(), kcache.transpose(-1, -2)) + additive_mask
    return torch.matmul(torch.softmax(weights, dim=-1), vcache)


@pytest.mark.test_set_ci
# Compile-heavy; keep the RBLN compile cache across tests (skip per-test dynamo reset).
@pytest.mark.no_dynamo_reset
class TestCustomKernelRBLN(TestCase):
    rbln_device = torch.device("rbln:0")

    def _assert_matches(self, out, reference, dtype, name):
        """Metadata contract, then the numbers against the fp32 CPU reference."""
        self.assertEqual(out.shape, reference.shape)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.device.type, "rbln")

        tol = OUTPUT_TOL_FRACTION * float(reference.abs().max())
        _close(out, reference, tol, f"{name}: output vs the fp32 CPU reference")

    def _assert_cache_written(self, cache, reference_cache, name):
        """The op declares `mutates_args`; check the write landed where it belongs."""
        tol = CACHE_TOL_FRACTION * float(reference_cache.abs().max())
        _close(cache, reference_cache, tol, f"{name} after the call")

    # Prefill fills an empty cache from slot 0; decode appends one token to a cache that already
    # holds SEQ_LEN of context, one block per batch item so batch 1 lands outside block 0.
    def _prefill_inputs(self, dtype):
        batch = 1  # the prefill kernel is specialized for batch size 1
        shape = [batch, NUM_KV_HEADS, 1, SEQ_LEN, HEAD_DIM]
        return {
            "q": torch.randn([batch, NUM_KV_HEADS, NUM_Q_GROUPS, SEQ_LEN, HEAD_DIM], dtype=dtype),
            "k": torch.randn(shape, dtype=dtype),
            "v": torch.randn(shape, dtype=dtype),
            "kcache": torch.zeros([batch, NUM_KV_HEADS, 1, MAX_SEQ_LENGTH, HEAD_DIM], dtype=dtype),
            "vcache": torch.zeros([batch, NUM_KV_HEADS, 1, MAX_SEQ_LENGTH, HEAD_DIM], dtype=dtype),
            "seq": torch.tensor([[0]] * batch, dtype=torch.int32),
            "block_table": torch.tensor([0] * batch, dtype=torch.int16),
        }

    def _decode_inputs(self, dtype):
        batch = 2  # exercises the multi-batch decode path and a non-zero block index
        shape = [batch, NUM_KV_HEADS, 1, 1, HEAD_DIM]
        kcache = torch.zeros([batch, NUM_KV_HEADS, 1, MAX_SEQ_LENGTH, HEAD_DIM], dtype=dtype)
        vcache = torch.zeros([batch, NUM_KV_HEADS, 1, MAX_SEQ_LENGTH, HEAD_DIM], dtype=dtype)
        cached = (batch, NUM_KV_HEADS, 1, SEQ_LEN, HEAD_DIM)
        kcache[:, :, :, :SEQ_LEN, :] = torch.randn(cached, dtype=dtype)
        vcache[:, :, :, :SEQ_LEN, :] = torch.randn(cached, dtype=dtype)
        return {
            "q": torch.randn([batch, NUM_KV_HEADS, NUM_Q_GROUPS, 1, HEAD_DIM], dtype=dtype),
            "k": torch.randn(shape, dtype=dtype),
            "v": torch.randn(shape, dtype=dtype),
            "kcache": kcache,
            "vcache": vcache,
            "seq": torch.tensor([[SEQ_LEN]] * batch, dtype=torch.int32),
            "block_table": torch.tensor([[b] for b in range(batch)], dtype=torch.int16),
        }

    def _on_device(self, inputs):
        return {name: value.to(self.rbln_device) for name, value in inputs.items()}

    # ``mask`` is the op's 0/1 validity mask; None means causal.
    def _prefill_reference(self, inputs, mask):
        kcache, vcache = inputs["kcache"].float().clone(), inputs["vcache"].float().clone()
        block, start = int(inputs["block_table"][0]), int(inputs["seq"][0][0])
        end = start + SEQ_LEN
        kcache[block, :, :, start:end, :] = inputs["k"].float()[0]
        vcache[block, :, :, start:end, :] = inputs["v"].float()[0]

        if mask is None:  # causal: query at `start + i` sees keys up to it
            rows = torch.arange(SEQ_LEN).unsqueeze(1) + start
            additive = torch.where(torch.arange(MAX_SEQ_LENGTH).unsqueeze(0) <= rows, 0.0, -torch.inf)
        else:
            additive = torch.where(mask.float() > 0, 0.0, -torch.inf)[0]

        out = _attend(inputs["q"].float()[0], kcache[block], vcache[block], additive)
        return out.unsqueeze(0), kcache.to(inputs["kcache"].dtype).float(), vcache.to(inputs["vcache"].dtype).float()

    def _decode_reference(self, inputs, mask):
        kcache, vcache = inputs["kcache"].float().clone(), inputs["vcache"].float().clone()
        outs = []
        for item in range(inputs["q"].shape[0]):
            block, position = int(inputs["block_table"][item][0]), int(inputs["seq"][item][0])
            kcache[block, :, :, position, :] = inputs["k"].float()[item, :, :, 0, :]
            vcache[block, :, :, position, :] = inputs["v"].float()[item, :, :, 0, :]

            if mask is None:  # causal: the new token sees the whole history and itself
                additive = torch.where(torch.arange(MAX_SEQ_LENGTH) <= position, 0.0, -torch.inf)
            else:
                additive = torch.where(mask.float()[item] > 0, 0.0, -torch.inf)

            outs.append(_attend(inputs["q"].float()[item], kcache[block], vcache[block], additive))
        return (
            torch.stack(outs, 0),
            kcache.to(inputs["kcache"].dtype).float(),
            vcache.to(inputs["vcache"].dtype).float(),
        )

    def _check(self, name, out, device_inputs, reference):
        expected, kcache_reference, vcache_reference = reference
        self._assert_matches(out, expected, device_inputs["q"].dtype, name)
        self._assert_cache_written(device_inputs["kcache"], kcache_reference, f"{name}: kcache")
        self._assert_cache_written(device_inputs["vcache"], vcache_reference, f"{name}: vcache")

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_prefill(self, dtype):
        inputs = self._prefill_inputs(dtype)
        mask = torch.ones([1, SEQ_LEN, MAX_SEQ_LENGTH], dtype=dtype).tril()
        mask = mask.view([1, 1, 1, SEQ_LEN, MAX_SEQ_LENGTH])
        reference = self._prefill_reference(inputs, mask)

        device = self._on_device(inputs)
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_attn_prefill(
                device["q"],
                device["k"],
                device["v"],
                mask.to(self.rbln_device),
                device["kcache"],
                device["vcache"],
                device["seq"],
                SCALE,
                device["block_table"],
                MAX_SEQ_LENGTH,
            )
        self._check("paged_attn_prefill", out, device, reference)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_decode(self, dtype):
        inputs = self._decode_inputs(dtype)
        batch = inputs["q"].shape[0]
        mask = torch.zeros([batch, 1, 1, 1, MAX_SEQ_LENGTH], dtype=dtype)
        mask[:, 0, 0, 0, : SEQ_LEN + 1] = 1.0  # the existing context plus the new token
        reference = self._decode_reference(inputs, mask)

        device = self._on_device(inputs)
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_attn_decode(
                device["q"],
                device["k"],
                device["v"],
                mask.to(self.rbln_device),
                device["kcache"],
                device["vcache"],
                device["seq"],
                SCALE,
                device["block_table"],
                MAX_SEQ_LENGTH,
            )
        self._check("paged_attn_decode", out, device, reference)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_prefill(self, dtype):
        inputs = self._prefill_inputs(dtype)
        reference = self._prefill_reference(inputs, None)

        device = self._on_device(inputs)
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
                device["q"],
                device["k"],
                device["v"],
                device["kcache"],
                device["vcache"],
                device["seq"],
                SCALE,
                device["block_table"],
                MAX_SEQ_LENGTH,
                False,  # is_bidirectional; always causal here, optional mask omitted
            )
        self._check("paged_causal_attn_prefill", out, device, reference)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_decode(self, dtype):
        inputs = self._decode_inputs(dtype)
        reference = self._decode_reference(inputs, None)

        device = self._on_device(inputs)
        with torch.no_grad():
            out = torch.ops.rbln_custom_ops.paged_causal_attn_decode(
                device["q"],
                device["k"],
                device["v"],
                device["kcache"],
                device["vcache"],
                device["seq"],
                SCALE,
                device["block_table"],
                MAX_SEQ_LENGTH,
            )
        self._check("paged_causal_attn_decode", out, device, reference)

    # flash_attention_naive: the reference arrives with the op's single-sourcing in rebel_compiler.
    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_prefill(self, dtype):
        self.skipTest("pending flash_attention_naive single-sourcing in rebel_compiler")

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_decode(self, dtype):
        self.skipTest("pending flash_attention_naive single-sourcing in rebel_compiler")


instantiate_device_type_tests(TestCustomKernelRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
