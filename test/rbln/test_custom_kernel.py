# Owner(s): ["module: PrivateUse1"]

"""
Test suite for RBLN custom kernel implementations.

This test suite covers the following custom kernels:
1. rbln_custom_ops::paged_attn_prefill
2. rbln_custom_ops::paged_attn_decode
3. rbln_custom_ops::paged_causal_attn_prefill
4. rbln_custom_ops::paged_causal_attn_decode
5. rbln_custom_ops::flash_attention_naive_prefill
6. rbln_custom_ops::flash_attention_naive_decode
"""

import math
from typing import Optional

import pytest
import torch
from torch import Tensor
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


# The following `paged_attn` functions are simplified mock implementations
# of the actual operators found in the 'optimum_rbln' library. They are
# defined here to enable unit testing of this module's functionality
# without requiring a full dependency on 'optimum_rbln'.
@torch.library.custom_op(
    "rbln_custom_ops::paged_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    """
    Implements the prefill step for paged attention as a custom PyTorch operator.

    This operator computes the attention for the initial prompt (prefill) and populates
    the key-value cache in a paged memory layout. The K/V cache tensors are
    mutated in-place, as declared in the `mutates_args` argument of the decorator.

    Note:
        This implementation currently only supports a batch size of 1.

    Args:
        q (Tensor): The query tensor for the prefill sequence.
        k (Tensor): The key tensor for the prefill sequence.
        v (Tensor): The value tensor for the prefill sequence.
        mask (Tensor): The attention mask, typically a causal mask, to apply to the attention scores.
        kcache (Tensor): The key cache tensor where the keys will be stored. This tensor is mutated in-place.
        vcache (Tensor): The value cache tensor where the values will be stored. This tensor is mutated in-place.
        seq (Tensor): A tensor containing sequence length information for the batch.
        scale (Tensor): A scalar tensor for the attention scaling factor (typically 1/sqrt(head_dim)).
        block_table (Tensor): The table that maps logical sequence blocks to physical blocks in the cache.
        block_size (int): The size of a single block in the paged K/V cache.

    Returns:
        Tensor: The attention output tensor, which has the same shape as the query tensor 'q'.
    """
    partition = kcache.size(-2)
    seq_len = q.size(-2)
    s = seq[0][0]
    e = s + seq_len
    block = 0

    Dk = k.size(-1)
    Dv = v.size(-1)
    assert Dk <= kcache.size(-1) and Dv <= vcache.size(-1)

    kcache[block].unsqueeze(0)[:, :, :, s:e, :Dk].copy_(k[:, :, :, :, :Dk])
    vcache[block].unsqueeze(0)[:, :, :, s:e, :Dv].copy_(v[:, :, :, :, :Dv])
    k_state = kcache[block].unsqueeze(0)[..., :Dk]
    v_state = vcache[block].unsqueeze(0)[..., :Dk]

    attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
    causal_mask = torch.where(mask[:, :, :, :, :partition] > 0, 0.0, -float("inf"))
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_state)
    return attn_output


@paged_attn_prefill.register_fake
def paged_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    """
    Fake tensor implementation for the 'paged_attn_prefill' custom operator.

    This function is executed when the operator is called under a FakeTensorMode
    context, such as during `torch.compile` tracing. Its sole purpose is to
    compute and return the metadata (shape, dtype, device, etc.) of the
    output tensor without performing any actual computation.

    Args:
        q (Tensor): The query tensor, provided as a FakeTensor.
        k (Tensor): The key tensor, provided as a FakeTensor.
        v (Tensor): The value tensor, provided as a FakeTensor.
        mask (Tensor): The attention mask, provided as a FakeTensor.
        kcache (Tensor): The key cache, provided as a FakeTensor.
        vcache (Tensor): The value cache, provided as a FakeTensor.
        seq (Tensor): A tensor with sequence information, as a FakeTensor.
        scale (Tensor): The attention scaling factor, as a FakeTensor.
        block_table (Tensor): The block table, provided as a FakeTensor.
        block_size (int): The size of a single block.

    Returns:
        Tensor: A new FakeTensor representing the operator's output, which has
                the same properties as the query tensor 'q'.
    """
    # For standard attention operations, the output tensor's shape, dtype, and
    # device are identical to those of the query tensor 'q'. `torch.empty_like`
    # is the canonical way to create a new tensor with the same metadata.
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    """
    Implements the decode step for paged attention as a custom PyTorch operator.

    This operator processes a batch of sequences for a single decode step, handling
    one new token per sequence. It appends the new key and value to their
    respective caches and computes attention for the new query against the entire
    key history stored in the cache. The K/V cache tensors are mutated in-place,
    as declared in `mutates_args`.

    Args:
        q (Tensor): The query tensor for the single new token, with a sequence length of 1.
        k (Tensor): The key tensor for the single new token (sequence length 1).
        v (Tensor): The value tensor for the single new token (sequence length 1).
        mask (Tensor): The attention mask, typically a padding mask for the full K/V context length.
        kcache (Tensor): The key cache to be updated. This tensor is mutated in-place.
        vcache (Tensor): The value cache to be updated. This tensor is mutated in-place.
        seq (Tensor): A tensor containing the current sequence length of each item in the batch.
        scale (Tensor): A scalar tensor for the attention scaling factor (typically 1/sqrt(head_dim)).
        block_table (Tensor): The table that maps logical sequence blocks to physical blocks in the cache.
        block_size (int): The size of a single block in the paged K/V cache.

    Returns:
        Tensor: The attention output tensor for the single query token, having the same shape as 'q'.
    """
    batch_size, num_layers, num_q_heads, _, head_dim = q.shape
    _, _, num_kv_heads, max_seq_len, _ = kcache.shape
    partition = kcache.size(-2)

    output_per_batch_item = []

    for b_idx in range(batch_size):
        q_item = q[b_idx]
        k_item = k[b_idx]
        v_item = v[b_idx]

        current_seq_len = seq[b_idx, 0].item()

        if current_seq_len < max_seq_len:
            kcache[b_idx, :, :, current_seq_len, :] = k_item.squeeze(-2)
            vcache[b_idx, :, :, current_seq_len, :] = v_item.squeeze(-2)

        k_state = kcache[b_idx].unsqueeze(0)[..., :head_dim]
        v_state = vcache[b_idx].unsqueeze(0)[..., :head_dim]

        if num_q_heads != num_kv_heads:
            q_item = q_item.view(num_layers, num_kv_heads, num_q_heads // num_kv_heads, 1, head_dim)
            k_state = k_state.unsqueeze(2)
            v_state = v_state.unsqueeze(2)

        attn_weights_item = torch.matmul(q_item, k_state.transpose(-1, -2)) * scale
        causal_mask = torch.where(mask[b_idx, :, :, :, :partition] > 0, 0.0, -float("inf"))
        attn_weights_item = attn_weights_item + causal_mask
        attn_softmax = torch.nn.functional.softmax(attn_weights_item, dim=-1, dtype=torch.float32).to(q.dtype)
        output_item = torch.matmul(attn_softmax, v_state)

        if num_q_heads != num_kv_heads:
            output_item = output_item.view(num_layers, num_q_heads, 1, head_dim)

        output_per_batch_item.append(output_item)

    final_output = torch.stack(output_per_batch_item, dim=0)

    return final_output


@paged_attn_decode.register_fake
def paged_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    """
    Fake tensor implementation for the 'paged_attn_decode' custom operator.

    This function is executed when the operator is called under a FakeTensorMode
    context (e.g., during torch.compile tracing). Its purpose is to compute the
    metadata (shape, dtype, device) of the output tensor without performing
    any actual computation.

    Args:
        q (Tensor): The query tensor for the single new token, as a FakeTensor.
        k (Tensor): The key tensor for the single new token, as a FakeTensor.
        v (Tensor): The value tensor for the single new token, as a FakeTensor.
        mask (Tensor): The attention mask, as a FakeTensor.
        kcache (Tensor): The key cache, as a FakeTensor.
        vcache (Tensor): The value cache, as a FakeTensor.
        seq (Tensor): A tensor with sequence information, as a FakeTensor.
        scale (Tensor): The attention scaling factor, as a FakeTensor.
        block_table (Tensor): The block table, as a FakeTensor.
        block_size (int): The size of a single block.

    Returns:
        Tensor: A new FakeTensor representing the operator's output, with the
                same properties as the query tensor 'q'.
    """
    # The output of an attention decode operation has the same metadata (shape,
    # dtype, device) as the single-token query tensor 'q'. `torch.empty_like`
    # is the standard way to create such a tensor.
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Implements the prefill step for paged causal attention as a custom PyTorch operator.

    This operator computes the attention for the initial prompt (prefill) and populates
    the key-value cache in a paged memory layout. The K/V cache tensors are
    mutated in-place, as declared in the `mutates_args` argument of the decorator.

    Key differences from paged_attn_prefill:
    - Supports bidirectional attention via is_bidirectional flag
    - Optional mask parameter for position-based masking

    Note:
        This implementation currently only supports a batch size of 1.

    Args:
        q (Tensor): The query tensor for the prefill sequence.
        k (Tensor): The key tensor for the prefill sequence.
        v (Tensor): The value tensor for the prefill sequence.
        kcache (Tensor): The key cache tensor where the keys will be stored. This tensor is mutated in-place.
        vcache (Tensor): The value cache tensor where the values will be stored. This tensor is mutated in-place.
        seq (Tensor): A tensor containing sequence length information for the batch.
        scale (Tensor): A scalar tensor for the attention scaling factor (typically 1/sqrt(head_dim)).
        block_table (Tensor): The table that maps logical sequence blocks to physical blocks in the cache.
        block_size (int): The size of a single block in the paged K/V cache.
        is_bidirectional (bool): Whether the attention is bidirectional at current sequence position.
        mask (Tensor, optional): Optional attention mask for position-based masking.

    Returns:
        Tensor: The attention output tensor, which has the same shape as the query tensor 'q'.
    """
    partition = kcache.size(-2)
    seq_len = q.size(-2)
    s = seq[0][0]
    e = s + seq_len
    block = 0

    Dk = k.size(-1)
    Dv = v.size(-1)
    assert Dk <= kcache.size(-1) and Dv <= vcache.size(-1)

    kcache[block].unsqueeze(0)[:, :, :, s:e, :Dk].copy_(k[:, :, :, :, :Dk])
    vcache[block].unsqueeze(0)[:, :, :, s:e, :Dv].copy_(v[:, :, :, :, :Dv])
    k_state = kcache[block].unsqueeze(0)[..., :Dk]
    v_state = vcache[block].unsqueeze(0)[..., :Dk]

    attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale

    # Apply causal mask if not bidirectional
    if not is_bidirectional:
        causal_mask = torch.tril(torch.ones(seq_len, partition, device=q.device, dtype=q.dtype))
        causal_mask = torch.where(causal_mask > 0, 0.0, -float("inf"))
        causal_mask = causal_mask.view(1, 1, 1, seq_len, partition)
        attn_weights = attn_weights + causal_mask

    # Apply optional position-based mask if provided
    if mask is not None:
        # mask shape: [batch=1, max_seq_len]
        mask_expanded = mask.view(1, 1, 1, 1, -1)
        mask_expanded = mask_expanded[:, :, :, :, :partition]
        position_mask = torch.where(mask_expanded > 0, 0.0, -float("inf"))
        attn_weights = attn_weights + position_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_state)
    return attn_output


@paged_causal_attn_prefill.register_fake
def paged_causal_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Fake tensor implementation for the 'paged_causal_attn_prefill' custom operator.

    This function is executed when the operator is called under a FakeTensorMode
    context, such as during `torch.compile` tracing. Its sole purpose is to
    compute and return the metadata (shape, dtype, device, etc.) of the
    output tensor without performing any actual computation.

    Args:
        q (Tensor): The query tensor, provided as a FakeTensor.
        k (Tensor): The key tensor, provided as a FakeTensor.
        v (Tensor): The value tensor, provided as a FakeTensor.
        kcache (Tensor): The key cache, provided as a FakeTensor.
        vcache (Tensor): The value cache, provided as a FakeTensor.
        seq (Tensor): A tensor with sequence information, as a FakeTensor.
        scale (Tensor): The attention scaling factor, as a FakeTensor.
        block_table (Tensor): The block table, provided as a FakeTensor.
        block_size (int): The size of a single block.
        is_bidirectional (bool): Whether the attention is bidirectional.
        mask (Tensor, optional): Optional attention mask, as a FakeTensor.

    Returns:
        Tensor: A new FakeTensor representing the operator's output, which has
                the same properties as the query tensor 'q'.
    """
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Implements the decode step for paged causal attention as a custom PyTorch operator.

    This operator processes a batch of sequences for a single decode step, handling
    one new token per sequence. It appends the new key and value to their
    respective caches and computes attention for the new query against the entire
    key history stored in the cache. The K/V cache tensors are mutated in-place,
    as declared in `mutates_args`.

    Key differences from paged_attn_decode:
    - Optional mask parameter for position-based masking

    Args:
        q (Tensor): The query tensor for the single new token, with a sequence length of 1.
        k (Tensor): The key tensor for the single new token (sequence length 1).
        v (Tensor): The value tensor for the single new token (sequence length 1).
        kcache (Tensor): The key cache to be updated. This tensor is mutated in-place.
        vcache (Tensor): The value cache to be updated. This tensor is mutated in-place.
        seq (Tensor): A tensor containing the current sequence length of each item in the batch.
        scale (Tensor): A scalar tensor for the attention scaling factor (typically 1/sqrt(head_dim)).
        block_table (Tensor): The table that maps logical sequence blocks to physical blocks in the cache.
        block_size (int): The size of a single block in the paged K/V cache.
        mask (Tensor, optional): Optional attention mask for position-based masking.

    Returns:
        Tensor: The attention output tensor for the single query token, having the same shape as 'q'.
    """
    batch_size, num_layers, num_q_heads, _, head_dim = q.shape
    _, _, num_kv_heads, max_seq_len, _ = kcache.shape
    partition = kcache.size(-2)

    output_per_batch_item = []

    for b_idx in range(batch_size):
        q_item = q[b_idx]
        k_item = k[b_idx]
        v_item = v[b_idx]

        current_seq_len = seq[b_idx, 0].item()

        if current_seq_len < max_seq_len:
            kcache[b_idx, :, :, current_seq_len, :] = k_item.squeeze(-2)
            vcache[b_idx, :, :, current_seq_len, :] = v_item.squeeze(-2)

        k_state = kcache[b_idx].unsqueeze(0)[..., :head_dim]
        v_state = vcache[b_idx].unsqueeze(0)[..., :head_dim]

        if num_q_heads != num_kv_heads:
            q_item = q_item.view(num_layers, num_kv_heads, num_q_heads // num_kv_heads, 1, head_dim)
            k_state = k_state.unsqueeze(2)
            v_state = v_state.unsqueeze(2)

        attn_weights_item = torch.matmul(q_item, k_state.transpose(-1, -2)) * scale

        # Apply causal mask (always causal for decode)
        # In decode, we only attend to tokens up to current_seq_len + 1 (including the new token)
        # Since we're at the last position, this is effectively a valid length mask
        if mask is not None:
            # If mask is provided, use it (similar to paged_attn_decode)
            # mask shape: [batch_size, max_seq_len] -> expand to match attention weights
            mask_expanded = mask[b_idx : b_idx + 1].view(1, 1, 1, 1, -1)
            mask_expanded = mask_expanded[:, :, :, :, :partition]
            causal_mask = torch.where(mask_expanded > 0, 0.0, -float("inf"))
        else:
            # If mask is not provided, create a causal mask up to current_seq_len + 1
            valid_length = min(current_seq_len + 1, partition)
            causal_mask = torch.ones(1, partition, device=q.device, dtype=q.dtype) * float("-inf")
            causal_mask[:, :valid_length] = 0.0
            causal_mask = causal_mask.view(1, 1, 1, 1, partition)

        attn_weights_item = attn_weights_item + causal_mask

        attn_softmax = torch.nn.functional.softmax(attn_weights_item, dim=-1, dtype=torch.float32).to(q.dtype)
        output_item = torch.matmul(attn_softmax, v_state)

        if num_q_heads != num_kv_heads:
            output_item = output_item.view(num_layers, num_q_heads, 1, head_dim)

        output_per_batch_item.append(output_item)

    final_output = torch.stack(output_per_batch_item, dim=0)

    return final_output


@paged_causal_attn_decode.register_fake
def paged_causal_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Fake tensor implementation for the 'paged_causal_attn_decode' custom operator.

    This function is executed when the operator is called under a FakeTensorMode
    context (e.g., during torch.compile tracing). Its purpose is to compute the
    metadata (shape, dtype, device) of the output tensor without performing
    any actual computation.

    Args:
        q (Tensor): The query tensor for the single new token, as a FakeTensor.
        k (Tensor): The key tensor for the single new token, as a FakeTensor.
        v (Tensor): The value tensor for the single new token, as a FakeTensor.
        kcache (Tensor): The key cache, as a FakeTensor.
        vcache (Tensor): The value cache, as a FakeTensor.
        seq (Tensor): A tensor with sequence information, as a FakeTensor.
        scale (Tensor): The attention scaling factor, as a FakeTensor.
        block_table (Tensor): The block table, provided as a FakeTensor.
        block_size (int): The size of a single block.
        mask (Tensor, optional): Optional attention mask, as a FakeTensor.

    Returns:
        Tensor: A new FakeTensor representing the operator's output, with the
                same properties as the query tensor 'q'.
    """
    return torch.empty_like(q)


# ---------------------------------------------------------------------------
# flash_attention_naive_prefill / flash_attention_naive_decode
#
# Unified paged KV cache (kv_cache axis 0 stacks [K, V]) flash attention
# variants. The CPU reference mirrors vllm-rbln's reference
# implementation (single-partition, slice_scatter-based cache update) so the
# test exercises the same schema (q, k, v, kv_cache, mask, scale, seq_idx,
# block_tables, slot_mapping[, sinks]) that vllm-rbln registers and the
# rebel compiler expects.
# ---------------------------------------------------------------------------


def _flash_attn_naive_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Tensor,
    mask: Tensor,
    scale: Tensor,
    seq_idx: Tensor,
    block_tables: Tensor,
) -> Tensor:
    """Single-partition reference matching vllm-rbln's flash_attention_naive reference.

    Writes new K/V slices into the cache for block ``block_tables[0]`` at
    offset ``seq_idx[0][0]``, then computes causal-masked attention over the
    partition. ``kv_cache`` is mutated in place for both K (axis 0) and V
    (axis 1).
    """
    partition = kv_cache.size(-2)
    seq_len = q.size(-2)
    s = seq_idx[0][0]
    e = s + seq_len
    block = block_tables.reshape(-1)[0].to(torch.int32)

    k_state = kv_cache[0][block].unsqueeze(0).slice_scatter(k, dim=3, start=s, end=e)
    v_state = kv_cache[1][block].unsqueeze(0).slice_scatter(v, dim=3, start=s, end=e)
    kv_cache[0][block] = k_state.squeeze(0)
    kv_cache[1][block] = v_state.squeeze(0)

    attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
    causal_mask = torch.where(mask[:, :, :, :, :partition] > 0, 0.0, -float("inf")).to(attn_weights.dtype)
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn_weights, v_state)


@torch.library.custom_op("rbln_custom_ops::flash_attention_naive_prefill", mutates_args=["kv_cache"])
def flash_attention_naive_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Tensor,
    mask: Tensor,
    scale: Tensor,
    seq_idx: Tensor,
    block_tables: Tensor,
    slot_mapping: Tensor,
    sinks: Optional[Tensor] = None,
) -> Tensor:
    """CPU reference for 'flash_attention_naive_prefill'.

    ``slot_mapping`` and ``sinks`` are accepted to match the kernel signature
    but are not used by this single-partition reference.
    """
    return _flash_attn_naive_reference(q, k, v, kv_cache, mask, scale, seq_idx, block_tables)


@flash_attention_naive_prefill.register_fake
def flash_attention_naive_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Tensor,
    mask: Tensor,
    scale: Tensor,
    seq_idx: Tensor,
    block_tables: Tensor,
    slot_mapping: Tensor,
    sinks: Optional[Tensor] = None,
) -> Tensor:
    """Fake-tensor implementation used during ``torch.compile`` tracing."""
    return torch.empty_like(q)


@torch.library.custom_op("rbln_custom_ops::flash_attention_naive_decode", mutates_args=["kv_cache"])
def flash_attention_naive_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Tensor,
    mask: Tensor,
    scale: Tensor,
    seq_idx: Tensor,
    block_tables: Tensor,
    slot_mapping: Tensor,
    sinks: Optional[Tensor] = None,
) -> Tensor:
    """CPU reference for 'flash_attention_naive_decode'. Same shape contract as prefill."""
    return _flash_attn_naive_reference(q, k, v, kv_cache, mask, scale, seq_idx, block_tables)


@flash_attention_naive_decode.register_fake
def flash_attention_naive_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Tensor,
    mask: Tensor,
    scale: Tensor,
    seq_idx: Tensor,
    block_tables: Tensor,
    slot_mapping: Tensor,
    sinks: Optional[Tensor] = None,
) -> Tensor:
    """Fake-tensor implementation used during ``torch.compile`` tracing."""
    return torch.empty_like(q)


@pytest.mark.test_set_ci
# In sequential test runs, the autouse fixture calls torch._dynamo.reset(),
# which clears module-level custom op registrations. This can break later
# custom op tests, so this class opts out the fixture.
@pytest.mark.no_dynamo_reset
class TestCustomKernelRBLN(TestCase):
    rtol = 1e-3
    atol = 1.2e-2

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_prefill_comparison(self, dtype):
        # --- 1. Test Configuration ---
        # Define the parameters for the attention operation and tensors.
        # This test case uses a batch size of 1, as the prefill kernel is
        # specialized for this scenario.
        batch_size = 1
        max_seq_length = 8192
        seq_len = 256
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        # --- 2. Reference CPU Tensor Creation ---
        # Create random input tensors (Query, Key, Value) for the RBLN device.
        q_cpu = torch.randn([batch_size, num_kv_heads, num_q_heads, seq_len, head_dim], dtype=dtype)
        k_cpu = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)
        v_cpu = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)

        # Initialize the Key-Value cache tensors with zeros. These will be mutated in-place.
        k_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)

        # Create a lower-triangular causal mask to prevent attention to future tokens.
        base_mask = torch.ones([batch_size, seq_len, max_seq_length], dtype=dtype).tril()
        mask_cpu = base_mask.view([batch_size, 1, 1, seq_len, max_seq_length])

        # Create placeholder tensors for other required arguments.
        seq_cpu = torch.tensor([[0]] * batch_size, dtype=torch.int32)
        scale_tensor = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = 8192
        block_table_cpu = torch.tensor([0] * batch_size, dtype=torch.int16)

        # --- 3. Create RBLN Tensors ---
        q_rbln = q_cpu.clone().to(self.rbln_device)
        k_rbln = k_cpu.clone().to(self.rbln_device)
        v_rbln = v_cpu.clone().to(self.rbln_device)
        mask_rbln = mask_cpu.clone().to(self.rbln_device)
        k_cache_rbln = k_cache_cpu.clone().to(self.rbln_device)
        v_cache_rbln = v_cache_cpu.clone().to(self.rbln_device)
        seq_rbln = seq_cpu.clone().to(self.rbln_device)
        block_table_rbln = block_table_cpu.clone().to(self.rbln_device)

        # Pack arguments for both RBLN and CPU operator calls.
        rbln_args = (
            q_rbln,
            k_rbln,
            v_rbln,
            mask_rbln,
            k_cache_rbln,
            v_cache_rbln,
            seq_rbln,
            scale_tensor,
            block_table_rbln,
            block_size,
        )
        cpu_args = (
            q_cpu,
            k_cpu,
            v_cpu,
            mask_cpu,
            k_cache_cpu,
            v_cache_cpu,
            seq_cpu,
            scale_tensor,
            block_table_cpu,
            block_size,
        )

        # --- 4. Operator Execution ---
        # Execute the custom operator on both the RBLN device and the CPU.
        # `torch.no_grad()` is used as we are only testing inference.
        with torch.no_grad():
            output_rbln = torch.ops.rbln_custom_ops.paged_attn_prefill(*rbln_args)
            output_cpu = torch.ops.rbln_custom_ops.paged_attn_prefill(*cpu_args)

        # --- 5. Validation ---
        # First, verify that the output metadata (shape, dtype) is identical.
        self.assertEqual(output_rbln.shape, output_cpu.shape)
        self.assertEqual(output_rbln.dtype, output_cpu.dtype)

        # Second, verify that the numerical results are close within a tolerance.
        # This accounts for minor floating-point differences between hardware.
        self.assertEqual(output_rbln.cpu(), output_cpu, rtol=self.rtol, atol=self.atol)

        # Finally, verify that the in-place mutations of the K/V caches are also correct.
        self.assertEqual(k_cache_rbln.cpu(), k_cache_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(v_cache_rbln.cpu(), v_cache_cpu, rtol=self.rtol, atol=self.atol)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_attn_decode_comparison(self, dtype):
        # --- 1. Test Configuration ---
        # Define parameters for the decode step. This test uses a batch size of 2
        # to ensure multi-batch logic is handled correctly.
        batch_size = 2
        max_seq_length = 8192
        seq_len = 256  # Represents the length of the existing context in the cache.
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        # --- 2. Reference CPU Tensor Creation ---
        # For the decode step, Q, K, and V represent a single new token (seq length = 1).
        q_cpu = torch.randn([batch_size, num_kv_heads, num_q_heads, 1, head_dim], dtype=dtype)
        k_cpu = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype)
        v_cpu = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype)

        # Initialize K/V caches and pre-populate them with random data up to `seq_len`
        # to simulate a state where a prompt has already been processed.
        k_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        k_slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        k_cache_cpu[:, :, :, :seq_len, :] = torch.randn(k_slice_shape, dtype=dtype)
        v_slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        v_cache_cpu[:, :, :, :seq_len, :] = torch.randn(v_slice_shape, dtype=dtype)

        # The attention mask for decode handles padding. It marks all valid tokens
        # (existing context + the new token) with 1.0.
        attention_mask = torch.zeros([batch_size, 1, 1, 1, max_seq_length], dtype=dtype)
        # This loop is needed for multi-batch, but since seq_len is the same for both, it's simplified.
        attention_mask[:, 0, 0, 0, : seq_len + 1] = 1.0
        mask_cpu = attention_mask

        # Create other arguments. `seq` now indicates the current length of each sequence.
        scale_tensor = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = max_seq_length
        seq_cpu = torch.tensor([[seq_len], [seq_len]], dtype=torch.int32)
        block_table_cpu = torch.tensor([[0], [1]], dtype=torch.int16)

        # --- 3. Create RBLN Tensors ---
        q_rbln = q_cpu.clone().to(self.rbln_device)
        k_rbln = k_cpu.clone().to(self.rbln_device)
        v_rbln = v_cpu.clone().to(self.rbln_device)
        mask_rbln = mask_cpu.clone().to(self.rbln_device)
        k_cache_rbln = k_cache_cpu.clone().to(self.rbln_device)
        v_cache_rbln = v_cache_cpu.clone().to(self.rbln_device)
        seq_rbln = seq_cpu.clone().to(self.rbln_device)
        block_table_rbln = block_table_cpu.clone().to(self.rbln_device)

        # Pack arguments for both operator calls.
        rbln_args = (
            q_rbln,
            k_rbln,
            v_rbln,
            mask_rbln,
            k_cache_rbln,
            v_cache_rbln,
            seq_rbln,
            scale_tensor,
            block_table_rbln,
            block_size,
        )
        cpu_args = (
            q_cpu,
            k_cpu,
            v_cpu,
            mask_cpu,
            k_cache_cpu,
            v_cache_cpu,
            seq_cpu,
            scale_tensor,
            block_table_cpu,
            block_size,
        )

        # --- 4. Operator Execution ---
        with torch.no_grad():
            output_rbln = torch.ops.rbln_custom_ops.paged_attn_decode(*rbln_args)
            output_cpu = torch.ops.rbln_custom_ops.paged_attn_decode(*cpu_args)

        # --- 5. Validation ---
        self.assertEqual(output_rbln.shape, output_cpu.shape)
        self.assertEqual(output_rbln.dtype, output_cpu.dtype)
        self.assertEqual(output_rbln.cpu(), output_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(k_cache_rbln.cpu(), k_cache_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(v_cache_rbln.cpu(), v_cache_cpu, rtol=self.rtol, atol=self.atol)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_prefill_comparison(self, dtype):
        # --- 1. Test Configuration ---
        # Define the parameters for the causal attention prefill operation.
        # This test case uses a batch size of 1, as the prefill kernel is
        # specialized for this scenario.
        batch_size = 1
        max_seq_length = 8192
        seq_len = 256
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        # --- 2. Reference CPU Tensor Creation ---
        # Create random input tensors (Query, Key, Value) for the CPU device.
        q_cpu = torch.randn([batch_size, num_kv_heads, num_q_heads, seq_len, head_dim], dtype=dtype)
        k_cpu = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)
        v_cpu = torch.randn([batch_size, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)

        # Initialize the Key-Value cache tensors with zeros. These will be mutated in-place.
        k_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)

        # Create placeholder tensors for other required arguments.
        seq_cpu = torch.tensor([[0]] * batch_size, dtype=torch.int32)
        scale_tensor = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = 8192
        block_table_cpu = torch.tensor([0] * batch_size, dtype=torch.int16)

        # is_bidirectional is always False for causal attention
        is_bidirectional = False

        # --- 3. Create RBLN Tensors ---
        q_rbln = q_cpu.clone().to(self.rbln_device)
        k_rbln = k_cpu.clone().to(self.rbln_device)
        v_rbln = v_cpu.clone().to(self.rbln_device)
        k_cache_rbln = k_cache_cpu.clone().to(self.rbln_device)
        v_cache_rbln = v_cache_cpu.clone().to(self.rbln_device)
        seq_rbln = seq_cpu.clone().to(self.rbln_device)
        block_table_rbln = block_table_cpu.clone().to(self.rbln_device)

        # Pack arguments for both RBLN and CPU operator calls.
        # mask is not provided (optional parameter, not used in practice)
        rbln_args = (
            q_rbln,
            k_rbln,
            v_rbln,
            k_cache_rbln,
            v_cache_rbln,
            seq_rbln,
            scale_tensor,
            block_table_rbln,
            block_size,
            is_bidirectional,
        )
        cpu_args = (
            q_cpu,
            k_cpu,
            v_cpu,
            k_cache_cpu,
            v_cache_cpu,
            seq_cpu,
            scale_tensor,
            block_table_cpu,
            block_size,
            is_bidirectional,
        )

        # --- 4. Operator Execution ---
        # Execute the custom operator on both the RBLN device and the CPU.
        # `torch.no_grad()` is used as we are only testing inference.
        with torch.no_grad():
            output_rbln = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(*rbln_args)
            output_cpu = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(*cpu_args)

        # --- 5. Validation ---
        # First, verify that the output metadata (shape, dtype) is identical.
        self.assertEqual(output_rbln.shape, output_cpu.shape)
        self.assertEqual(output_rbln.dtype, output_cpu.dtype)

        # Second, verify that the numerical results are close within a tolerance.
        # This accounts for minor floating-point differences between hardware.
        self.assertEqual(output_rbln.cpu(), output_cpu, rtol=self.rtol, atol=self.atol)

        # Finally, verify that the in-place mutations of the K/V caches are also correct.
        self.assertEqual(k_cache_rbln.cpu(), k_cache_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(v_cache_rbln.cpu(), v_cache_cpu, rtol=self.rtol, atol=self.atol)

    @dtypes(*SUPPORTED_DTYPES)
    def test_paged_causal_attn_decode_comparison(self, dtype):
        # --- 1. Test Configuration ---
        # Define parameters for the decode step. This test uses a batch size of 2
        # to ensure multi-batch logic is handled correctly.
        batch_size = 2
        max_seq_length = 8192
        seq_len = 256  # Represents the length of the existing context in the cache.
        num_q_heads = 4
        num_kv_heads = 8
        head_dim = 64

        # --- 2. Reference CPU Tensor Creation ---
        # For the decode step, Q, K, and V represent a single new token (seq length = 1).
        q_cpu = torch.randn([batch_size, num_kv_heads, num_q_heads, 1, head_dim], dtype=dtype)
        k_cpu = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype)
        v_cpu = torch.randn([batch_size, num_kv_heads, 1, 1, head_dim], dtype=dtype)

        # Initialize K/V caches and pre-populate them with random data up to `seq_len`
        # to simulate a state where a prompt has already been processed.
        k_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        v_cache_cpu = torch.zeros([batch_size, num_kv_heads, 1, max_seq_length, head_dim], dtype=dtype)
        k_slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        k_cache_cpu[:, :, :, :seq_len, :] = torch.randn(k_slice_shape, dtype=dtype)
        v_slice_shape = (batch_size, num_kv_heads, 1, seq_len, head_dim)
        v_cache_cpu[:, :, :, :seq_len, :] = torch.randn(v_slice_shape, dtype=dtype)

        # Create other arguments. `seq` now indicates the current length of each sequence.
        scale_tensor = torch.tensor(1.0 / math.sqrt(head_dim))
        block_size = max_seq_length
        seq_cpu = torch.tensor([[seq_len], [seq_len]], dtype=torch.int32)
        block_table_cpu = torch.tensor([[0], [1]], dtype=torch.int16)

        # --- 3. Create CPU Reference Tensors ---
        q_rbln = q_cpu.clone().to(self.rbln_device)
        k_rbln = k_cpu.clone().to(self.rbln_device)
        v_rbln = v_cpu.clone().to(self.rbln_device)
        k_cache_rbln = k_cache_cpu.clone().to(self.rbln_device)
        v_cache_rbln = v_cache_cpu.clone().to(self.rbln_device)
        seq_rbln = seq_cpu.clone().to(self.rbln_device)
        block_table_rbln = block_table_cpu.clone().to(self.rbln_device)

        # Pack arguments for both operator calls.
        # mask is not provided (optional parameter, not used in practice)
        rbln_args = (
            q_rbln,
            k_rbln,
            v_rbln,
            k_cache_rbln,
            v_cache_rbln,
            seq_rbln,
            scale_tensor,
            block_table_rbln,
            block_size,
        )
        cpu_args = (
            q_cpu,
            k_cpu,
            v_cpu,
            k_cache_cpu,
            v_cache_cpu,
            seq_cpu,
            scale_tensor,
            block_table_cpu,
            block_size,
        )

        # --- 4. Operator Execution ---
        with torch.no_grad():
            output_rbln = torch.ops.rbln_custom_ops.paged_causal_attn_decode(*rbln_args)
            output_cpu = torch.ops.rbln_custom_ops.paged_causal_attn_decode(*cpu_args)

        # --- 5. Validation ---
        self.assertEqual(output_rbln.shape, output_cpu.shape)
        self.assertEqual(output_rbln.dtype, output_cpu.dtype)
        self.assertEqual(output_rbln.cpu(), output_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(k_cache_rbln.cpu(), k_cache_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(v_cache_rbln.cpu(), v_cache_cpu, rtol=self.rtol, atol=self.atol)

    def _run_flash_attention_naive(self, dtype, *, seq_len: int, op_name: str) -> None:
        """Shared body for flash_attention_naive prefill/decode comparison tests."""
        # --- 1. Test Configuration ---
        # Single batch / single block / single partition keeps the CPU reference
        # in sync with the simplified layout used by `_flash_attn_naive_reference`.
        num_kv_heads = 8  # H
        num_q_groups = 4  # G (num_q_heads == num_kv_heads * num_q_groups)
        head_dim = 64  # D — must be a multiple of 64 for the kernel
        num_blocks = 1  # B
        partition_size = 256  # P (also equals context length since NP=1)

        is_decode = "decode" in op_name
        # Prefill writes the full prompt starting at slot 0; decode appends one
        # token to a partially-filled cache. The mask covers exactly the valid
        # cache region (history + new token) so both reference and kernel
        # attend over the same positions.
        prefilled = 32 if is_decode else 0
        valid_len = prefilled + seq_len

        # --- 2. CPU Tensor Creation ---
        q_cpu = torch.randn([1, num_kv_heads, num_q_groups, seq_len, head_dim], dtype=dtype)
        k_cpu = torch.randn([1, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)
        v_cpu = torch.randn([1, num_kv_heads, 1, seq_len, head_dim], dtype=dtype)

        # Unified KV cache: axis 0 stacks [K, V], paged by (B, H, 1, P, D).
        kv_cache_cpu = torch.randn([2, num_blocks, num_kv_heads, 1, partition_size, head_dim], dtype=dtype)

        # Mask is 1 for valid positions [0, valid_len), 0 elsewhere.
        mask_cpu = torch.zeros([1, 1, 1, seq_len, partition_size], dtype=dtype)
        mask_cpu[:, :, :, :, :valid_len] = 1

        scale_tensor = torch.tensor(1.0 / math.sqrt(head_dim))
        seq_idx_cpu = torch.tensor([[prefilled]], dtype=torch.int32)
        # prefill: [num_partitions], decode: [batch, num_partitions]
        block_tables_cpu = torch.tensor([[0]], dtype=torch.int16) if is_decode else torch.tensor([0], dtype=torch.int16)
        # slot_mapping is consumed by the kernel for memory placement but not
        # by the CPU reference; any positionally valid tensor is fine here.
        slot_mapping_cpu = torch.arange(seq_len, dtype=torch.int32)

        # --- 3. RBLN Tensors ---
        q_rbln = q_cpu.clone().to(self.rbln_device)
        k_rbln = k_cpu.clone().to(self.rbln_device)
        v_rbln = v_cpu.clone().to(self.rbln_device)
        kv_cache_rbln = kv_cache_cpu.clone().to(self.rbln_device)
        mask_rbln = mask_cpu.clone().to(self.rbln_device)
        seq_idx_rbln = seq_idx_cpu.clone().to(self.rbln_device)
        block_tables_rbln = block_tables_cpu.clone().to(self.rbln_device)
        slot_mapping_rbln = slot_mapping_cpu.clone().to(self.rbln_device)

        rbln_args = (
            q_rbln,
            k_rbln,
            v_rbln,
            kv_cache_rbln,
            mask_rbln,
            scale_tensor,
            seq_idx_rbln,
            block_tables_rbln,
            slot_mapping_rbln,
        )
        cpu_args = (
            q_cpu,
            k_cpu,
            v_cpu,
            kv_cache_cpu,
            mask_cpu,
            scale_tensor,
            seq_idx_cpu,
            block_tables_cpu,
            slot_mapping_cpu,
        )

        # --- 4. Operator Execution ---
        op = getattr(torch.ops.rbln_custom_ops, op_name)
        with torch.no_grad():
            output_rbln = op(*rbln_args)
            output_cpu = op(*cpu_args)

        # --- 5. Validation ---
        self.assertEqual(output_rbln.shape, output_cpu.shape)
        self.assertEqual(output_rbln.dtype, output_cpu.dtype)
        self.assertEqual(output_rbln.cpu(), output_cpu, rtol=self.rtol, atol=self.atol)
        self.assertEqual(kv_cache_rbln.cpu(), kv_cache_cpu, rtol=self.rtol, atol=self.atol)

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_prefill_comparison(self, dtype):
        self._run_flash_attention_naive(dtype, seq_len=256, op_name="flash_attention_naive_prefill")

    @dtypes(*SUPPORTED_DTYPES)
    def test_flash_attention_naive_decode_comparison(self, dtype):
        self._run_flash_attention_naive(dtype, seq_len=1, op_name="flash_attention_naive_decode")


instantiate_device_type_tests(TestCustomKernelRBLN, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
