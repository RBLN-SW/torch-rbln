# Owner(s): ["module: PrivateUse1"]
"""Optimum RBLN LLM tests: prefill/decode, KV cache validation, TP1/TP2 and eager mode."""

import os
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import pytest
import torch
from optimum.rbln import (
    RBLNExaoneForCausalLM,
    RBLNExaoneForCausalLMConfig,
    RBLNLlamaForCausalLM,
    RBLNLlamaForCausalLMConfig,
    RBLNQwen2ForCausalLM,
    RBLNQwen2ForCausalLMConfig,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import DecoderOnlyWrapper
from optimum.rbln.transformers.models.exaone.exaone_architecture import ExaoneForCausalLMWrapper
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from transformers import AutoConfig, AutoTokenizer

from test.utils import run_in_isolated_process, SUPPORTED_DTYPES


class ModelType(Enum):
    """Supported model types"""

    LLAMA_1B = "llama_1b"
    LLAMA_3B = "llama_3b"
    EXAONE_2_4B = "exaone_2_4b"
    EXAONE_7_8B = "exaone_7_8b"
    QWEN_1_5B = "qwen_1_5b"


@dataclass
class ModelConfig:
    """Model-specific configuration"""

    model_id: str
    rbln_model_class: Any
    rbln_config_class: Any
    wrapper_class: Any
    num_key_value_head: int
    head_dim: int
    use_rotary_emb: bool = True


@dataclass
class DecodeContext:
    """Context to run one decode step after prefill (model, next-token input, cache position, etc.)."""

    model: Any
    decode_input: torch.Tensor
    block_table: torch.Tensor
    cache_position: torch.Tensor
    original_length: int


MODEL_CONFIGS: dict[ModelType, ModelConfig] = {
    ModelType.LLAMA_1B: ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        rbln_model_class=RBLNLlamaForCausalLM,
        rbln_config_class=RBLNLlamaForCausalLMConfig,
        wrapper_class=DecoderOnlyWrapper,
        num_key_value_head=8,
        head_dim=64,
        use_rotary_emb=True,
    ),
    ModelType.LLAMA_3B: ModelConfig(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        rbln_model_class=RBLNLlamaForCausalLM,
        rbln_config_class=RBLNLlamaForCausalLMConfig,
        wrapper_class=DecoderOnlyWrapper,
        num_key_value_head=8,
        head_dim=128,
        use_rotary_emb=True,
    ),
    ModelType.EXAONE_2_4B: ModelConfig(
        model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        rbln_model_class=RBLNExaoneForCausalLM,
        rbln_config_class=RBLNExaoneForCausalLMConfig,
        wrapper_class=ExaoneForCausalLMWrapper,
        num_key_value_head=8,
        head_dim=80,
        use_rotary_emb=True,
    ),
    ModelType.EXAONE_7_8B: ModelConfig(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        rbln_model_class=RBLNExaoneForCausalLM,
        rbln_config_class=RBLNExaoneForCausalLMConfig,
        wrapper_class=ExaoneForCausalLMWrapper,
        num_key_value_head=8,
        head_dim=80,
        use_rotary_emb=True,
    ),
    ModelType.QWEN_1_5B: ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        rbln_model_class=RBLNQwen2ForCausalLM,
        rbln_config_class=RBLNQwen2ForCausalLMConfig,
        wrapper_class=DecoderOnlyWrapper,
        num_key_value_head=2,
        head_dim=128,
        use_rotary_emb=True,
    ),
}


EXPECTED_RESULTS = {
    (ModelType.LLAMA_1B, "tp1_inputs_embeds"): {"prefill": " Paris", "decode": ".\n"},
    (ModelType.LLAMA_1B, "tp1_input_ids"): {"prefill": " Paris", "decode": ".\n"},
    (ModelType.LLAMA_1B, "tp1_eager"): {"prefill": " Paris", "decode": ".\n"},
    (ModelType.LLAMA_1B, "tp2_inputs_embeds"): {"prefill": " Paris", "decode": ".\n"},
    (ModelType.LLAMA_1B, "tp2_input_ids"): {"prefill": " Paris", "decode": ".\n"},
    (ModelType.QWEN_1_5B, "tp1_inputs_embeds"): {"prefill": " The", "decode": " capital"},
    (ModelType.QWEN_1_5B, "tp1_input_ids"): {"prefill": " The", "decode": " capital"},
    (ModelType.QWEN_1_5B, "tp1_eager"): {"prefill": " The", "decode": " capital"},
    (ModelType.QWEN_1_5B, "tp2_inputs_embeds"): {"prefill": " The", "decode": " capital"},
    (ModelType.QWEN_1_5B, "tp2_input_ids"): {"prefill": " The", "decode": " capital"},
}


def create_model(
    model_type: ModelType,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    use_inputs_embeds: bool,
) -> tuple[Any, Any, ModelConfig]:
    """Create and initialize model"""
    config_info = MODEL_CONFIGS[model_type]

    rbln_config = config_info.rbln_config_class(
        batch_size=1,
        attn_impl="eager",
        max_seq_len=max_seq_len,
        kvcache_partition_len=None,
        use_inputs_embeds=use_inputs_embeds,
        use_attention_mask=False,
        use_position_ids=False,
        sliding_window_layers=[],
        kvcache_block_size=max_seq_len,
    )
    rbln_config._attn_implementation = "eager"

    hf_config = AutoConfig.from_pretrained(
        config_info.model_id,
        trust_remote_code=True,
    )

    orig_model = config_info.rbln_model_class.get_pytorch_model(
        config_info.model_id,
        rbln_config=rbln_config,
        dtype=dtype,
        trust_remote_code=True,
        config=hf_config,
    )

    wrapped_model = config_info.wrapper_class(
        orig_model,
        use_rotary_emb=config_info.use_rotary_emb,
        rbln_config=rbln_config,
    )

    model = wrapped_model.to(device)

    return model, rbln_config, config_info


def prepare_inputs(
    tokenizer: Any,
    prompt: str,
    target_seq_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Prepare input tokens"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    original_length = input_ids.shape[1]

    if original_length < target_seq_length:
        padding_length = target_seq_length - original_length
        padding = torch.zeros(1, padding_length, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    else:
        input_ids = input_ids[:, :target_seq_length]

    input_ids = input_ids.to(device)
    return input_ids, original_length


def prepare_past_key_values(
    model: Any,
    batch_size: int,
    num_key_value_head: int,
    max_seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    eager_mode: bool,
) -> list[torch.Tensor]:
    """Initialize past key values with optional eager mode support"""
    past_key_values = [
        torch.zeros(batch_size, num_key_value_head, max_seq_len, head_dim, dtype=dtype).to(device)
        for _ in range(model.num_hidden_layers * 2)
    ]

    return past_key_values


def _get_embedding_layer(model: Any) -> Any:
    """Get embedding layer from model, trying multiple possible locations"""

    # Helper function to recursively search for embedding layer
    def _search_embedding(obj: Any, visited: Optional[set] = None) -> Any:
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return None
        visited.add(obj_id)

        # Try get_input_embeddings method
        if hasattr(obj, "get_input_embeddings"):
            try:
                emb = obj.get_input_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                pass

        # Try embed_tokens attribute
        if hasattr(obj, "embed_tokens"):
            emb = obj.embed_tokens
            if emb is not None:
                return emb

        # Try common attribute names
        for attr_name in ["model", "orig_model", "_model", "base_model", "transformer", "encoder", "decoder"]:
            if hasattr(obj, attr_name):
                try:
                    sub_obj = getattr(obj, attr_name)
                    if sub_obj is not None and sub_obj is not obj:
                        result = _search_embedding(sub_obj, visited)
                        if result is not None:
                            return result
                except Exception:
                    pass

        return None

    # Try direct access first
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                return emb
        except Exception:
            pass

    # Try explicit paths based on wrapper structure
    # Path: model.model.model (DecoderOnlyWrapper -> DecoderOnlyForCausalLM -> DecoderOnlyModel)
    if hasattr(model, "model") and hasattr(model.model, "model"):
        inner_model = model.model.model
        # Try get_embedding method (DecoderOnlyModel has this)
        if hasattr(inner_model, "get_embedding"):
            try:
                emb = inner_model.get_embedding()
                if emb is not None:
                    return emb
            except Exception:
                pass
        # Try get_input_embeddings method
        if hasattr(inner_model, "get_input_embeddings"):
            try:
                emb = inner_model.get_input_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                pass
        # Try embed_tokens attribute
        if hasattr(inner_model, "embed_tokens"):
            emb = inner_model.embed_tokens
            if emb is not None:
                return emb

    # Recursive search
    result = _search_embedding(model)
    if result is not None:
        return result

    raise AttributeError("Could not find embedding layer in model")


def run_prefill(
    model: Any,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    block_table: torch.Tensor,
    query_position: torch.Tensor,
    past_key_values: list[torch.Tensor],
    eager_mode: bool,
) -> torch.Tensor:
    """Run prefill step with optional eager mode support"""
    if eager_mode:
        # Eager mode: direct model call without compilation
        outputs = model(inputs_embeds, cache_position, block_table, query_position, *past_key_values)
    else:
        # Graph mode: compile and run
        compiled_model = torch.compile(model, backend="rbln", dynamic=False)
        outputs = compiled_model(inputs_embeds, cache_position, block_table, query_position, *past_key_values)

    return outputs


def run_decode(
    model: Any,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    block_table: torch.Tensor,
    past_key_values: list[torch.Tensor],
    eager_mode: bool,
) -> torch.Tensor:
    """Run decode step with optional eager mode support"""
    model.phase = "decode"

    if eager_mode:
        # Eager mode: direct model call without compilation
        outputs = model(inputs_embeds, cache_position, block_table, *past_key_values)
    else:
        # Graph mode: compile and run
        compiled_model = torch.compile(model, backend="rbln", dynamic=False)
        outputs = compiled_model(inputs_embeds, cache_position, block_table, *past_key_values)

    return outputs


def build_decode_context_after_prefill(
    model: Any,
    prefill_outputs: torch.Tensor,
    use_inputs_embeds: bool,
    device: str,
    original_length: int,
    block_table: torch.Tensor,
) -> DecodeContext:
    """Build context for one decode step from prefill logits. Does not run prefill."""
    next_token = torch.argmax(prefill_outputs.to("cpu"), dim=-1, keepdim=True)
    next_token_ids = next_token.squeeze(0)
    if use_inputs_embeds:
        embedding_layer = _get_embedding_layer(model)
        next_token_ids_cpu = next_token_ids.to("cpu")
        embedding_layer_cpu = embedding_layer.to("cpu")
        decode_input = embedding_layer_cpu(next_token_ids_cpu).to(device)
    else:
        decode_input = next_token_ids.to(device)
    cache_position_decode = torch.tensor([[original_length]], dtype=torch.int32)
    return DecodeContext(
        model=model,
        decode_input=decode_input,
        block_table=block_table,
        cache_position=cache_position_decode,
        original_length=original_length,
    )


def _validate_kv_layer_after_prefill(
    self,
    layer_tensor: torch.Tensor,
    layer_idx: int,
    seq_length: int,
    *,
    inplace_update_and_return_vals: bool = True,
) -> list[float] | None:
    """Validate one layer's key or value cache after prefill: copy to CPU/RBLN, print slice;
    optionally in-place update and return test_vals (for key only)."""
    self.assertEqual(layer_tensor.dim(), 4)
    self.assertEqual(layer_tensor.dtype, torch.float16, msg="KV cache should be float16")

    print(f"layer {layer_idx}: copy cache to CPU and check shape")
    cpu_t = layer_tensor.to("cpu")
    self.assertEqual(cpu_t.shape, layer_tensor.shape)
    valid_slice = cpu_t[..., :seq_length, :].float()
    # KV cache after prefill is typically mostly non-zero in the valid range.
    nonzero_ratio = (valid_slice.abs() > 1e-5).float().mean().item()
    self.assertGreaterEqual(
        nonzero_ratio,
        0.5,
        f"past_key_values layer {layer_idx} cache should be mostly non-zero in valid range "
        f"after prefill (got {nonzero_ratio:.2%} non-zero)",
    )
    print(f"layer {layer_idx}: nonzero_ratio: {nonzero_ratio:.2%}")
    self.assertTrue(
        torch.isfinite(valid_slice).all(),
        f"past_key_values layer {layer_idx} cache should have no nan/inf in valid range",
    )

    print(f"layer {layer_idx}: copy_ to another RBLN tensor and compare on CPU")
    dst_rbln = torch.zeros_like(layer_tensor, device=layer_tensor.device)
    dst_rbln.copy_(layer_tensor)
    self.assertTrue(
        torch.allclose(cpu_t.float(), dst_rbln.to("cpu").float(), rtol=1e-2, atol=1e-2),
        f"copy_ from past_key_values layer {layer_idx} to another RBLN tensor should preserve values",
    )

    small = layer_tensor[:, :, :4, :4]
    # print(small)
    print(f"layer {layer_idx}: assert str(small slice) contains 'tensor'")
    self.assertIn("tensor", str(small).lower())

    if not inplace_update_and_return_vals:
        return None

    n_heads = layer_tensor.shape[1]
    test_vals = [float(h + 1) for h in range(n_heads)]
    print(f"layer {layer_idx}: in-place update (0, head_i, 3, 3) and read back (n_heads={n_heads})")
    for head_i, val in enumerate(test_vals):
        layer_tensor[0, head_i, 3, 3] = val
    for head_i, expected in enumerate(test_vals):
        read_back = layer_tensor.to("cpu")[0, head_i, 3, 3].item()
        self.assertAlmostEqual(
            read_back,
            expected,
            places=1,
            msg=f"past_key_values layer {layer_idx} in-place update at head {head_i} should be visible when read back",
        )
    return test_vals


def _logits_shape_consistency(
    self,
    prefill_outputs: torch.Tensor,
    decode_outputs: torch.Tensor,
) -> None:
    """Compare prefill (last position) and decode encoding results: shape consistency and finite checks."""
    prefill_last = prefill_outputs[:, -1:, :].to("cpu").float()
    decode_logits = decode_outputs.to("cpu").float()
    self.assertEqual(
        prefill_last.shape,
        decode_logits.shape,
        "Prefill last-position logits shape should match decode logits shape",
    )
    self.assertTrue(
        torch.isfinite(prefill_last).all(),
        "Prefill logits (last position) should be finite",
    )
    self.assertTrue(
        torch.isfinite(decode_logits).all(),
        "Decode logits should be finite",
    )
    print(f"logits_shape_consistency: shapes match {prefill_last.shape}, prefill_last finite, decode finite")


def _validate_inplace_values_preserved_after_decode(
    self,
    past_key_values: list[torch.Tensor],
    layer_key_indices: list[int],
    layer_test_vals: list[list[float]],
) -> None:
    """Assert that in-place updated values at (0, head_i, 3, 3) are still present after decode."""
    print(f"check in-place values (0, head_i, 3, 3) preserved after decode ({len(layer_key_indices)} layers)")
    for layer_idx, pkv_idx in enumerate(layer_key_indices):
        layer_key = past_key_values[pkv_idx]
        test_vals = layer_test_vals[layer_idx]

        # print test
        small = layer_key.to("cpu")[:, :, :4, :4]
        # print(small)
        self.assertIn("tensor", str(small).lower())

        for head_i, expected in enumerate(test_vals):
            read_back = layer_key.to("cpu")[0, head_i, 3, 3].item()
            self.assertAlmostEqual(
                read_back,
                expected,
                places=1,
                msg=f"after decode: past_key_values layer {layer_idx} in-place value at head {head_i} should still be present",
            )
    print("in-place values preserved for all layers")


def _validate_kv_new_slot_written_after_decode(
    self,
    past_key_values: list[torch.Tensor],
    layer_key_indices: list[int],
    new_slot: int,
) -> None:
    """Assert that key and value caches at seq position new_slot are updated (not all zeros)."""
    print(f"check KV cache at seq_position={new_slot} updated after decode ({len(layer_key_indices)} layers)")
    for layer_idx, pkv_idx in enumerate(layer_key_indices):
        key_cache = past_key_values[pkv_idx].to("cpu").float()
        val_cache = past_key_values[pkv_idx + 1].to("cpu").float()
        key_at_new = key_cache[0, :, new_slot, :]
        val_at_new = val_cache[0, :, new_slot, :]
        self.assertFalse(
            torch.allclose(key_at_new, torch.zeros_like(key_at_new)),
            f"after decode: layer {layer_idx} key cache at seq_position {new_slot} should be updated (not all zeros)",
        )
        self.assertFalse(
            torch.allclose(val_at_new, torch.zeros_like(val_at_new)),
            f"after decode: layer {layer_idx} value cache at seq_position {new_slot} should be updated (not all zeros)",
        )
    print("KV new_slot written for all layers")


def _validate_kv_layers_differ(
    self,
    past_key_values: list[torch.Tensor],
    layer_key_indices: list[int],
    seq_length: int,
) -> None:
    """Assert that key caches for different layers are not identical (catches same buffer reused)."""
    if len(layer_key_indices) < 2:
        return
    print("check key caches differ between layers (no shared buffer)")
    k0 = past_key_values[layer_key_indices[0]].to("cpu").float()[..., :seq_length, :]
    k1 = past_key_values[layer_key_indices[1]].to("cpu").float()[..., :seq_length, :]
    self.assertFalse(
        torch.allclose(k0, k1),
        msg="layer 0 and layer 1 key caches should not be identical (different layers must differ)",
    )


def model_worker(
    prompt: str,
    max_seq_len: int,
    target_seq_length: int,
    model_type: ModelType,
    dtype: torch.dtype,
    test_case_name: str,
    eager_mode: bool,
    use_inputs_embeds: bool,
    device: str | torch.device = "rbln:0",
):
    """Module-level worker for running test case in subprocess (must be picklable)."""
    device = torch.device(device) if isinstance(device, str) else device

    model, rbln_config, config_info = create_model(
        model_type=model_type,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
        use_inputs_embeds=use_inputs_embeds,
    )

    tokenizer = AutoTokenizer.from_pretrained(config_info.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, original_length = prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        target_seq_length=target_seq_length,
        device=device,
    )

    past_key_values = prepare_past_key_values(
        model=model,
        batch_size=1,
        num_key_value_head=config_info.num_key_value_head,
        max_seq_len=max_seq_len,
        head_dim=config_info.head_dim,
        dtype=dtype,
        device=device,
        eager_mode=eager_mode,
    )

    # Prefill step: use inputs_embeds or input_ids according to use_inputs_embeds
    if use_inputs_embeds:
        embedding_layer = _get_embedding_layer(model)
        input_ids_cpu = input_ids.to("cpu")
        embedding_layer_cpu = embedding_layer.to("cpu")
        prefill_input = embedding_layer_cpu(input_ids_cpu).to(device)
    else:
        prefill_input = input_ids

    seq_length = input_ids.shape[-1]  # Always the padded token count
    if eager_mode:
        query_position = torch.scalar_tensor(original_length - 1, dtype=torch.int16).to(device)
        # Match production runtime: real tokens get sequential positions, padding gets 0.
        cache_position = torch.arange(0, original_length, dtype=torch.int32)
        if original_length < seq_length:
            cache_position = torch.nn.functional.pad(cache_position, (0, seq_length - original_length))
        cache_position = cache_position.unsqueeze(0).to(device)
        block_table = torch.arange(1, dtype=torch.int16).to(device)
    else:
        query_position = torch.scalar_tensor(original_length - 1, dtype=torch.int16)
        cache_position = torch.arange(0, seq_length, dtype=torch.int32).unsqueeze(0)
        block_table = torch.arange(1, dtype=torch.int16)

    outputs = run_prefill(
        model=model,
        inputs_embeds=prefill_input,
        cache_position=cache_position,
        block_table=block_table,
        query_position=query_position,
        past_key_values=past_key_values,
        eager_mode=eager_mode,
    )

    next_token_logits = outputs.to("cpu")
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    prefill_decoded = tokenizer.decode(next_token[0][0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    expected_key = (model_type, test_case_name)
    expected_prefill = EXPECTED_RESULTS.get(expected_key, {}).get("prefill", "")
    if expected_prefill:
        assert prefill_decoded == expected_prefill, (
            f"Prefill output mismatch for {model_type.value} {test_case_name} "
            f"(use_inputs_embeds={use_inputs_embeds}). "
            f"Expected: '{expected_prefill}', Got: '{prefill_decoded}'"
        )

    # Decode step: use inputs_embeds or input_ids according to use_inputs_embeds
    next_token_ids = next_token.squeeze(0)
    if use_inputs_embeds:
        embedding_layer = _get_embedding_layer(model)
        next_token_ids_cpu = next_token_ids.to("cpu")
        embedding_layer_cpu = embedding_layer.to("cpu")
        decode_input = embedding_layer_cpu(next_token_ids_cpu).to(device)
    else:
        decode_input = next_token_ids.to(device)

    if eager_mode:
        cache_position_decode = torch.tensor([[original_length]], dtype=torch.int32, device=device)
    else:
        cache_position_decode = torch.tensor([[original_length]], dtype=torch.int32)

    outputs = run_decode(
        model=model,
        inputs_embeds=decode_input,
        cache_position=cache_position_decode,
        block_table=block_table,
        past_key_values=past_key_values,
        eager_mode=eager_mode,
    )

    next_token_logits = outputs.to("cpu")
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    decode_decoded = tokenizer.decode(next_token[0][0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    expected_decode = EXPECTED_RESULTS.get(expected_key, {}).get("decode", "")
    if expected_decode:
        assert decode_decoded == expected_decode, (
            f"Decode output mismatch for {model_type.value} {test_case_name} "
            f"(use_inputs_embeds={use_inputs_embeds}). "
            f"Expected: '{expected_decode}', Got: '{decode_decoded}'"
        )


def past_key_values_worker(
    prompt: str,
    max_seq_len: int,
    target_seq_length: int,
    model_type: ModelType,
    dtype: torch.dtype,
    device: str | torch.device = "rbln:0",
) -> None:
    """Subprocess worker: same env contract as model_worker (RBLN_NPUS_PER_DEVICE set before spawn)."""
    device = torch.device(device) if isinstance(device, str) else device
    tc = unittest.TestCase()

    model, _rbln_config, config_info = create_model(
        model_type=model_type,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype,
        use_inputs_embeds=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config_info.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, original_length = prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        target_seq_length=target_seq_length,
        device=device,
    )

    past_key_values = prepare_past_key_values(
        model=model,
        batch_size=1,
        num_key_value_head=config_info.num_key_value_head,
        max_seq_len=max_seq_len,
        head_dim=config_info.head_dim,
        dtype=dtype,
        device=device,
        eager_mode=False,
    )

    embedding_layer = _get_embedding_layer(model)
    input_ids_cpu = input_ids.to("cpu")
    embedding_layer_cpu = embedding_layer.to("cpu")
    prefill_input = embedding_layer_cpu(input_ids_cpu).to(device)

    seq_length = prefill_input.shape[-2]
    query_position = torch.scalar_tensor(original_length - 1, dtype=torch.int16)
    cache_position = torch.arange(0, seq_length, dtype=torch.int32).unsqueeze(0)
    block_table = torch.arange(1, dtype=torch.int16)

    prefill_outputs = run_prefill(
        model=model,
        inputs_embeds=prefill_input,
        cache_position=cache_position,
        block_table=block_table,
        query_position=query_position,
        past_key_values=past_key_values,
        eager_mode=False,
    )

    decode_ctx = build_decode_context_after_prefill(
        model=model,
        prefill_outputs=prefill_outputs,
        use_inputs_embeds=True,
        device=device,
        original_length=original_length,
        block_table=block_table,
    )

    test_case_name = "tp2_inputs_embeds" if os.environ.get("RBLN_NPUS_PER_DEVICE", "1") == "2" else "tp1_inputs_embeds"
    expected_key = (model_type, test_case_name)
    expected_prefill = EXPECTED_RESULTS.get(expected_key, {}).get("prefill", "")
    expected_decode = EXPECTED_RESULTS.get(expected_key, {}).get("decode", "")

    prefill_last_logits = prefill_outputs[:, -1:, :].to("cpu")
    prefill_next_token = torch.argmax(prefill_last_logits, dim=-1, keepdim=True)
    prefill_decoded = tokenizer.decode(
        prefill_next_token[0][0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(f"prefill_decoded: {prefill_decoded}, expected_prefill: {expected_prefill}")
    if expected_prefill:
        tc.assertEqual(
            prefill_decoded,
            expected_prefill,
            (
                f"Prefill output mismatch for {model_type.value} {test_case_name}. "
                f"Expected: '{expected_prefill}', Got: '{prefill_decoded}'"
            ),
        )

    num_layers_to_check = 3
    layer_key_indices = [l * 2 for l in range(num_layers_to_check)]
    tc.assertGreaterEqual(len(past_key_values), layer_key_indices[-1] + 1)

    layer_test_vals = []
    for layer_idx, pkv_idx in enumerate(layer_key_indices):
        layer_test_vals.append(_validate_kv_layer_after_prefill(tc, past_key_values[pkv_idx], layer_idx, seq_length))
        _validate_kv_layer_after_prefill(
            tc,
            past_key_values[pkv_idx + 1],
            layer_idx,
            seq_length,
            inplace_update_and_return_vals=False,
        )
    _validate_kv_layers_differ(tc, past_key_values, layer_key_indices, seq_length)

    decode_outputs = run_decode(
        model=decode_ctx.model,
        inputs_embeds=decode_ctx.decode_input,
        cache_position=decode_ctx.cache_position,
        block_table=decode_ctx.block_table,
        past_key_values=past_key_values,
        eager_mode=False,
    )

    _logits_shape_consistency(tc, prefill_outputs, decode_outputs)

    decode_next_token = torch.argmax(decode_outputs.to("cpu"), dim=-1, keepdim=True)
    decode_decoded = tokenizer.decode(
        decode_next_token[0][0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(f"decode_decoded: {decode_decoded}, expected_decode: {expected_decode}")
    if expected_decode:
        tc.assertEqual(
            decode_decoded,
            expected_decode,
            (
                f"Decode output mismatch for {model_type.value} {test_case_name}. "
                f"Expected: '{expected_decode}', Got: '{decode_decoded}'"
            ),
        )

    _validate_inplace_values_preserved_after_decode(tc, past_key_values, layer_key_indices, layer_test_vals)
    _validate_kv_new_slot_written_after_decode(tc, past_key_values, layer_key_indices, decode_ctx.original_length)


class TestLlamaBase(TestCase):
    """Base test class with common functionality for both graph and eager modes"""

    prompt = "What is the capital of France?"
    max_seq_len = 8192
    target_seq_length = 128

    def _run_test_case(
        self,
        model_type: ModelType,
        dtype: torch.dtype,
        test_case_name: str,
        eager_mode: bool,
        use_inputs_embeds: bool,
        tp_size: Optional[int] = None,
        device: str | torch.device = "rbln:0",
    ):
        """
        Run a single test case in an isolated process.

        `RBLN_NPUS_PER_DEVICE` must be set in the parent process before spawn because DeviceMappingManager reads it
        during C++ singleton initialization when the child process imports the RBLN backend.
        """
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TORCH_RBLN_DISABLE_FALLBACK", "compile_error")
            if tp_size is not None:
                mp.setenv("RBLN_NPUS_PER_DEVICE", str(tp_size))

            run_in_isolated_process(
                model_worker,
                self.prompt,
                self.max_seq_len,
                self.target_seq_length,
                model_type,
                dtype,
                test_case_name,
                eager_mode,
                use_inputs_embeds,
                device,
            )

    def _run_past_key_values_case(
        self,
        model_type: ModelType,
        dtype: torch.dtype,
        tp_size: int,
        device: str | torch.device = "rbln:0",
    ) -> None:
        """KV cache validation in an isolated process (RBLN_NPUS_PER_DEVICE must be set before spawn)."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TORCH_RBLN_DISABLE_FALLBACK", "compile_error")
            mp.setenv("RBLN_NPUS_PER_DEVICE", str(tp_size))

            run_in_isolated_process(
                past_key_values_worker,
                self.prompt,
                self.max_seq_len,
                self.target_seq_length,
                model_type,
                dtype,
                device,
            )


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
class TestLlamaRSDTP(TestLlamaBase):
    """Test cases for RSD with Tensor Parallelism"""

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("tp_size", [1, 2])
    @parametrize("use_inputs_embeds", [True, False])
    @parametrize(
        "model_type",
        [ModelType.LLAMA_1B, ModelType.QWEN_1_5B],
        name_fn=lambda mt: mt.value,
    )
    def test_rsd_tensor_parallel(self, dtype, model_type, tp_size, use_inputs_embeds):
        """RSD graph mode, TP1/TP2 with RBLN I/O only (CPU host I/O for this path is not exercised)."""
        if tp_size >= 2:
            n_phys = torch.rbln.physical_device_count()
            if n_phys < tp_size:
                pytest.skip(f"Requires at least {tp_size} physical devices, found {n_phys}")
        path = "inputs_embeds" if use_inputs_embeds else "input_ids"
        test_case_name = f"tp{tp_size}_{path}"
        self._run_test_case(
            model_type=model_type,
            dtype=dtype,
            test_case_name=test_case_name,
            eager_mode=False,
            use_inputs_embeds=use_inputs_embeds,
            tp_size=tp_size,
            device="rbln:0",
        )

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("tp_size", [1, 2])
    @parametrize("model_type", [ModelType.LLAMA_1B], name_fn=lambda mt: mt.value)
    def test_past_key_values_valid(self, dtype, model_type, tp_size):
        """Validate past_key_values after prefill then decode; isolated process (TP matches RBLN_NPUS_PER_DEVICE)."""
        if tp_size >= 2:
            n_phys = torch.rbln.physical_device_count()
            if n_phys < tp_size:
                pytest.skip(f"Requires at least {tp_size} physical devices, found {n_phys}")
        self._run_past_key_values_case(model_type, dtype, tp_size=tp_size)


@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_deploy_mode")
class TestLlamaEager(TestLlamaBase):
    """Test cases for eager mode tests (without torch.compile)"""

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_type",
        [ModelType.LLAMA_1B, ModelType.QWEN_1_5B],
        name_fn=lambda mt: mt.value,
    )
    def test_tp1_eager_inputs_embeds(self, dtype, model_type):
        """Test RBLN input to RBLN output in eager mode with use_inputs_embeds=True"""
        self._run_test_case(
            model_type=model_type,
            dtype=dtype,
            test_case_name="tp1_eager",
            eager_mode=True,
            use_inputs_embeds=True,
        )

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "model_type",
        [ModelType.LLAMA_1B, ModelType.QWEN_1_5B],
        name_fn=lambda mt: mt.value,
    )
    def test_tp1_eager_input_ids(self, dtype, model_type):
        """Test RBLN input to RBLN output in eager mode with use_inputs_embeds=False"""
        self._run_test_case(
            model_type=model_type,
            dtype=dtype,
            test_case_name="tp1_eager",
            eager_mode=True,
            use_inputs_embeds=False,
        )


instantiate_device_type_tests(TestLlamaRSDTP, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestLlamaEager, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
