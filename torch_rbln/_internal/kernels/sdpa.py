"""SDPA (Scaled Dot Product Attention) kernel for RBLN device.

Forward: can_use_rbln_sdpa() -> RBLN path or CPU fallback
Backward: uses cached attn_weights from forward
"""

import math
from typing import Optional

import torch

from torch_rbln._internal.log_utils import rbln_log_cpu_fallback
from torch_rbln._internal.ops_utils import is_cpu_fallback_cases, to_cpu


# --- Attention Weights Cache for Overrideable SDPA ---
# This cache stores attn_weights from forward to be used in backward.
# Key: output tensor's data_ptr(), Value: attn_weights tensor
_sdpa_attn_weights_cache: dict[int, torch.Tensor] = {}


def _cache_attn_weights(output: torch.Tensor, attn_weights: torch.Tensor) -> None:
    """Cache attention weights using output tensor's data_ptr as key."""
    key = output.data_ptr()
    _sdpa_attn_weights_cache[key] = attn_weights


def _get_cached_attn_weights(output: torch.Tensor) -> Optional[torch.Tensor]:
    """Retrieve cached attention weights using output tensor's data_ptr as key."""
    key = output.data_ptr()
    return _sdpa_attn_weights_cache.pop(key, None)


def _clear_attn_weights_cache() -> None:
    """Clear all cached attention weights."""
    _sdpa_attn_weights_cache.clear()


# --- Tensor Subclass Detection ---


def _is_tensor_subclass(t: torch.Tensor) -> bool:
    """Check if tensor is a subclass (e.g., CompositeCompliantTensor)."""
    return type(t) is not torch.Tensor


def _any_tensor_subclass(*tensors: torch.Tensor) -> bool:
    """Check if any tensor is a subclass."""
    return any(_is_tensor_subclass(t) for t in tensors if t is not None)


# --- RBLN SDPA Constraints ---
# HW supports: 3D/4D tensors, float16, 64-byte aligned shapes
# PyTorch overrideable path requires 4D; non-4D uses SDPBackend::math

RBLN_SDPA_SUPPORTED_DTYPES = {torch.float16}
RBLN_SDPA_MIN_DIM = 3
RBLN_SDPA_MAX_DIM = 4
_RBLN_SDPA_SHAPE_ALIGNMENT = 32  # 64 bytes / 2 bytes (float16)


def needs_sdpa_shape_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[bool, str]:
    """Check if SDPA needs CPU fallback due to shape alignment (64-byte boundary)."""
    head_dim = query.size(-1)
    L = query.size(-2)
    S = key.size(-2)

    if head_dim % _RBLN_SDPA_SHAPE_ALIGNMENT != 0:
        return True, f"head_dim {head_dim} not aligned to {_RBLN_SDPA_SHAPE_ALIGNMENT}"
    if L % _RBLN_SDPA_SHAPE_ALIGNMENT != 0:
        return True, f"query sequence length {L} not aligned to {_RBLN_SDPA_SHAPE_ALIGNMENT}"
    if S % _RBLN_SDPA_SHAPE_ALIGNMENT != 0:
        return True, f"key sequence length {S} not aligned to {_RBLN_SDPA_SHAPE_ALIGNMENT}"

    return False, ""


# --- Input Validation ---


def validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> None:
    """Validate SDPA inputs. Raises ValueError for invalid inputs."""
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )

    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )

    if attn_mask is not None:
        if not (attn_mask.dtype == torch.bool or attn_mask.dtype.is_floating_point):
            raise ValueError(
                f"Expected attn_mask to have dtype bool or floating point, "
                f"but got attn_mask.dtype: {attn_mask.dtype} instead."
            )
        if attn_mask.device != query.device:
            raise ValueError(
                f"Expected attn_mask to have the same device as query, "
                f"but got attn_mask.device: {attn_mask.device} and query.device: {query.device} instead."
            )


def can_use_rbln_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> tuple[bool, str]:
    """Check if RBLN optimized SDPA can be used.

    Returns (True, "") if RBLN can handle this, or (False, reason) for CPU fallback.
    """
    if query.dim() < RBLN_SDPA_MIN_DIM or key.dim() < RBLN_SDPA_MIN_DIM or value.dim() < RBLN_SDPA_MIN_DIM:
        return False, f"dim < {RBLN_SDPA_MIN_DIM} not supported"

    if query.dim() > RBLN_SDPA_MAX_DIM or key.dim() > RBLN_SDPA_MAX_DIM or value.dim() > RBLN_SDPA_MAX_DIM:
        return False, f"dim > {RBLN_SDPA_MAX_DIM} not supported"

    # Nested tensors not supported
    try:
        query_nested = query.is_nested() if callable(query.is_nested) else query.is_nested
        key_nested = key.is_nested() if callable(key.is_nested) else key.is_nested
        value_nested = value.is_nested() if callable(value.is_nested) else value.is_nested
        if query_nested or key_nested or value_nested:
            return False, "nested tensors not supported"
    except (AttributeError, TypeError):
        pass

    # Only float16 supported on RBLN
    if query.dtype not in RBLN_SDPA_SUPPORTED_DTYPES:
        return False, f"dtype {query.dtype} not supported"

    if dropout_p > 0.0 and (query.requires_grad or key.requires_grad or value.requires_grad):
        return False, "dropout with gradients"

    if query.dim() == 4 and key.dim() == 4:
        q_heads, k_heads = query.size(1), key.size(1)
        if q_heads != k_heads and q_heads % k_heads != 0:
            return False, f"GQA: query heads ({q_heads}) not divisible by key heads ({k_heads})"

    if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
        return False, "empty tensors"

    if attn_mask is not None and attn_mask.dim() not in (2, 3, 4):
        return False, f"attn_mask dim {attn_mask.dim()} not in (2, 3, 4)"

    needs_fallback, reason = needs_sdpa_shape_fallback(query, key)
    if needs_fallback:
        return False, reason

    return True, ""


# --- GQA and Mask Utilities ---


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads n_rep times for GQA support."""
    if n_rep == 1:
        return hidden_states
    if hidden_states.dim() == 3:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _convert_bool_mask_to_float(
    attn_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert boolean mask to float: True→0.0, False→-inf."""
    if attn_mask is None or attn_mask.dtype != torch.bool:
        return attn_mask

    mask_device = attn_mask.device
    return torch.where(
        attn_mask,
        torch.zeros((), dtype=dtype, device=mask_device),
        torch.full((), float("-inf"), dtype=dtype, device=mask_device),
    )


def _create_causal_mask(
    L: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create causal mask outside compiled graph to avoid RBLN memory issues.

    This creates the upper triangular mask on CPU first, then moves to device.
    Creating tensors inside torch.compile'd graphs on RBLN can cause malloc corruption.
    """
    # Create on CPU first to avoid RBLN allocator issues in compiled graph
    causal_mask = torch.triu(
        torch.full((L, S), float("-inf"), dtype=dtype, device="cpu"),
        diagonal=1,
    )
    return causal_mask.to(device)


def _merge_masks(
    attn_mask: Optional[torch.Tensor],
    causal_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Merge attention mask and causal mask."""
    if causal_mask is None:
        return attn_mask
    if attn_mask is None:
        return causal_mask
    # Both masks exist - add them (both are additive masks with -inf for masked positions)
    return attn_mask + causal_mask


# --- CPU Fallback (using PyTorch's native SDPA) ---


def _sdpa_cpu_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """CPU fallback for SDPA. Upcasts float16 to float32 to prevent overflow in Q @ K^T."""
    # Handle GQA
    if query.dim() == 4 and key.dim() == 4:
        q_heads = query.size(1)
        k_heads = key.size(1)
        if q_heads != k_heads and q_heads % k_heads == 0:
            n_rep = q_heads // k_heads
            key = _repeat_kv(key, n_rep)
            value = _repeat_kv(value, n_rep)
            enable_gqa = False

    attn_mask = _convert_bool_mask_to_float(attn_mask, query.dtype, query.device)

    original_dtype = query.dtype
    original_device = query.device

    # Upcast float16 → float32 to prevent overflow in Q @ K^T
    compute_dtype = torch.float32 if original_dtype == torch.float16 else original_dtype

    query_cpu = to_cpu(query).to(compute_dtype)
    key_cpu = to_cpu(key).to(compute_dtype)
    value_cpu = to_cpu(value).to(compute_dtype)
    attn_mask_cpu = to_cpu(attn_mask).to(compute_dtype) if attn_mask is not None else None

    result_cpu = torch._C._nn.scaled_dot_product_attention(
        query_cpu,
        key_cpu,
        value_cpu,
        attn_mask=attn_mask_cpu,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    return result_cpu.to(original_dtype).to(original_device)


# --- RBLN Compiled Modules ---


def _compile_sdpa_attn_weights_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compile wrapper for: Q @ K^T * scale + mask -> softmax -> attn_weights.

    Note: Causal mask is now passed via attn_mask parameter instead of being
    created inside the module. This avoids RBLN memory allocation issues
    when tensors are created inside torch.compile'd graphs.
    """
    # Handle GQA (4D only)
    if query.dim() == 4:
        q_heads, k_heads = query.size(1), key.size(1)
        if q_heads != k_heads:
            key = _repeat_kv(key, q_heads // k_heads)

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Causal mask is now pre-computed and merged into attn_mask
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = torch.softmax(attn_weights, dim=-1)

    return attn_weights


def _compile_sdpa_output_fn(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Compile wrapper for: attn_weights @ V -> output."""
    # Handle GQA (4D only)
    if attn_weights.dim() == 4 and value.dim() == 4:
        a_heads, v_heads = attn_weights.size(1), value.size(1)
        if a_heads != v_heads:
            value = _repeat_kv(value, a_heads // v_heads)

    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

    output = torch.matmul(attn_weights, value)

    return output


# --- RBLN Forward Computation ---


def _sdpa_compute_attn_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """Compute attention weights using RBLN compiled graph.

    Causal mask is created outside the compiled graph to avoid RBLN memory issues.
    """
    from torch_rbln.device.context_holder import out_tensor_context

    if is_cpu_fallback_cases((query, key)):
        return _sdpa_attn_weights_fallback(query, key, attn_mask, is_causal, scale)

    query = query.contiguous()
    key = key.contiguous()

    # Create causal mask OUTSIDE compiled graph to avoid RBLN memory issues
    causal_mask = None
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = _create_causal_mask(L, S, query.dtype, query.device)

    # Merge attn_mask and causal_mask
    attn_mask = _convert_bool_mask_to_float(attn_mask, query.dtype, query.device)
    merged_mask = _merge_masks(attn_mask, causal_mask)
    if merged_mask is not None:
        merged_mask = merged_mask.contiguous()

    with out_tensor_context():
        compiled = torch.compile(
            _compile_sdpa_attn_weights_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        external_result = compiled(query, key, attn_mask=merged_mask, scale=scale)

    return external_result


def _sdpa_attn_weights_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """CPU fallback for attention weights. Upcasts float16 to float32 to prevent overflow."""
    L, S = query.size(-2), key.size(-2)
    original_dtype = query.dtype
    original_device = query.device

    # Upcast float16 → float32 to prevent overflow in Q @ K^T
    compute_dtype = torch.float32 if original_dtype == torch.float16 else original_dtype

    q_cpu = to_cpu(query).to(compute_dtype)
    k_cpu = to_cpu(key).to(compute_dtype)
    mask_cpu = to_cpu(attn_mask).to(compute_dtype) if attn_mask is not None else None

    if q_cpu.dim() == 4:
        q_heads, k_heads = q_cpu.size(1), k_cpu.size(1)
        if q_heads != k_heads:
            k_cpu = _repeat_kv(k_cpu, q_heads // k_heads)

    attn_weights = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.full((L, S), float("-inf"), dtype=compute_dtype, device="cpu"),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

    if mask_cpu is not None:
        attn_weights = attn_weights + mask_cpu

    attn_weights = torch.softmax(attn_weights, dim=-1)

    return attn_weights.to(original_dtype).to(original_device)


def _sdpa_compute_output(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """Compute output using RBLN compiled graph."""
    from torch_rbln.device.context_holder import out_tensor_context

    if is_cpu_fallback_cases((attn_weights, value)):
        return _sdpa_output_fallback(attn_weights, value, dropout_p)

    attn_weights = attn_weights.contiguous()
    value = value.contiguous()

    with out_tensor_context():
        compiled = torch.compile(
            _compile_sdpa_output_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        external_result = compiled(attn_weights, value, dropout_p=dropout_p)

    return external_result


def _sdpa_output_fallback(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    """CPU fallback for SDPA output. Upcasts float16 to float32 for numerical consistency."""
    original_dtype = attn_weights.dtype
    original_device = attn_weights.device

    # Upcast float16 → float32 for numerical consistency
    compute_dtype = torch.float32 if original_dtype == torch.float16 else original_dtype

    weights_cpu = to_cpu(attn_weights).to(compute_dtype)
    v_cpu = to_cpu(value).to(compute_dtype)

    if weights_cpu.dim() == 4 and v_cpu.dim() == 4:
        a_heads, v_heads = weights_cpu.size(1), v_cpu.size(1)
        if a_heads != v_heads:
            v_cpu = _repeat_kv(v_cpu, a_heads // v_heads)

    if dropout_p > 0.0:
        weights_cpu = torch.dropout(weights_cpu, dropout_p, train=True)

    output = torch.matmul(weights_cpu, v_cpu)

    return output.to(original_dtype).to(original_device)


# --- RBLN Backward Compiled Modules ---


def _compile_sdpa_grad_value_fn(attn_weights: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Compile wrapper for: grad_V = attn_weights^T @ grad_output"""
    result = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
    return result


def _compile_sdpa_grad_scores_fn(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compile wrapper for: grad_scores = softmax_backward(grad_output @ V^T, attn_weights) * scale"""
    grad_attn_weights = torch.matmul(grad_output, value.transpose(-2, -1))
    sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_scores = attn_weights * (grad_attn_weights - sum_term) * scale
    return grad_scores


def _compile_sdpa_grad_query_fn(grad_scores: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compile wrapper for: grad_Q = grad_scores @ K"""
    result = torch.matmul(grad_scores, key)
    return result


def _compile_sdpa_grad_key_fn(grad_scores: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Compile wrapper for: grad_K = grad_scores^T @ Q"""
    result = torch.matmul(grad_scores.transpose(-2, -1), query)
    return result


# --- RBLN Backward Computation ---


def _sdpa_backward_compiled(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA backward using 4 compiled graphs."""
    from torch_rbln.device.context_holder import out_tensor_context

    S = key.size(-2)

    # Handle GQA (4D only)
    gqa_enabled = False
    num_kv_heads_orig = None
    n_rep = 1
    if query.dim() == 4:
        q_heads = query.size(1)
        num_kv_heads_orig = key.size(1)
        gqa_enabled = q_heads != num_kv_heads_orig
        if gqa_enabled:
            n_rep = q_heads // num_kv_heads_orig
            key_expanded = _repeat_kv(key, n_rep)
            value_expanded = _repeat_kv(value, n_rep)
        else:
            key_expanded = key
            value_expanded = value
    else:
        key_expanded = key
        value_expanded = value

    args = (grad_output, query, key, value, attn_weights)
    if is_cpu_fallback_cases(args):
        return _sdpa_backward_fallback(grad_output, query, key, value, attn_weights, scale)

    grad_output = grad_output.contiguous()
    query = query.contiguous()
    key_expanded = key_expanded.contiguous()
    value_expanded = value_expanded.contiguous()
    attn_weights = attn_weights.contiguous()

    with out_tensor_context():
        # Graph 1: grad_V = attn_weights^T @ grad_output
        compiled_grad_v = torch.compile(
            _compile_sdpa_grad_value_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        grad_value = compiled_grad_v(attn_weights, grad_output)

        # Graph 2: grad_scores = softmax_backward * scale
        compiled_grad_scores = torch.compile(
            _compile_sdpa_grad_scores_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        grad_scores = compiled_grad_scores(grad_output, value_expanded, attn_weights, scale)

        # Graph 3: grad_Q = grad_scores @ K
        grad_scores = grad_scores.contiguous()
        compiled_grad_q = torch.compile(
            _compile_sdpa_grad_query_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        grad_query = compiled_grad_q(grad_scores, key_expanded)

        # Graph 4: grad_K = grad_scores^T @ Q
        compiled_grad_k = torch.compile(
            _compile_sdpa_grad_key_fn, backend="rbln", dynamic=False, options={"disable_logger": True}
        )
        grad_key = compiled_grad_k(grad_scores, query)

    # Handle GQA gradient reduction
    if gqa_enabled:
        batch, head_dim = grad_key.size(0), grad_key.size(-1)
        grad_key = grad_key.view(batch, num_kv_heads_orig, n_rep, S, head_dim).sum(dim=2)
        grad_value = grad_value.view(batch, num_kv_heads_orig, n_rep, S, head_dim).sum(dim=2)

    return grad_query, grad_key, grad_value


def _sdpa_backward_fallback(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_weights: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU fallback for SDPA backward. Upcasts float16 to float32 to prevent overflow."""
    S = key.size(-2)
    original_device = grad_output.device
    original_dtype = grad_output.dtype

    # Upcast float16 → float32 to prevent overflow
    compute_dtype = torch.float32 if original_dtype == torch.float16 else original_dtype

    grad_output_cpu = to_cpu(grad_output).to(compute_dtype)
    query_cpu = to_cpu(query).to(compute_dtype)
    key_cpu = to_cpu(key).to(compute_dtype)
    value_cpu = to_cpu(value).to(compute_dtype)
    attn_weights_cpu = to_cpu(attn_weights).to(compute_dtype)

    # Handle GQA (4D only)
    gqa_enabled = False
    num_kv_heads_orig = None
    n_rep = 1
    if query_cpu.dim() == 4:
        q_heads = query_cpu.size(1)
        num_kv_heads_orig = key_cpu.size(1)
        gqa_enabled = q_heads != num_kv_heads_orig
        if gqa_enabled:
            n_rep = q_heads // num_kv_heads_orig
            key_cpu = _repeat_kv(key_cpu, n_rep)
            value_cpu = _repeat_kv(value_cpu, n_rep)

    grad_value = torch.matmul(attn_weights_cpu.transpose(-2, -1), grad_output_cpu)
    grad_attn_weights = torch.matmul(grad_output_cpu, value_cpu.transpose(-2, -1))
    sum_term = (grad_attn_weights * attn_weights_cpu).sum(dim=-1, keepdim=True)
    grad_scores = attn_weights_cpu * (grad_attn_weights - sum_term) * scale
    grad_query = torch.matmul(grad_scores, key_cpu)
    grad_key = torch.matmul(grad_scores.transpose(-2, -1), query_cpu)

    # Handle GQA gradient reduction
    if gqa_enabled:
        batch, head_dim = grad_key.size(0), grad_key.size(-1)
        grad_key = grad_key.view(batch, num_kv_heads_orig, n_rep, S, head_dim).sum(dim=2)
        grad_value = grad_value.view(batch, num_kv_heads_orig, n_rep, S, head_dim).sum(dim=2)

    return (
        grad_query.to(original_dtype).to(original_device),
        grad_key.to(original_dtype).to(original_device),
        grad_value.to(original_dtype).to(original_device),
    )


# --- Main Entry Points ---
def scaled_dot_product_fused_attention_overrideable_rbln(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
) -> tuple[
    torch.Tensor,  # output
    torch.Tensor,  # logsumexp
    torch.Tensor,  # cum_seq_q
    torch.Tensor,  # cum_seq_k
    int,  # max_q
    int,  # max_k
    torch.Tensor,  # philox_seed
    torch.Tensor,  # philox_offset
    torch.Tensor,  # debug_attn_mask
]:
    """RBLN implementation of _scaled_dot_product_fused_attention_overrideable.

    Called by PyTorch's SDPA when SDPBackend::overrideable is selected.
    RBLN HW supports 3D and 4D tensors.
    """
    validate_sdpa_input(query, key, value, attn_bias, dropout_p, is_causal, scale)

    # Check RBLN capability
    can_use, reason = can_use_rbln_sdpa(query, key, value, attn_bias, dropout_p, is_causal, scale, enable_gqa=False)
    if not can_use:
        rbln_log_cpu_fallback(f"sdpa_overrideable ({reason})")
        output = _sdpa_cpu_fallback(query, key, value, attn_bias, dropout_p, is_causal, scale, enable_gqa=False)
    else:
        if not query.is_contiguous():
            query = query.contiguous()
        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()
        if attn_bias is not None and not attn_bias.is_contiguous():
            attn_bias = attn_bias.contiguous()

        attn_bias = _convert_bool_mask_to_float(attn_bias, query.dtype, query.device)
        computed_scale = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))

        attn_weights = _sdpa_compute_attn_weights(query, key, attn_bias, is_causal, computed_scale)
        output = _sdpa_compute_output(attn_weights, value, dropout_p)
        _cache_attn_weights(output, attn_weights)

    # Auxiliary tensors for interface (shape depends on input dims)
    if query.dim() == 4:
        batch_size, num_heads, max_seqlen_q = query.size(0), query.size(1), query.size(2)
        max_seqlen_kv = key.size(2)
        logsumexp = torch.empty((batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device=query.device)
        if return_debug_mask:
            debug_attn_mask = torch.empty(
                (batch_size, num_heads, max_seqlen_q, max_seqlen_kv), dtype=torch.float32, device=query.device
            )
        else:
            debug_attn_mask = torch.empty(0, dtype=torch.float32, device=query.device)
    else:
        max_seqlen_q = query.size(-2)
        max_seqlen_kv = key.size(-2)
        logsumexp = torch.empty(0, dtype=torch.float32, device=query.device)
        debug_attn_mask = torch.empty(0, dtype=torch.float32, device=query.device)

    cum_seq_q = torch.empty(0, dtype=torch.int64, device=query.device)
    cum_seq_k = torch.empty(0, dtype=torch.int64, device=query.device)
    philox_seed = torch.empty((), dtype=torch.int64, device="cpu")
    philox_offset = torch.empty((), dtype=torch.int64, device="cpu")

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )


def scaled_dot_product_fused_attention_overrideable_backward_rbln(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    grad_input_mask: list[bool],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RBLN implementation of SDPA backward. Uses cached attn_weights from forward."""
    computed_scale = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))

    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    attn_weights = _get_cached_attn_weights(out)
    use_cpu_fallback = attn_weights is None or _any_tensor_subclass(grad_out, query, key, value, out)

    if use_cpu_fallback:
        if attn_weights is None:
            rbln_log_cpu_fallback("sdpa_backward (cache miss)")
        else:
            rbln_log_cpu_fallback("sdpa_backward (tensor subclass)")
        attn_weights = _sdpa_attn_weights_fallback(query, key, attn_bias, is_causal, computed_scale)
        grad_query, grad_key, grad_value = _sdpa_backward_fallback(
            grad_out, query, key, value, attn_weights, computed_scale
        )
    else:
        grad_query, grad_key, grad_value = _sdpa_backward_compiled(
            grad_out, query, key, value, attn_weights, computed_scale
        )

    # Compute grad_attn_bias if requested
    if grad_input_mask[3] and attn_bias is not None and attn_bias.numel() > 0:
        grad_output_cpu = to_cpu(grad_out)
        value_cpu = to_cpu(value)
        attn_weights_cpu = to_cpu(attn_weights)

        # Handle GQA
        if query.dim() == 4 and value_cpu.dim() == 4:
            q_heads = query.size(1)
            v_heads = value_cpu.size(1)
            if q_heads != v_heads:
                value_cpu = _repeat_kv(value_cpu, q_heads // v_heads)

        grad_attn_weights = torch.matmul(grad_output_cpu, value_cpu.transpose(-2, -1))
        sum_term = (grad_attn_weights * attn_weights_cpu).sum(dim=-1, keepdim=True)
        grad_attn_bias = (attn_weights_cpu * (grad_attn_weights - sum_term)).to(grad_out.device)
    else:
        grad_attn_bias = torch.empty_like(attn_bias) if attn_bias is not None else torch.empty(0, device=query.device)

    return grad_query, grad_key, grad_value, grad_attn_bias
