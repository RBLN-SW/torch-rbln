import math

import torch

from torch_rbln._internal.log_utils import rbln_log_cpu_fallback
from torch_rbln._internal.ops_utils import (
    compile_and_execute,
    cpu_fallback_path,
    finalize_output_tensor,
    handle_empty_binary,
    is_cpu_fallback_cases,
    is_type_promotion_allowed,
    make_op_module,
    prepare_args_for_contiguous,
)


_softmax_op_module = make_op_module(torch.softmax)


def custom_softmax_out_rbln(self, dim: int, half_to_float: bool, *, out=None):
    result_tensor = None

    if is_cpu_fallback_cases(self):
        result_tensor = cpu_fallback_path(torch.softmax, (self,), result=out, op_name="aten::softmax", dim=dim)
    else:
        self = self.contiguous()
        result_tensor = compile_and_execute(_softmax_op_module, (self,), {"dim": dim}, out_tensor=out)

    finalize_output_tensor(out, result_tensor, result_tensor.shape, tuple(self), {})


_pow_op_module = make_op_module(torch.pow)


def _is_integer_exponent(exponent) -> bool:
    """Device can only handle integer exponent values (2, 2.0 OK; 2.5 fallback)."""
    if isinstance(exponent, (int, float)):
        return round(exponent) == exponent
    return False


def pow_tensor_scalar_out_rbln(self, exponent, *, out):
    # pow.Tensor_Scalar: exponent must be scalar (int/float), not tensor
    if isinstance(exponent, torch.Tensor):
        raise RuntimeError("pow.Tensor_Scalar expects scalar exponent (int/float), not tensor.")

    # Validate device/dtype match (preserved from original implementation)
    if self.device != out.device:
        raise RuntimeError(f"Input device {self.device} does not match output device {out.device}.")
    if self.dtype != out.dtype and not is_type_promotion_allowed((self,), out):
        raise RuntimeError(f"Unsafe cast: input has dtype {self.dtype} but output tensor has dtype {out.dtype}.")

    result_tensor = None

    if self.numel() == 0:
        result_tensor, _ = handle_empty_binary((self,))
    elif not _is_integer_exponent(exponent) or is_cpu_fallback_cases((self, exponent)):
        result_tensor = cpu_fallback_path(torch.pow, (self, exponent), result=out, op_name="aten::pow")
    else:
        self = self.contiguous()
        result_tensor = compile_and_execute(_pow_op_module, (self, exponent), {}, out_tensor=out)

    finalize_output_tensor(out, result_tensor, result_tensor.shape, (self,), {})


def custom_zero__rbln(self):
    # zeros op is compilable in RBLN, but due to its in-place, it has problems with graph capture,
    # so it is always processed on the host.
    cpu_self = torch.empty_like(self, device=torch.device("cpu"))
    rbln_log_cpu_fallback("aten::zero")
    result_tensor = torch.zero_(cpu_self)

    finalize_output_tensor(self, result_tensor, result_tensor.shape, self, {})


# ---------------------------------------------------------------------------
# Paged attention custom kernels
# ---------------------------------------------------------------------------


def _validate_kv_cache_alignment(k_cache, v_cache, alignment=64):
    """Validate that K/V cache tensors have their last dimension aligned to the given multiple."""
    if k_cache.size(-1) % alignment != 0:
        raise ValueError(
            f"The last dimension of K-cache must be a multiple of {alignment}, but got shape {k_cache.shape}"
        )
    if v_cache.size(-1) % alignment != 0:
        raise ValueError(
            f"The last dimension of V-cache must be a multiple of {alignment}, but got shape {v_cache.shape}"
        )


def _validate_prefill_batch_size(args):
    """Validate that Q, K, V tensors have batch size of 1 (prefill constraint)."""
    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        if tensor.size(0) != 1:
            raise ValueError(
                f"Custom kernel with prefill: batch size of {name} must be 1, but got shape {tensor.shape}"
            )


def _compile_and_execute_kernel(op_module_factory, contig_args, contig_kwargs):
    """Compile an attention kernel module and execute it, returning the result tensor.

    Unlike :func:`compile_and_execute`, this always uses tp_size=1 (custom kernel
    compiler constraint) and allocates a fresh result tensor matching the query shape.
    """
    from torch_rbln.device.context_holder import out_tensor_context

    result_tensor = torch.empty(contig_args[0].shape, dtype=torch.float16, device=contig_args[0].device)

    with out_tensor_context(result_tensor):
        compiled = torch.compile(
            op_module_factory(),
            backend="rbln",
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
        )
        external_result = compiled(*contig_args, **contig_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


class custom_rbln_paged_attn_prefill(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: rtosa.multiply cannot accept tensor scalar value. scale must be constant tensor.
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        return torch.ops.rbln_custom_ops.paged_attn_prefill(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], scale, args[8], args[9]
        )


def paged_attn_prefill_rbln(*args, **kwargs):
    if len(args) != 10:
        raise RuntimeError("paged_attn_prefill takes 10 inputs.")

    _validate_prefill_batch_size(args)
    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)
    _validate_kv_cache_alignment(contig_args[4], contig_args[5])

    return _compile_and_execute_kernel(custom_rbln_paged_attn_prefill, contig_args, contig_kwargs)


class custom_rbln_paged_attn_decode(torch.nn.Module):
    def forward(self, *args, **kwargs):
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        return torch.ops.rbln_custom_ops.paged_attn_decode(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], scale, args[8], args[9]
        )


def paged_attn_decode_rbln(*args, **kwargs):
    if len(args) != 10:
        raise RuntimeError("paged_attn_decode takes 10 inputs.")

    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)
    _validate_kv_cache_alignment(contig_args[4], contig_args[5])

    return _compile_and_execute_kernel(custom_rbln_paged_attn_decode, contig_args, contig_kwargs)


class custom_rbln_paged_causal_attn_prefill(torch.nn.Module):
    def forward(self, *args, **kwargs):
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        call_args = [args[0], args[1], args[2], args[3], args[4], args[5], scale, args[7], args[8], args[9]]
        if len(args) > 10 and args[10] is not None:
            call_args.append(args[10])
        return torch.ops.rbln_custom_ops.paged_causal_attn_prefill(*call_args)


def paged_causal_attn_prefill_rbln(*args, **kwargs):
    if len(args) < 10 or len(args) > 11:
        raise RuntimeError(f"paged_causal_attn_prefill takes 10 or 11 inputs, but got {len(args)}.")

    _validate_prefill_batch_size(args)
    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)
    _validate_kv_cache_alignment(contig_args[3], contig_args[4])

    return _compile_and_execute_kernel(custom_rbln_paged_causal_attn_prefill, contig_args, contig_kwargs)


class custom_rbln_paged_causal_attn_decode(torch.nn.Module):
    def forward(self, *args, **kwargs):
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        call_args = [args[0], args[1], args[2], args[3], args[4], args[5], scale, args[7], args[8]]
        if len(args) > 9 and args[9] is not None:
            call_args.append(args[9])
        return torch.ops.rbln_custom_ops.paged_causal_attn_decode(*call_args)


def paged_causal_attn_decode_rbln(*args, **kwargs):
    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"paged_causal_attn_decode takes 9 or 10 inputs, but got {len(args)}.")

    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)
    _validate_kv_cache_alignment(contig_args[3], contig_args[4])

    return _compile_and_execute_kernel(custom_rbln_paged_causal_attn_decode, contig_args, contig_kwargs)


rbln_custom_impl = torch.library.Library("rbln_custom_ops", "IMPL")  # noqa: TOR901
rbln_custom_impl.impl("paged_attn_prefill", paged_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_attn_decode", paged_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_prefill", paged_causal_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_decode", paged_causal_attn_decode_rbln, "PrivateUse1")
