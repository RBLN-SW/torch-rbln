import math

import torch

from torch_rbln._internal.compile_cache import compile_rbln_cached
from torch_rbln._internal.env_utils import use_device_group_tensor_parallel_size
from torch_rbln._internal.ops_utils import (
    can_use_out_tensor_directly,
    cpu_fallback_path,
    extract_device_id_from_inputs,
    finalize_output_tensor,
    handle_empty_binary,
    is_cpu_fallback_cases,
    is_type_promotion_allowed,
    prepare_args_for_contiguous,
)


class OpModule_softmax(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.softmax(*args, **kwargs)


_softmax_op_module = OpModule_softmax().eval()


def custom_softmax_out_rbln(self, dim: int, half_to_float: bool, *, out=None):
    from torch_rbln.device.context_holder import out_tensor_context

    result_tensor = None

    if is_cpu_fallback_cases(self):
        result = cpu_fallback_path(torch.softmax, (self,), result=out, op_name="aten::softmax", dim=dim)
        result_tensor = result
    else:
        self = self.contiguous()

        # Prepare result tensor and compile options based on out_tensor availability
        # Base compile options: always include eager_mode for eager execution context
        compile_options = {"disable_logger": True}
        # By default, eager mode ops use tp_size=1 due to current compiler structure.
        # If TORCH_RBLN_USE_DEVICE_TP=ON, eager mode ops will follow
        # the logical device size (RBLN_NPUS_PER_DEVICE) like torch.compile operations.
        if not use_device_group_tensor_parallel_size():
            compile_options["tensor_parallel_size"] = 1

        if out is None:
            result_tensor = None
        else:
            # Check if out_tensor can be used directly by compiler
            can_use_out_tensor_directly_flag = can_use_out_tensor_directly((self,), dict({"dim": dim}, out=out))

            if can_use_out_tensor_directly_flag:
                # Use out tensor directly - compiler will write results here
                result_tensor = out
            else:
                result_tensor = None

        with out_tensor_context(result_tensor):
            compiled = compile_rbln_cached(
                _softmax_op_module,
                dynamic=False,
                options=compile_options,
                device_cache_key=self.device.index,
            )
            external_result = compiled(self, dim=dim)
            if result_tensor is None:
                result_tensor = external_result
            elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
                result_tensor.copy_(external_result)

    finalize_output_tensor(out, result_tensor, result_tensor.shape, tuple(self), {})


class OpModule_pow(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.pow(*args, **kwargs)


_pow_op_module = OpModule_pow().eval()


def _is_integer_exponent(exponent) -> bool:
    """Device can only handle integer exponent values (2, 2.0 OK; 2.5 fallback)."""
    if isinstance(exponent, (int, float)):
        return round(exponent) == exponent
    return False


def pow_tensor_scalar_out_rbln(self, exponent, *, out):
    from torch_rbln.device.context_holder import out_tensor_context

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

        compile_options = {"disable_logger": True}
        if not use_device_group_tensor_parallel_size():
            compile_options["tensor_parallel_size"] = 1

        can_use_out_tensor_directly_flag = can_use_out_tensor_directly((self, exponent), {"out": out})

        if can_use_out_tensor_directly_flag:
            result_tensor = out
        else:
            result_tensor = None

        with out_tensor_context(result_tensor):
            compiled = compile_rbln_cached(
                _pow_op_module,
                dynamic=False,
                options=compile_options,
                device_cache_key=self.device.index,
            )
            external_result = compiled(self, exponent)
            if result_tensor is None:
                result_tensor = external_result
            elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
                result_tensor.copy_(external_result)

    finalize_output_tensor(out, result_tensor, result_tensor.shape, (self,), {})


def custom_zero__rbln(self):
    # zeros op is compilable in RBLN, but due to its in-place nature it has problems with graph
    # capture, so it is always processed on the host.
    # Optimization: mark the VMemory as EMPTY_INIT_WITH_ZERO without allocating any host memory.
    # - NPU write-first (e.g. KV-cache output): zero transfer is skipped entirely.
    # - NPU read-first: zeros are transferred via a temporary buffer at transfer time and freed.
    # This avoids permanent host allocation for large tensors such as KV-cache.
    if self.numel() == 0:
        return

    from torch_rbln._C import _mark_zeros

    _mark_zeros(self.data_ptr())


class custom_rbln_paged_attn_prefill(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: rtosa.multiply cannot accept tensor scalar value. scale must be constant tensor.
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        return torch.ops.rbln_custom_ops.paged_attn_prefill(
            args[0],  # q [1,2,6,33,128]
            args[1],  # k [1,2,1,33,128]
            args[2],  # v [1,2,1,33,128]
            args[3],  # attn_mask [1,33,1]
            args[4],  # kcache [1,1,1]
            args[5],  # vcache [1,1,1]
            args[6],  # seq [1,1]
            scale,  # scale, float32
            args[8],  # block_table
            args[9],  # block_size, int
        )


_paged_attn_prefill_op_module = custom_rbln_paged_attn_prefill().eval()


def paged_attn_prefill_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import out_tensor_context

    if len(args) != 10:
        raise RuntimeError("paged_attn_prefill takes 10 inputs.")

    # Check if batch size (dim=0) is 1 for Q, K, and V tensors.
    # This kernel is constrained to operate on a batch size of 1.
    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        assert tensor.size(0) == 1, (
            f"Custom kernel with prefill must batch size of {name} must be 1, but got shape {tensor.shape}"
        )

    # origin_dim = args[0].size(-1)
    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)

    # The K/V cache tensors (args[4], args[5]) are allocated externally, often
    # as placeholders. This prefill operation is the first time they are populated.
    # We must explicitly set their metadata dtype to custom_float16 here, as the
    # underlying kernel will fill the cache with data in the custom_float16 format,
    # regardless of the tensor's original torch.dtype.
    k_cache = contig_args[4]
    v_cache = contig_args[5]

    # [Compiler Constraint] Check K/V cache alignment.
    # Due to current compiler limitations, the K/V-cache tensors
    # must be allocated externally with their last dimension being a multiple of 64.
    # This alignment is required for the optimized kernel to function correctly.
    assert k_cache.size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {k_cache.shape}"
    )
    assert v_cache.size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {v_cache.shape}"
    )

    # Create aligned result tensor with 64-byte alignment for last dimension
    result_tensor = torch.empty(contig_args[0].shape, dtype=torch.float16, device=contig_args[0].device)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        # tensor_parallel_size=1 is hardcoded due to custom kernel compiler constraints.
        compiled = compile_rbln_cached(
            _paged_attn_prefill_op_module,
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
            device_cache_key=extract_device_id_from_inputs(*contig_args, **contig_kwargs),
        )
        external_result = compiled(*contig_args, **contig_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


class custom_rbln_paged_attn_decode(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: rtosa.multiply cannot accept tensor scalar value. scale must be constant tensor.
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        return torch.ops.rbln_custom_ops.paged_attn_decode(
            args[0],  # q [1,2,6,33,128]
            args[1],  # k [1,2,1,33,128]
            args[2],  # v [1,2,1,33,128]
            args[3],  # attn_mask [1,33,1]
            args[4],  # kcache [1,1,1]
            args[5],  # vcache [1,1,1]
            args[6],  # seq [1,1]
            scale,  # scale, float32
            args[8],  # block_table
            args[9],  # block_size, int
        )


_paged_attn_decode_op_module = custom_rbln_paged_attn_decode().eval()


def paged_attn_decode_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import out_tensor_context

    if len(args) != 10:
        raise RuntimeError("paged_attn_prefill takes 10 inputs.")

    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)

    k_cache = contig_args[4]
    v_cache = contig_args[5]

    # [Compiler Constraint] Check K/V cache alignment.
    # Due to current compiler limitations, the K/V-cache tensors
    # must be allocated externally with their last dimension being a multiple of 64.
    # This alignment is required for the optimized kernel to function correctly.
    assert k_cache.size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {k_cache.shape}"
    )
    assert v_cache.size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {v_cache.shape}"
    )

    # Create aligned result tensor with 64-byte alignment for last dimension
    result_tensor = torch.empty(contig_args[0].shape, dtype=torch.float16, device=contig_args[0].device)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        # tensor_parallel_size=1 is hardcoded due to custom kernel compiler constraints.
        compiled = compile_rbln_cached(
            _paged_attn_decode_op_module,
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
            device_cache_key=extract_device_id_from_inputs(*contig_args, **contig_kwargs),
        )
        external_result = compiled(*contig_args, **contig_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


class custom_rbln_paged_causal_attn_prefill(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: rtosa.multiply cannot accept tensor scalar value. scale must be constant tensor.
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        # paged_causal_attn_prefill: q, k, v, kcache, vcache, seq, scale, block_table, block_size, is_bidirectional, mask (optional)
        # args[0]: q, args[1]: k, args[2]: v, args[3]: kcache, args[4]: vcache, args[5]: seq,
        # args[6]: scale (computed), args[7]: block_table, args[8]: block_size, args[9]: is_bidirectional, args[10]: mask (optional)
        call_args = [
            args[0],  # q [1,2,6,33,128]
            args[1],  # k [1,2,1,33,128]
            args[2],  # v [1,2,1,33,128]
            args[3],  # kcache [1,1,1]
            args[4],  # vcache [1,1,1]
            args[5],  # seq [1,1]
            scale,  # scale, float32
            args[7],  # block_table
            args[8],  # block_size, int
            args[9],  # is_bidirectional, bool
        ]
        # mask is optional
        if len(args) > 10 and args[10] is not None:
            call_args.append(args[10])  # mask
        return torch.ops.rbln_custom_ops.paged_causal_attn_prefill(*call_args)


_paged_causal_attn_prefill_op_module = custom_rbln_paged_causal_attn_prefill().eval()


def paged_causal_attn_prefill_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import out_tensor_context

    # paged_causal_attn_prefill: q, k, v, kcache, vcache, seq, scale, block_table, block_size, is_bidirectional, mask (optional)
    if len(args) < 10 or len(args) > 11:
        raise RuntimeError(f"paged_causal_attn_prefill takes 10 or 11 inputs, but got {len(args)}.")

    # Check if batch size (dim=0) is 1 for Q, K, and V tensors.
    # This kernel is constrained to operate on a batch size of 1.
    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        assert tensor.size(0) == 1, (
            f"Custom kernel with prefill must batch size of {name} must be 1, but got shape {tensor.shape}"
        )

    # origin_dim = args[0].size(-1)
    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)

    # The K/V cache tensors (args[3], args[4]) are allocated externally, often
    # as placeholders. This prefill operation is the first time they are populated.
    # We must explicitly set their metadata dtype to custom_float16 here, as the
    # underlying kernel will fill the cache with data in the custom_float16 format,
    # regardless of the tensor's original torch.dtype.
    k_cache = contig_args[3]
    v_cache = contig_args[4]

    # [Compiler Constraint] Check K/V cache alignment.
    # Due to current compiler limitations, the K/V-cache tensors
    # must be allocated externally with their last dimension being a multiple of 64.
    # This alignment is required for the optimized kernel to function correctly.
    assert k_cache.size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {k_cache.shape}"
    )
    assert v_cache.size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {v_cache.shape}"
    )

    # Create aligned result tensor with 64-byte alignment for last dimension
    result_tensor = torch.empty(contig_args[0].shape, dtype=torch.float16, device=contig_args[0].device)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        # tensor_parallel_size=1 is hardcoded due to custom kernel compiler constraints.
        compiled = compile_rbln_cached(
            _paged_causal_attn_prefill_op_module,
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
            device_cache_key=extract_device_id_from_inputs(*contig_args, **contig_kwargs),
        )
        external_result = compiled(*contig_args, **contig_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


class custom_rbln_paged_causal_attn_decode(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # TODO: rtosa.multiply cannot accept tensor scalar value. scale must be constant tensor.
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        # paged_causal_attn_decode: q, k, v, kcache, vcache, seq, scale, block_table, block_size, mask (optional)
        # args[0]: q, args[1]: k, args[2]: v, args[3]: kcache, args[4]: vcache, args[5]: seq,
        # args[6]: scale (computed), args[7]: block_table, args[8]: block_size, args[9]: mask (optional)
        call_args = [
            args[0],  # q [1,2,6,33,128]
            args[1],  # k [1,2,1,33,128]
            args[2],  # v [1,2,1,33,128]
            args[3],  # kcache [1,1,1]
            args[4],  # vcache [1,1,1]
            args[5],  # seq [1,1]
            scale,  # scale, float32
            args[7],  # block_table
            args[8],  # block_size, int
        ]
        # mask is optional
        if len(args) > 9 and args[9] is not None:
            call_args.append(args[9])  # mask
        return torch.ops.rbln_custom_ops.paged_causal_attn_decode(*call_args)


_paged_causal_attn_decode_op_module = custom_rbln_paged_causal_attn_decode().eval()


def paged_causal_attn_decode_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import out_tensor_context

    # paged_causal_attn_decode: q, k, v, kcache, vcache, seq, scale, block_table, block_size, mask (optional)
    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"paged_causal_attn_decode takes 9 or 10 inputs, but got {len(args)}.")

    (contig_args, contig_kwargs), changed_any = prepare_args_for_contiguous(args, kwargs)

    k_cache = contig_args[3]
    v_cache = contig_args[4]

    # [Compiler Constraint] Check K/V cache alignment.
    # Due to current compiler limitations, the K/V-cache tensors
    # must be allocated externally with their last dimension being a multiple of 64.
    # This alignment is required for the optimized kernel to function correctly.
    assert k_cache.size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {k_cache.shape}"
    )
    assert v_cache.size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {v_cache.shape}"
    )

    # Create aligned result tensor with 64-byte alignment for last dimension
    result_tensor = torch.empty(contig_args[0].shape, dtype=torch.float16, device=contig_args[0].device)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        # tensor_parallel_size=1 is hardcoded due to custom kernel compiler constraints.
        compiled = compile_rbln_cached(
            _paged_causal_attn_decode_op_module,
            dynamic=False,
            options={"disable_logger": True, "tensor_parallel_size": 1},
            device_cache_key=extract_device_id_from_inputs(*contig_args, **contig_kwargs),
        )
        external_result = compiled(*contig_args, **contig_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


rbln_custom_impl = torch.library.Library("rbln_custom_ops", "IMPL")  # noqa: TOR901
rbln_custom_impl.impl("paged_attn_prefill", paged_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_attn_decode", paged_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_prefill", paged_causal_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_decode", paged_causal_attn_decode_rbln, "PrivateUse1")
