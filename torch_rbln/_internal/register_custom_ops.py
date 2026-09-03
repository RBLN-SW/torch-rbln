import math

import torch

from torch_rbln._internal.compile_cache import compile_rbln_cached
from torch_rbln._internal.ops_utils import (
    compile_and_run_view_aware,
    cpu_fallback_path,
    extract_device_id_from_inputs,
    finalize_output_tensor,
    get_view_op_module,
    handle_empty_binary,
    is_cpu_fallback_cases,
    is_type_promotion_allowed,
    prepare_args_view_aware,
)


def custom_softmax_out_rbln(self, dim: int, half_to_float: bool, *, out=None):
    if is_cpu_fallback_cases(self):
        result_tensor = cpu_fallback_path(torch.softmax, (self,), result=out, op_name="aten::softmax", dim=dim)
    else:
        # ``dim`` is a positional kwarg (int) — passed through to torch.softmax
        # inside the wrapper OpModule. View-aware path detects any non-contig
        # ``self`` and lowers the view step (permute / narrow / etc.) on
        # device alongside softmax.
        result_tensor = compile_and_run_view_aware(
            torch.softmax,
            "aten::softmax",
            (self,),
            {"dim": dim},
            out,
        )

    finalize_output_tensor(out, result_tensor, result_tensor.shape, tuple(self), {})


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

    if self.numel() == 0:
        result_tensor, _ = handle_empty_binary((self,))
    elif not _is_integer_exponent(exponent) or is_cpu_fallback_cases((self, exponent)):
        result_tensor = cpu_fallback_path(torch.pow, (self, exponent), result=out, op_name="aten::pow")
    else:
        # ``exponent`` is a Python scalar so it flows through unchanged; the
        # view-aware helper detects any non-contig ``self`` and dispatches the
        # view step on device alongside pow.
        result_tensor = compile_and_run_view_aware(
            torch.pow,
            "aten::pow",
            (self, exponent),
            {},
            out,
        )

    finalize_output_tensor(out, result_tensor, result_tensor.shape, (self,), {})


def _dispatch_custom_kernel(op_module, args, view_recipes):
    """Compile the kernel module through the rebel backend and run it."""
    compiled = compile_rbln_cached(
        op_module,
        dynamic=False,
        options={"disable_logger": True, "num_devices": 1},
        device_cache_key=(extract_device_id_from_inputs(*args), view_recipes),
    )
    return compiled(*args)


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
    if len(args) != 10:
        raise RuntimeError("paged_attn_prefill takes 10 inputs.")

    # Check if batch size (dim=0) is 1 for Q, K, and V tensors.
    # This kernel is constrained to operate on a batch size of 1.
    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        assert tensor.size(0) == 1, (
            f"Custom kernel with prefill must batch size of {name} must be 1, but got shape {tensor.shape}"
        )

    # K/V cache alignment must be checked on the user-visible shape (caches are
    # always contig in production; if they are non-contig views the recipe
    # would have changed the underlying base shape — caller bug — but we keep
    # the assert against ``args[K]`` so the error message is still meaningful).
    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[4].shape}"
    )
    assert args[5].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[5].shape}"
    )

    # View-aware arg prep: pure-permute / single-step views become an explicit
    # aten op inside the wrapper OpModule's forward; the rebel-backend lowers
    # them on device. Other views fall back to ``.contiguous()`` (one-time
    # warning emitted via ``_maybe_warn_view_fallback``).
    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_paged_attn_prefill_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


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
    if len(args) != 10:
        raise RuntimeError("paged_attn_prefill takes 10 inputs.")

    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[4].shape}"
    )
    assert args[5].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[5].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_paged_attn_decode_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


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
    # paged_causal_attn_prefill: q, k, v, kcache, vcache, seq, scale, block_table, block_size, is_bidirectional, mask (optional)
    if len(args) < 10 or len(args) > 11:
        raise RuntimeError(f"paged_causal_attn_prefill takes 10 or 11 inputs, but got {len(args)}.")

    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        assert tensor.size(0) == 1, (
            f"Custom kernel with prefill must batch size of {name} must be 1, but got shape {tensor.shape}"
        )

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[3].shape}"
    )
    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[4].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_paged_causal_attn_prefill_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


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
    # paged_causal_attn_decode: q, k, v, kcache, vcache, seq, scale, block_table, block_size, mask (optional)
    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"paged_causal_attn_decode takes 9 or 10 inputs, but got {len(args)}.")

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[3].shape}"
    )
    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[4].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_paged_causal_attn_decode_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


class custom_rbln_flash_attention_naive_prefill(torch.nn.Module):
    def forward(self, *args, **kwargs):
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        call_args = [
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            scale,
            args[6],
            args[7],
            args[8],
        ]
        if len(args) > 9 and args[9] is not None:
            call_args.append(args[9])
        return torch.ops.rbln_custom_ops.flash_attention_naive_prefill(*call_args)


_flash_attention_naive_prefill_op_module = custom_rbln_flash_attention_naive_prefill().eval()


def flash_attention_naive_prefill_rbln(*args, **kwargs):
    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"flash_attention_naive_prefill takes 9 or 10 inputs (optional sinks), but got {len(args)}.")

    for i, name in enumerate(["Query (q)", "Key (k)", "Value (v)"]):
        tensor = args[i]
        assert tensor.size(0) == 1, (
            f"flash_attention_naive_prefill batch size of {name} must be 1, but got shape {tensor.shape}"
        )

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of kv_cache must be a multiple of 64, but got shape {args[3].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_flash_attention_naive_prefill_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


class custom_rbln_flash_attention_naive_decode(torch.nn.Module):
    def forward(self, *args, **kwargs):
        scale = torch.tensor(1 / math.sqrt(args[0].size(-1)))
        call_args = [
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            scale,
            args[6],
            args[7],
            args[8],
        ]
        if len(args) > 9 and args[9] is not None:
            call_args.append(args[9])
        return torch.ops.rbln_custom_ops.flash_attention_naive_decode(*call_args)


_flash_attention_naive_decode_op_module = custom_rbln_flash_attention_naive_decode().eval()


def flash_attention_naive_decode_rbln(*args, **kwargs):
    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"flash_attention_naive_decode takes 9 or 10 inputs (optional sinks), but got {len(args)}.")

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of kv_cache must be a multiple of 64, but got shape {args[3].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    assert args[0].size(-1) % 64 == 0

    op_module = get_view_op_module(_flash_attention_naive_decode_op_module, view_recipes)
    return _dispatch_custom_kernel(op_module, view_args, view_recipes)


rbln_custom_impl = torch.library.Library("rbln_custom_ops", "IMPL")  # noqa: TOR901
rbln_custom_impl.impl("paged_attn_prefill", paged_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_attn_decode", paged_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_prefill", paged_causal_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_decode", paged_causal_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("flash_attention_naive_prefill", flash_attention_naive_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("flash_attention_naive_decode", flash_attention_naive_decode_rbln, "PrivateUse1")
