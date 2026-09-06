import math
import threading
from collections import OrderedDict

import torch

from torch_rbln._internal.compile_cache import compile_rbln_cached
from torch_rbln._internal.ops_utils import (
    _detect_view_recipe,
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

    # Result shape, dtype, and device match Q's user-visible (post-view) form.
    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        op_module = get_view_op_module(_paged_attn_prefill_op_module, view_recipes)
        compiled = compile_rbln_cached(
            op_module,
            dynamic=False,
            options={"disable_logger": True, "num_devices": 1},
            device_cache_key=(extract_device_id_from_inputs(*view_args, **view_kwargs), view_recipes),
        )
        external_result = compiled(*view_args, **view_kwargs)
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

    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[4].shape}"
    )
    assert args[5].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[5].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        op_module = get_view_op_module(_paged_attn_decode_op_module, view_recipes)
        compiled = compile_rbln_cached(
            op_module,
            dynamic=False,
            options={"disable_logger": True, "num_devices": 1},
            device_cache_key=(extract_device_id_from_inputs(*view_args, **view_kwargs), view_recipes),
        )
        external_result = compiled(*view_args, **view_kwargs)
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

    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        op_module = get_view_op_module(_paged_causal_attn_prefill_op_module, view_recipes)
        compiled = compile_rbln_cached(
            op_module,
            dynamic=False,
            options={"disable_logger": True, "num_devices": 1},
            device_cache_key=(extract_device_id_from_inputs(*view_args, **view_kwargs), view_recipes),
        )
        external_result = compiled(*view_args, **view_kwargs)
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

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of K-cache must be a multiple of 64, but got shape {args[3].shape}"
    )
    assert args[4].size(-1) % 64 == 0, (
        f"The last dimension of V-cache must be a multiple of 64, but got shape {args[4].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    with out_tensor_context(result_tensor):
        op_module = get_view_op_module(_paged_causal_attn_decode_op_module, view_recipes)
        compiled = compile_rbln_cached(
            op_module,
            dynamic=False,
            options={"disable_logger": True, "num_devices": 1},
            device_cache_key=(extract_device_id_from_inputs(*view_args, **view_kwargs), view_recipes),
        )
        external_result = compiled(*view_args, **view_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)

    return result_tensor


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


def flash_attention_naive_prefill_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import helper

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

    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    helper.set_out_tensor(result_tensor)
    base_module = custom_rbln_flash_attention_naive_prefill()
    op_module = get_view_op_module(base_module, view_recipes)
    compiled = torch.compile(
        op_module,
        backend="rbln",
        dynamic=False,
        options={"disable_logger": True, "num_devices": 1},
    )
    external_result = compiled(*view_args, **view_kwargs)
    if result_tensor is None:
        result_tensor = external_result
    elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
        result_tensor.copy_(external_result)
    helper.clear_out_tensor()

    return result_tensor


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


def flash_attention_naive_decode_rbln(*args, **kwargs):
    from torch_rbln.device.context_holder import helper

    if len(args) < 9 or len(args) > 10:
        raise RuntimeError(f"flash_attention_naive_decode takes 9 or 10 inputs (optional sinks), but got {len(args)}.")

    assert args[3].size(-1) % 64 == 0, (
        f"The last dimension of kv_cache must be a multiple of 64, but got shape {args[3].shape}"
    )

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs)

    result_tensor = torch.empty_like(args[0], memory_format=torch.contiguous_format)
    assert result_tensor.size(-1) % 64 == 0

    helper.set_out_tensor(result_tensor)
    base_module = custom_rbln_flash_attention_naive_decode()
    op_module = get_view_op_module(base_module, view_recipes)
    compiled = torch.compile(
        op_module,
        backend="rbln",
        dynamic=False,
        options={"disable_logger": True, "num_devices": 1},
    )
    external_result = compiled(*view_args, **view_kwargs)
    if result_tensor is None:
        result_tensor = external_result
    elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
        result_tensor.copy_(external_result)
    helper.clear_out_tensor()

    return result_tensor


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """The op under the view recipe: lay the replayed view out contiguously.

    ``torch.clone`` would be the obvious choice and is wrong here -- it preserves the
    source's memory format, so cloning a permuted view hands back the same strides, not
    the contiguous destination layout ``copy_`` was asked for.
    """
    return t.contiguous()


# A compiled program binds its I/O once, and rebinding it to another buffer is not a
# cheap patch: the relocated instruction streams are re-uploaded through the allocator,
# which first waits for every transfer still in flight on the device. A caller that
# alternates buffers -- a transfer pipeline staging blocks in turn -- would pay that on
# every call, and pay it with the overlap its pipeline exists for. So each source address
# gets its own program. The compile cache is keyed on (device, shapes, dtypes) and on the
# identity of the callable under the recipe, never on an address, so the callable is
# what carries the buffer identity here: one distinct function object per address, which
# ``get_view_op_module`` then keys on.
#
# Bounded, so an unbounded caller cannot grow the cache without limit; past the cap the
# least recently used address gives up its slot (a program's rebind, once) rather than
# two live addresses sharing a slot and rebinding on every call. The cap covers a
# transfer pool of four threads, each staging into two pipeline slots of four chiplet
# shards: 4 x 2 x 4 = 32 live buffers when a thread's two directions share them, half
# the cap. A slot costs a compile the first time it is used (~0.5 s).
_MAX_COPY_SLOTS = 64
_copy_slot_of_src: OrderedDict[int, int] = OrderedDict()  # address -> slot, least recent first
_copy_slot_ops: dict[int, object] = {}
_copy_slot_lock = threading.Lock()


def _copy_slot_for(ptr: int) -> int:
    """The program slot for a source address: its own while it is one of the
    ``_MAX_COPY_SLOTS`` most recently used, else the least recently used one's."""
    slot = _copy_slot_of_src.get(ptr)
    if slot is not None:
        _copy_slot_of_src.move_to_end(ptr)
        return slot
    if len(_copy_slot_of_src) < _MAX_COPY_SLOTS:
        slot = len(_copy_slot_of_src)
    else:
        _, slot = _copy_slot_of_src.popitem(last=False)
    _copy_slot_of_src[ptr] = slot
    return slot


def _materialize_for(src: torch.Tensor):
    """``_materialize`` for this source buffer, one callable per address.

    ``torch.compile`` caches on the code object, so each slot is compiled from its own
    source -- the same reason ``compiled_permute`` generates one function per slot.
    """
    with _copy_slot_lock:
        slot = _copy_slot_for(src.data_ptr())
        op = _copy_slot_ops.get(slot)
        if op is None:
            namespace: dict = {}
            exec(  # noqa: S102
                compile(
                    "def _materialize_slot(t):\n    return t.contiguous()\n",
                    f"<copy_strided_view slot {slot}>",
                    "exec",
                ),
                {},
                namespace,
            )
            op = _copy_slot_ops[slot] = namespace["_materialize_slot"]
        return op


def copy_strided_view_rbln(src, out) -> bool:
    """``out.copy_(src)`` for a strided device ``src``, as one compiled device program.

    ``copy_``'s device->device path calls this instead of walking the copy one contiguous
    run at a time. The view chain is replayed *inside* the compiled graph by the same
    machinery every other view-aware op uses (``prepare_args_view_aware`` classifies it,
    ``get_view_op_module`` replays it), so the device reads the base buffer and writes
    ``out`` in one program -- no descriptor per run, no host round-trip. Nothing here is
    specific to a permutation: whatever the detector can classify, this covers.

    Returns False when the detector cannot classify ``src``, leaving ``copy_`` on its
    existing route. That check has to happen before ``compile_and_run_view_aware``:
    unclassified views take ``.contiguous()`` in there, which re-enters ``copy_``.
    """
    if _detect_view_recipe(src) is None:
        return False
    result = compile_and_run_view_aware(
        _materialize_for(src), "torch_rbln::copy_strided_view", (src,), {}, out
    )
    if result is not None and result.data_ptr() != out.data_ptr():
        # The compile path only writes the caller's buffer for dtypes in
        # SupportedDtypes.dispatch; otherwise it hands back its own. Both are contiguous
        # here, so this lands on copy_'s direct memcpy and cannot recurse.
        out.copy_(result)
    return True


# Geometries (sizes, strides, storage offset, dtype, device) for which the compiled view
# copy has been shown to permute a buffer in place correctly. Writing the buffer a program
# reads is only correct when its schedule finishes reading a region before writing it --
# a property of the compiler's tiling, not of anything here -- so the first in-place copy
# of a geometry is checked against a host reference and the verdict remembered.
_inplace_view_copy_verified: dict[tuple, bool] = {}
_inplace_view_copy_lock = threading.Lock()


def _inplace_view_copy_key(src: torch.Tensor) -> tuple:
    return (tuple(src.shape), tuple(src.stride()), src.storage_offset(), src.dtype, str(src.device))


def _run_view_copy(src: torch.Tensor, out: torch.Tensor, allow_alias: bool) -> None:
    result = compile_and_run_view_aware(
        _materialize_for(src), "torch_rbln::copy_strided_view", (src,), {}, out, allow_alias=allow_alias
    )
    if result is not None and result.data_ptr() != out.data_ptr():
        # The compile path only writes the caller's buffer for dtypes in
        # SupportedDtypes.dispatch; otherwise it hands back its own. Both are contiguous
        # here, so this lands on copy_'s direct memcpy and cannot recurse.
        out.copy_(result)


def _verify_inplace_geometry(src: torch.Tensor, out: torch.Tensor) -> None:
    """Run the first in-place copy of a geometry against a host reference, once.

    The reference costs a device-to-host copy rather than a second device buffer, which
    is what the in-place form exists to avoid. It uses the caller's data, so a constant
    buffer proves less than a varied one.
    """
    key = _inplace_view_copy_key(src)
    verified = _inplace_view_copy_verified.get(key)
    if verified is None:
        with _inplace_view_copy_lock:
            verified = _inplace_view_copy_verified.get(key)
            if verified is None:
                # Read the source view to the host first: it is both the reference and
                # the last chance to see the input, since the copy overwrites it.
                reference = src.cpu()
                _run_view_copy(src, out, allow_alias=True)
                verified = torch.equal(out.cpu(), reference)
                _inplace_view_copy_verified[key] = verified
                if not verified:
                    raise _inplace_unsupported(src)
                return
    if not verified:
        raise _inplace_unsupported(src)
    _run_view_copy(src, out, allow_alias=True)


def _inplace_unsupported(src: torch.Tensor) -> RuntimeError:
    return RuntimeError(
        f"copy_strided_view(inplace=True): the compiled program does not permute "
        f"{list(src.shape)} (strides {list(src.stride())}) correctly in place; copy into a "
        "separate buffer instead"
    )


def copy_strided_view_rbln(src, out, inplace=False) -> bool:
    """``out.copy_(src)`` for a strided device ``src``, as one compiled device program.

    ``copy_``'s device->device path calls this instead of walking the copy one contiguous
    run at a time. The view chain is replayed *inside* the compiled graph by the same
    machinery every other view-aware op uses (``prepare_args_view_aware`` classifies it,
    ``get_view_op_module`` replays it), so the device reads the base buffer and writes
    ``out`` in one program -- no descriptor per run, no host round-trip. Nothing here is
    specific to a permutation: whatever the detector can classify, this covers.

    ``inplace`` says ``out`` is the very buffer ``src`` views, so the program permutes it
    in its own storage and a caller that only needs the permuted layout holds one buffer
    instead of two. ``copy_`` never asks for it -- ATen refuses partially overlapping
    pairs before dispatch -- so an aliasing pair is always a caller's explicit intent.
    ``out`` must then be the whole storage ``src`` views: contiguous, storage offset 0,
    the same base address and byte size. Correctness rests on the compiled schedule
    finishing its reads of a region before writing it, which is the compiler's tiling
    rather than anything here, so the first call for a geometry is checked against a host
    reference and a geometry that fails raises.

    Returns False when the detector cannot classify ``src``, leaving the caller on its
    existing route. Raises RuntimeError when ``inplace`` is set and ``out`` is not
    ``src``'s whole storage, or the program does not permute that geometry in place.
    """
    if _detect_view_recipe(src) is None:
        return False
    if not inplace:
        _run_view_copy(src, out, allow_alias=False)
        return True
    base_ptr = src.data_ptr() - src.storage_offset() * src.element_size()
    if (
        out.data_ptr() != base_ptr
        or out.storage_offset() != 0
        or not out.is_contiguous()
        or out.numel() * out.element_size() != src.storage().nbytes()
    ):
        raise RuntimeError(
            "copy_strided_view(inplace=True): out must be the whole buffer src views "
            "(contiguous, storage_offset 0, same base address and size)"
        )
    _verify_inplace_geometry(src, out)
    return True


rbln_custom_impl = torch.library.Library("rbln_custom_ops", "IMPL")  # noqa: TOR901
rbln_custom_impl.impl("paged_attn_prefill", paged_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_attn_decode", paged_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_prefill", paged_causal_attn_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("paged_causal_attn_decode", paged_causal_attn_decode_rbln, "PrivateUse1")
rbln_custom_impl.impl("flash_attention_naive_prefill", flash_attention_naive_prefill_rbln, "PrivateUse1")
rbln_custom_impl.impl("flash_attention_naive_decode", flash_attention_naive_decode_rbln, "PrivateUse1")

# The schema is declared in RBLNRegisterOps.cpp, in the namespace this backend owns; only
# the implementation lives here, because it drives the compile path.
torch_rbln_impl = torch.library.Library("torch_rbln", "IMPL")  # noqa: TOR901
torch_rbln_impl.impl("copy_strided_view", copy_strided_view_rbln, "PrivateUse1")
