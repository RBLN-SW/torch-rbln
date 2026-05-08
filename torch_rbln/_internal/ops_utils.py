import math
import os
import sys
import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import Optional, Union

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_rbln._internal.log_utils import rbln_log_cpu_fallback, rbln_log_warn


def _estimate_mm_shape(shape1, shape2):
    if len(shape1) != 2 or len(shape2) != 2:
        raise RuntimeError("mm input shape is invalid")

    if shape1[1] != shape2[0]:
        raise RuntimeError("mm input shapes are unmatched")

    result_shape = (shape1[0], shape2[1])
    return result_shape


def _needs_broadcast(tensor_args):
    if len(tensor_args) <= 1:
        return False
    first_shape = tensor_args[0].shape
    return any(t.shape != first_shape for t in tensor_args[1:])


def finalize_output_tensor(
    out_tensor: torch.Tensor, result: torch.Tensor, result_shape: tuple[int, ...], args: tuple, kwargs: dict
):
    """
    Ensure `out_tensor` has the correct shape, storage, and metadata to match
    `result` and `result_shape`, handling both resizing and data movement.
    """
    # 1) Resize if shape mismatches
    if out_tensor.shape != result_shape:
        # Warn if tensor had existing elements
        if out_tensor.numel() != 0:
            warnings.warn("An output with one or more elements")  # pytorch rule
        out_tensor.resize_(result_shape)

    # 2) Reconcile storage: copy or replace
    if result.data_ptr() != out_tensor.data_ptr():
        out_tensor.copy_(result)


def _make_contig(obj):
    changed = False

    if not isinstance(obj, torch.Tensor) or obj.numel() == 0:
        return obj, changed

    t = obj

    if not obj.is_contiguous():
        t = obj.contiguous()
        changed = True

    return t, changed


def _contains_nan_or_inf(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return False
        if x.numel() == 1 and torch.isreal(x):  # possibly scalar tensor and real number
            return math.isnan(x.item()) or math.isinf(x.item())
        return (torch.isnan(x) | torch.isinf(x)).any().item()
    elif isinstance(x, (float, int)):
        return math.isnan(x) or math.isinf(x) if isinstance(x, float) else False
    return False


def has_invalid_tensor(args):
    return any(_contains_nan_or_inf(x) for x in args)


def is_type_promotion_allowed(input_tensors, output_tensor):
    if not input_tensors:
        raise ValueError("Input tensors list cannot be empty")
    if output_tensor is None:
        raise ValueError("Output tensor cannot be None")
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(f"Output must be a torch.Tensor, but got {type(output_tensor)}")

    # Flatten input structure
    flat_inputs, _ = tree_flatten(input_tensors)

    # Filter only Tensors
    tensor_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
    if not tensor_inputs:
        raise TypeError("No tensor inputs found in input_tensors")

    # Promote types across all input tensors
    promoted_dtype = tensor_inputs[0].dtype
    for tensor in tensor_inputs[1:]:
        promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)

    # Compare with output tensor dtype
    final_promoted_dtype = torch.promote_types(promoted_dtype, output_tensor.dtype)
    return final_promoted_dtype == output_tensor.dtype


def is_type_promotion_allowed_dtype(input_dtypes, output_dtype):
    if not input_dtypes:
        raise ValueError("Input dtypes list cannot be empty")
    if output_dtype is None:
        raise ValueError("Output dtype cannot be None")

    final_promoted_dtype = torch.promote_types(input_dtypes, output_dtype)

    return final_promoted_dtype == output_dtype


def extract_tensors(obj):
    """
    Extract all torch.Tensor objects from an arbitrarily nested structure and return them as a flat list
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(extract_tensors(item))
        return tensors
    elif isinstance(obj, dict):
        tensors = []
        for v in obj.values():
            tensors.extend(extract_tensors(v))
        return tensors
    else:
        return []


def extract_device_id_from_inputs(*args, **kwargs):
    """
    Extract RBLN device_id from tensor inputs.

    This function searches through all positional and keyword arguments to find
    the first RBLN tensor and returns its device index.

    Args:
        *args: Positional arguments that may contain tensors.
        **kwargs: Keyword arguments that may contain tensors.

    Returns:
        Optional[int]: The device index of the first RBLN tensor found, or None if no RBLN tensor is found.
    """
    input_tensors = extract_tensors(args) + extract_tensors(kwargs)
    for tensor in input_tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "rbln":
            return tensor.device.index
    return None


def extract_warm_cache_key(*args, **kwargs):
    # Build a per-(device, input shape/dtype) cache key for compile_rbln_cached
    # so the rebel backend re-runs (and the C++ warm cache picks up a fresh
    # DynamoRuntime) when input profiles change. Without shape/dtype in the key,
    # compile_rbln_cached would reuse the first compiled callable across all
    # shapes and the warm-cache install path would only ever fire once.
    device_id = None
    profiles = []
    input_tensors = extract_tensors(args) + extract_tensors(kwargs)
    for tensor in input_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        if device_id is None and tensor.device.type == "rbln":
            device_id = tensor.device.index
        profiles.append((tuple(tensor.shape), str(tensor.dtype)))
    return (device_id, tuple(profiles))


def remove_empty_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return None if obj.numel() == 0 else obj
    elif isinstance(obj, (list, tuple)):
        filtered = [remove_empty_tensors(item) for item in obj]
        filtered = [x for x in filtered if x is not None]
        return type(obj)(filtered)
    elif isinstance(obj, dict):
        return {k: v for k, v in ((k, remove_empty_tensors(v)) for k, v in obj.items()) if v is not None}
    else:
        return obj


def to_cpu(x):
    if isinstance(x, torch.Tensor):  # for convert Tensor to cpu
        return x.cpu() if x.device != torch.device("cpu") else x
    elif isinstance(x, list):  # for convert list (recursive)
        return [to_cpu(item) for item in x]
    elif isinstance(x, tuple):  # for convert tuple (recursive)
        return tuple(to_cpu(item) for item in x)
    elif isinstance(x, dict):  # for convert dict (recursive)
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def handle_empty_reduction(input: torch.Tensor, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False):
    # Handle the case where dim is None or empty list []
    is_full_reduction = dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0)

    if is_full_reduction:
        out_shape = torch.Size([]) if not keepdim else torch.Size([1] * len(input.shape))
    else:
        if isinstance(dim, int):
            dim = [dim]
        out_shape = list(input.shape)
        # Sort dimensions in reverse order to avoid index shifting when deleting
        for d in sorted(dim, reverse=True):
            if keepdim:
                out_shape[d] = 1
            else:
                del out_shape[d]
        out_shape = torch.Size(out_shape)
    ret = torch.empty(out_shape, dtype=input.dtype, device="rbln")
    return ret, ret.shape


def handle_empty_mm(tensor_args):
    result_shape = _estimate_mm_shape(tensor_args[0].shape, tensor_args[1].shape)
    ret = torch.zeros(result_shape, dtype=tensor_args[0].dtype, device="rbln")
    return ret, ret.shape


def handle_empty_where(args):
    condition = args[0]
    # The result should have the same shape as the condition tensor
    # The dtype is usually taken from 'x', following PyTorch's behavior
    # No actual computation is needed since the condition is empty
    ret = torch.empty_like(condition, dtype=args[1].dtype, device="rbln")
    return ret, ret.shape


def handle_empty_binary(args):
    # Compute the broadcast result shape from all tensor args. With the
    # implicit-broadcast path (broadcast_args_general now validates only),
    # raw arg shapes may differ; the empty result must still have the
    # broadcast result shape so downstream copy_/resize_ behaves correctly.
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        raise RuntimeError("Can't find reference tensor for out")
    out_shape = torch.broadcast_shapes(*[t.shape for t in tensor_args])
    ref = tensor_args[0]
    ret = torch.empty(out_shape, dtype=ref.dtype, device="rbln")
    return ret, ret.shape


def _has_zero_dim_broadcast(tensor_args, out_shape):
    """Detect a 0-dim (scalar) tensor that needs broadcast to a non-empty shape.

    Pattern observed in fp16 backward of reductions: ``y = cos(x).sum()``
    sets ``grad_y`` to a 0-dim fp16 tensor; the autograd-emitted backward
    graph then performs ``mul(grad_y, -sin(x))`` against a vector. Without
    pre-broadcast the rebel-compiler receives an IR that lowers the
    0-dim → N-dim broadcast as ``expand_dims(scalar, 0)`` followed by
    ``repeat(., N, axis=0)`` — and aborts inside ``build_internal`` for
    fp16 scalar+vector multiply (verified 2026-05-07: the same call passes
    when one of the inputs is forced to vector shape via pre-broadcast).

    Pre-broadcast on the host turns the call into an N-dim × N-dim mul
    that rebel-compiler handles cleanly.
    """
    if not out_shape:
        return False
    for t in tensor_args:
        if t.dim() == 0:
            return True
    return False


def _has_last_dim_size_one_broadcast(tensor_args, out_shape):
    """Detect tensors that need a stride-0 expand on the LAST dim.

    Known UNSUPPORTED rebel-backend implicit-broadcast pattern (2026-04-30):
    when one operand has ``shape[-1] == 1`` and the broadcast result has
    ``shape[-1] > 1``, rebel-compiler's graph optimizer raises
    ``Graph Optimization: [UNEXPECTED_GRAPH]`` instead of compiling the
    stride-0 expand on the last axis.

    Confirmed-failing patterns (covered by ``TestBinaryOpsBroadcast``)::

        (3, 4)        × (3, 1)        → (3, 4)        # softmax_backward shape0
        (2, 3, 4)     × (2, 3, 1)     → (2, 3, 4)     # softmax_backward shape2
        (4, 8, 16)    × (4, 8, 1)     → (4, 8, 16)
        (1, 32, 4)    × (1, 32, 1)    → (1, 32, 4)

    Confirmed-WORKING patterns (validate-only is fine)::

        (B, S, H) × (H,)              # RMSNorm — last dims already match
        (10, 20)  × (1, 20)           # leading-dim broadcast, last dims match
        (5, 10, 15) × (5, 1, 15)      # middle-dim broadcast

    Origin of pattern: ``torch.tensor.sum(dim=last_dim, keepdim=True)`` and
    similar reductions yield ``shape[-1] == 1``; multiplying that back into
    the unreduced tensor (the standard ``softmax_backward`` and
    ``layernorm_backward`` formulas) hits this case.

    If a future workload trips a different rebel-broadcast bug, broaden this
    helper rather than re-enabling explicit broadcast wholesale — the RMSNorm
    fast path is too valuable.
    """
    out_last = out_shape[-1] if out_shape else 1
    if out_last == 1:
        return False
    for t in tensor_args:
        if t.shape and t.shape[-1] == 1:
            return True
    return False


def broadcast_args_general(tensor_args, args):
    """Validate broadcast compatibility; pre-broadcast only when rebel needs it.

    Default: validate-only (no allocation). Rebel-backend lowers most
    implicit broadcasts inside the compiled graph (e.g. ``(B,S,H) * (H,)``
    for RMSNorm). Pre-broadcasting them via ``torch.broadcast_tensors``
    materializes stride-0 expand views through D2H + host broadcast + H2D
    (~600 ms / prefill step on LLaMA-1B), which we want to avoid.

    Exception (force pre-broadcast): the **last-dim-size-one** pattern
    documented in ``_has_last_dim_size_one_broadcast``. Rebel-compiler
    raises ``UNEXPECTED_GRAPH`` for those, so we materialize on the host
    side and feed compile same-shape inputs. The pre-broadcasted result is
    ``.contiguous()``-ed (else ``prepare_args_view_aware`` would re-detect
    the expand recipe and emit ``aten::expand`` into the FX graph, dropping
    back into the implicit-broadcast path that fails).

    Cache coherence: after pre-broadcast the runtime is compiled for
    POST-broadcast shapes while the C++ shim's pending warm-cache key was
    built from PRE-broadcast shapes. The shim's ``try_warmcache_hit``
    re-applies the same broadcast on warm-hit (see
    ``DispatchShim.cpp::needs_last_dim_one_broadcast``) so the cached
    runtime continues to receive same-shape buffers — install proceeds
    normally and warm-hits stay correct.
    """
    if not _needs_broadcast(tensor_args):
        return args
    try:
        out_shape = torch.broadcast_shapes(*[t.shape for t in tensor_args])
    except RuntimeError as e:
        tensor_shapes = [tuple(t.shape) for t in tensor_args]
        raise RuntimeError(f"Broadcasting failed for tensor shapes={tensor_shapes}") from e

    if not _has_last_dim_size_one_broadcast(tensor_args, out_shape) \
            and not _has_zero_dim_broadcast(tensor_args, out_shape):
        return args

    try:
        broadcasted = torch.broadcast_tensors(*tensor_args)
    except RuntimeError as e:
        tensor_shapes = [tuple(t.shape) for t in tensor_args]
        raise RuntimeError(f"Broadcasting failed for tensor shapes={tensor_shapes}") from e
    # Materialize stride-0 expand views into real contig buffers — otherwise
    # ``prepare_args_view_aware`` would re-detect the expand recipe and emit
    # ``aten::expand`` into the FX graph, dropping us right back into the
    # implicit-broadcast path that rebel can't compile for last-dim-1.
    new_args = []
    tensor_idx = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            t = broadcasted[tensor_idx]
            if not t.is_contiguous():
                t = t.contiguous()
            new_args.append(t)
            tensor_idx += 1
        else:
            new_args.append(arg)
    return tuple(new_args)


def handle_empty_addmm(tensor_args, beta):
    result_shape = _estimate_mm_shape(tensor_args[1].shape, tensor_args[2].shape)
    ret = torch.zeros(result_shape, dtype=tensor_args[0].dtype, device="rbln")
    ret.add_(tensor_args[0], alpha=beta)
    return ret, ret.shape


def addmm_broadcast_args(tensor_args, args):
    try:
        result_shape = _estimate_mm_shape(tensor_args[1].shape, tensor_args[2].shape)
        return (tensor_args[0].expand(result_shape), tensor_args[1], tensor_args[2])
    except RuntimeError as e:
        raise RuntimeError(f"Broadcasting failed for {tensor_args[0]}") from e


def handle_empty_linear(tensor_args):
    """Handle linear operation when input tensor is empty (numel == 0).

    Linear operation: output = input @ weight.T + bias

    For linear(input, weight, bias):
    - input shape: [..., in_features]
    - weight shape: [out_features, in_features]
    - output shape: [..., out_features]

    When input is empty (e.g., [0, 3]), output is also empty (e.g., [0, 4])
    regardless of weight/bias values.

    Args:
        tensor_args: List of tensors [input, weight, bias (optional)]

    Returns:
        tuple: (empty output tensor, output shape)
    """
    input_tensor = tensor_args[0]
    weight_tensor = tensor_args[1]

    # Output shape: input's batch dims + weight's out_features
    # input: [..., in_features], weight: [out_features, in_features]
    # output: [..., out_features]
    out_features = weight_tensor.shape[0]
    output_shape = list(input_tensor.shape[:-1]) + [out_features]

    ret = torch.empty(output_shape, dtype=input_tensor.dtype, device="rbln")
    return ret, ret.shape


def handle_empty_tensor(tensor_args):
    for a in tensor_args:
        ret = torch.empty(a.shape, dtype=a.dtype, device="rbln")
        return ret, ret.shape
    raise RuntimeError("Can't find reference tensor for out")


def prepare_args_for_contiguous(args, kwargs_filtered):
    flat_args, args_spec = tree_flatten((args, kwargs_filtered))
    contig_args, changed_any = [], False
    for a in flat_args:
        t, changed = _make_contig(a)
        contig_args.append(t)
        changed_any |= changed
    return tree_unflatten(contig_args, args_spec), changed_any


# ============================================================================
# View-on-device helpers (2026-04-30+; extended 2026-05-01)
#
# When a non-contig view reaches an op handler, we historically called
# ``.contiguous()`` which goes through host (D2H + host transform + H2D, see
# RBLNCopy.cpp). For most view types the data transformation can instead be
# expressed as an explicit ``aten::*`` node in the FX graph; rebel-backend
# then lowers it on device alongside the compute kernel.
#
# Detection works in two layers:
#
#   1. Fast single-step path: read ``t.shape``, ``t.stride()``,
#      ``t.storage_offset()`` and recognize a pure permute pattern (most
#      common LLaMA hot path). No ``_base`` walk required.
#
#   2. Composite chain path: walk ``t._base`` until we hit a contig+offset=0
#      ancestor. For each (parent, child) link, classify the single-step
#      view op (permute / expand / narrow / select / reshape / squeeze /
#      unsqueeze) by comparing shapes, strides, and offsets. Concatenate
#      into a recipe list applied in order to the root base.
#
# An op handler's wrapper OpModule applies the recipe in its ``forward``,
# so each step ends up as an explicit aten node in the FX graph and survives
# torch.compile's tracing into rebel-backend's MLIR pipeline.
#
# Unrecognized chains (no contig root, or step we can't classify) fall back
# to the legacy ``.contiguous()`` path. A ``rbln_log_warn`` and a counter
# (``_view_fallback_count``) are emitted so we can detect when this happens
# and either teach the classifier about a new view type, or accept the
# host materialization.
# ============================================================================

# Telemetry: how often the view-detection fell back to .contiguous() because
# we couldn't classify the chain. Should stay 0 in real workloads — non-zero
# means we have unhandled view patterns to investigate.
_view_fallback_count: int = 0
_view_fallback_warn_once: set = set()


def _view_fallback_count_get() -> int:
    return _view_fallback_count


def _view_fallback_count_reset() -> None:
    global _view_fallback_count
    _view_fallback_count = 0
    _view_fallback_warn_once.clear()


def _contig_strides_for(shape):
    """Compute contiguous strides for a given shape (C-order)."""
    n = len(shape)
    strides = [0] * n
    s = 1
    for k in range(n - 1, -1, -1):
        strides[k] = s
        s *= shape[k]
    return tuple(strides)


def _detect_permute(t: torch.Tensor):
    """Fast-path detector for a *single-step* pure permute view.

    Returns ``(base_shape, perm)`` such that ``base.permute(perm)`` reproduces
    ``t``, or ``None`` if not a pure permute. Pure permute = no expand
    (no stride 0), no slice (offset 0), all strides > 0, and strides match
    a permutation of the contig strides for some base shape. The contig
    "base" view of ``t``'s storage is constructible via ``as_strided``.

    Used by ``_detect_view_recipe`` as the fast path before the chain walk.
    """
    if t.numel() == 0:
        return None
    if t.storage_offset() != 0:
        return None
    if t.is_contiguous():
        return None
    n = t.dim()
    if n == 0:
        return None
    strides = t.stride()
    if any(s == 0 for s in strides):
        return None
    sizes = list(t.size())
    sorted_dims = sorted(range(n), key=lambda i: -strides[i])
    base_shape = tuple(sizes[d] for d in sorted_dims)
    contig_stride = _contig_strides_for(base_shape)
    for k in range(n):
        if strides[sorted_dims[k]] != contig_stride[k]:
            return None
    perm = [0] * n
    for k, d in enumerate(sorted_dims):
        perm[d] = k
    return base_shape, tuple(perm)


def _construct_permute_base(t: torch.Tensor, base_shape):
    """Return a contig view of ``t``'s storage with shape ``base_shape``."""
    return torch.as_strided(t, base_shape, _contig_strides_for(base_shape), 0)


def _classify_single_step_view(parent: torch.Tensor, child: torch.Tensor):
    """Classify the single view op that transforms ``parent`` into ``child``
    (which must share storage). Returns a recipe step tuple, or ``None`` if
    the op pattern is unrecognized.

    Recipe step encoding (all immutable / hashable):
      ('permute', perm_tuple)
      ('expand', target_shape)
      ('narrow', dim, start, length)
      ('select', dim, index)
      ('reshape', new_shape)
      ('squeeze', dim)
      ('unsqueeze', dim)

    Cases are mutually exclusive in the order checked. If multiple classifiers
    could match (e.g., 0-dim edge cases), the first wins; the wrapper applies
    them deterministically.
    """
    p_shape = list(parent.size())
    c_shape = list(child.size())
    p_stride = list(parent.stride())
    c_stride = list(child.stride())
    p_off = parent.storage_offset()
    c_off = child.storage_offset()
    p_ndim, c_ndim = len(p_shape), len(c_shape)

    # ---- 1. Same ndim cases ----
    if c_ndim == p_ndim and c_ndim > 0:
        # Permute (same numel, same offset, no stride 0, strides are perm
        # of parent's strides). Detect by finding a position-mapping.
        if (c_off == p_off
                and parent.numel() == child.numel()
                and not any(s == 0 for s in c_stride)
                and sorted(c_shape) == sorted(p_shape)
                and sorted(c_stride) == sorted(p_stride)):
            perm = []
            used = [False] * p_ndim
            ok = True
            for i in range(c_ndim):
                found = False
                for j in range(p_ndim):
                    if used[j]:
                        continue
                    if p_shape[j] == c_shape[i] and p_stride[j] == c_stride[i]:
                        perm.append(j)
                        used[j] = True
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and len(perm) == c_ndim:
                # Reject the identity perm (no actual transformation): caller
                # already handled "is_contiguous + offset 0" before calling.
                if tuple(perm) != tuple(range(c_ndim)):
                    return ("permute", tuple(perm))

        # Expand: same offset, same ndim, child has stride 0 in some dims and
        # the non-zero strides match the parent's at those positions. The
        # parent's size-1 dims become >1 in the child along the same axes.
        if c_off == p_off and any(cs == 0 for cs in c_stride):
            ok = True
            for i in range(c_ndim):
                if c_stride[i] == 0:
                    if p_shape[i] != 1:
                        ok = False
                        break
                else:
                    if p_shape[i] != c_shape[i] or p_stride[i] != c_stride[i]:
                        ok = False
                        break
            if ok:
                # Reject the recipe when ALL parent dims have size 1 (parent is a
                # rank-N broadcast of a single scalar): autograd's bprop emits
                # ``scalar.expand(target)`` for ``y.sum().backward()`` and similar
                # reductions; if we replay this expand inside the traced graph,
                # rebel-compiler lowers it as ``unsqueeze(scalar) + repeat`` and
                # then aborts in ``build_internal`` for fp16 scalar+vector
                # multiply (verified 2026-05-07 via the cos/neg/mul trio that
                # ``test_variant_consistency_eager_*_rbln_float16`` exercises).
                # Falling out of the recipe path makes the caller materialize a
                # contig vector via ``.contiguous()``, so torch.compile sees
                # vector × vector and the abort goes away.
                if all(s == 1 for s in p_shape):
                    pass
                else:
                    return ("expand", tuple(c_shape))

        # Narrow: exactly one dim shrunk, offset moved by start*stride[dim].
        diff_dims = [i for i in range(p_ndim) if p_shape[i] != c_shape[i]]
        if (len(diff_dims) == 1
                and p_stride == c_stride):
            d = diff_dims[0]
            stride_d = p_stride[d]
            if stride_d != 0 and c_shape[d] <= p_shape[d]:
                offset_delta = c_off - p_off
                if offset_delta % stride_d == 0:
                    start = offset_delta // stride_d
                    length = c_shape[d]
                    if start >= 0 and start + length <= p_shape[d]:
                        return ("narrow", d, start, length)

        # Reshape (within same ndim): same numel, offset 0, both contig.
        if (c_off == 0 and p_off == 0
                and parent.numel() == child.numel()
                and parent.is_contiguous() and child.is_contiguous()
                and c_shape != p_shape):
            return ("reshape", tuple(c_shape))

    # ---- 2. ndim shrunk by 1: squeeze or select ----
    if c_ndim == p_ndim - 1:
        # Squeeze: parent has a size-1 dim removed; offset same.
        if c_off == p_off:
            for sq in range(p_ndim):
                if p_shape[sq] != 1:
                    continue
                expected = p_shape[:sq] + p_shape[sq + 1:]
                if expected == c_shape:
                    # Strides also need to align — for size-1 dim, dropping
                    # it preserves the rest.
                    expected_stride = p_stride[:sq] + p_stride[sq + 1:]
                    if expected_stride == c_stride:
                        return ("squeeze", sq)
        # Select: indexing one position along a dim. offset increased by
        # index * stride_dim. The selected dim is removed.
        for sel in range(p_ndim):
            stride_d = p_stride[sel]
            if stride_d == 0:
                continue
            offset_delta = c_off - p_off
            if offset_delta % stride_d != 0:
                continue
            index = offset_delta // stride_d
            if not (0 <= index < p_shape[sel]):
                continue
            expected = p_shape[:sel] + p_shape[sel + 1:]
            if expected != c_shape:
                continue
            expected_stride = p_stride[:sel] + p_stride[sel + 1:]
            if expected_stride != c_stride:
                continue
            return ("select", sel, index)

    # ---- 3. ndim grew by 1: unsqueeze ----
    if c_ndim == p_ndim + 1:
        if c_off == p_off:
            for un in range(c_ndim):
                if c_shape[un] != 1:
                    continue
                expected = c_shape[:un] + c_shape[un + 1:]
                if expected == p_shape:
                    expected_stride = c_stride[:un] + c_stride[un + 1:]
                    if expected_stride == p_stride:
                        return ("unsqueeze", un)

    # ---- 4. Reshape with ndim change ----
    if (c_off == 0 and p_off == 0
            and parent.numel() == child.numel()
            and parent.is_contiguous() and child.is_contiguous()
            and c_shape != p_shape):
        return ("reshape", tuple(c_shape))

    return None


def _detect_permute_narrow_composite(base: torch.Tensor, t: torch.Tensor):
    """Try to express ``t`` as ``base.permute(perm).narrow(...)`` (one or more
    narrow steps). Returns a recipe tuple or None.

    PyTorch's ``t._base`` always points at the root storage owner, not the
    immediate parent — so for chained views like ``base.permute(...).narrow(...)``
    we don't actually walk an intermediate tensor. We have to reverse-engineer
    the recipe from ``(base.shape, base.stride, t.shape, t.stride, t.offset)``.

    Algorithm:
      1. Strides of t are preserved through narrow (only sizes change), so
         t.stride should be a permutation of base's contig strides.
      2. Recover ``perm`` from stride correspondence.
      3. Compute hypothetical post-permute shape; dims where t.shape differs
         (smaller) were narrowed. Solve for ``start`` per narrow dim from
         the storage_offset (greedy, descending stride).
    """
    p_ndim = base.dim()
    c_ndim = t.dim()
    if c_ndim != p_ndim or c_ndim == 0:
        return None
    t_stride = list(t.stride())
    if any(s == 0 for s in t_stride):
        return None  # not a permute+narrow (has expand)
    p_stride = list(base.stride())
    # Match t's stride positions to base's stride positions.
    perm = []
    used = [False] * p_ndim
    for i in range(c_ndim):
        found = False
        for j in range(p_ndim):
            if used[j]:
                continue
            if p_stride[j] == t_stride[i]:
                perm.append(j)
                used[j] = True
                found = True
                break
        if not found:
            return None
    if len(perm) != c_ndim:
        return None
    # Hypothetical shape after permute (before any narrow).
    perm_shape = [base.size(perm[i]) for i in range(c_ndim)]
    narrows = []
    for i in range(c_ndim):
        if t.size(i) == perm_shape[i]:
            continue
        if t.size(i) > perm_shape[i]:
            return None  # can't grow via narrow
        narrows.append(i)
    if not narrows:
        # No narrow → pure permute. Caller already handled identity; here we
        # emit the permute recipe (might've been missed by the fast path due
        # to size-1 dim ambiguity).
        if perm == list(range(c_ndim)):
            return None
        return (("permute", tuple(perm)),)
    # Solve starts: offset = sum(start_d * t.stride[d]). Greedy from largest
    # stride (single-narrow case is exact; multi-narrow common case has
    # unique solution under "0 <= start_d <= max_start_d").
    starts = {}
    sorted_narrows = sorted(narrows, key=lambda d: -t_stride[d])
    rem = t.storage_offset()
    for d in sorted_narrows:
        stride_d = t_stride[d]
        if stride_d == 0:
            return None
        max_start = perm_shape[d] - t.size(d)
        if max_start < 0:
            return None
        cand = rem // stride_d
        if cand < 0 or cand > max_start:
            return None
        starts[d] = cand
        rem -= cand * stride_d
    if rem != 0:
        return None
    # Emit narrow BEFORE permute. Algebraically equivalent — narrow on
    # post-permute dim ``d`` is the same as narrow on base dim ``perm[d]``
    # — but rebel-backend has trouble lowering ``narrow`` applied to a
    # permuted operand on non-leading dims with non-trivial offset
    # (observed 2026-04-30: ``base.narrow(0,2,4).permute(2,0,1)`` produced
    # ``inf`` outputs when emitted as ``permute → narrow(1,2,4)``).
    # Narrow-first keeps the tensor contig until the permute, so the
    # resulting graph has a single ``narrow`` on the original base layout
    # plus a final ``permute`` — the form rebel reliably handles.
    recipe = []
    for d in narrows:
        base_dim = perm[d]
        recipe.append(("narrow", base_dim, starts[d], t.size(d)))
    if perm != list(range(c_ndim)):
        recipe.append(("permute", tuple(perm)))
    return tuple(recipe)


def _detect_expand_permute_composite(base: torch.Tensor, t: torch.Tensor):
    """Try to express ``t`` as ``base.expand(target).permute(perm)`` (or just
    ``expand`` if the perm is identity). Returns a recipe tuple or None.

    Algorithm:
      Stride 0 in t means an expanded dim. Mapping non-zero-stride dims of t
      to base's dims by stride value gives perm; the post-expand shape is
      derived from t.shape by inverting the perm.
    """
    p_ndim = base.dim()
    c_ndim = t.dim()
    if c_ndim < p_ndim:
        return None
    t_stride = list(t.stride())
    if not any(s == 0 for s in t_stride):
        return None  # no expand
    if t.storage_offset() != base.storage_offset():
        return None
    # For each non-zero-stride dim in t, find the matching base dim.
    p_stride = list(base.stride())
    p_size = list(base.size())
    p_used = [False] * p_ndim
    # We'll express t = base.expand(target_shape).permute(perm) where
    #   target_shape has p_ndim + (extra) dims:
    #   - extra = c_ndim - p_ndim leading dims of size t.shape[some_position]
    #     (and stride 0)
    #   - rest match base dim by dim, with size-1 dims possibly broadened
    # Then permute to get t's order.
    #
    # To avoid full search, we use the following: for each non-zero-stride
    # dim i of t, we find j in base such that base.stride(j) == t.stride(i)
    # AND base.size(j) >= 1. The constraint is that the matched dim contributes
    # its original size to the post-expand shape (only size-1 base dims can
    # be expanded).
    perm_after_expand = [-1] * c_ndim
    for i in range(c_ndim):
        st = t_stride[i]
        if st == 0:
            continue  # expanded dim, no base correspondence
        found = -1
        for j in range(p_ndim):
            if p_used[j]:
                continue
            if p_stride[j] == st:
                # Stride match. The size constraint: t.size[i] must equal
                # base.size[j] (non-1 dim is preserved).
                if p_size[j] == t.size(i):
                    found = j
                    break
        if found == -1:
            return None
        perm_after_expand[i] = found
        p_used[found] = True
    # Remaining base dims must be size-1 (they map to expanded zero-stride
    # positions in t).
    expand_zero_dims = [i for i, s in enumerate(t_stride) if s == 0]
    expanded_p_dims = [j for j in range(p_ndim) if not p_used[j]]
    # The number of zero-stride dims in t includes (a) base size-1 dims that
    # were expanded and (b) extra leading dims that expand introduced.
    # Each zero-stride dim corresponds to either a base size-1 dim or a
    # newly-added dim. We need #expand_zero_dims >= #expanded_p_dims.
    if any(p_size[j] != 1 for j in expanded_p_dims):
        return None  # expand can only widen size-1 dims
    if len(expand_zero_dims) < len(expanded_p_dims):
        return None
    # Construct target_shape (after expand, before permute). The order of dims
    # is: leading "new" dims (one per zero-stride dim with no base correspondence),
    # followed by the base dims in original order (with size-1 dims potentially
    # widened to t's matched zero-stride size).
    n_extra = len(expand_zero_dims) - len(expanded_p_dims)
    if c_ndim - n_extra != p_ndim:
        return None
    # Determine perm: t came from permuting (target_shape) order. We need to
    # find perm such that t.shape[i] == target_shape[perm[i]] AND t.stride[i]
    # == target_stride[perm[i]] where target_stride matches contig strides
    # for target_shape with size-1 dims having stride 0.
    #
    # Simpler: just attempt the easier sub-case where expand happens (no perm).
    # That covers the very common "weight.expand(...)" pattern.
    # For more complex expand+permute composites, return None and fall back.
    if c_ndim == p_ndim:
        # No new leading dims, no permute (perm_after_expand should equal range)
        if perm_after_expand == list(range(c_ndim)):
            # Pure expand-in-place.
            return (("expand", tuple(t.size())),)
        # Else: permute + expand combo not handled here.
        return None
    # c_ndim > p_ndim: extra leading dims via expand. If all extra are at the
    # front and the rest is identity perm, we can express as expand alone.
    # i.e., perm_after_expand[:n_extra] are -1 (zero stride) and the rest is
    # range(n_extra, c_ndim) → matches base dims [0..p_ndim).
    leading_zero = all(perm_after_expand[i] == -1 for i in range(n_extra))
    rest_identity = all(perm_after_expand[n_extra + i] == i for i in range(p_ndim))
    if leading_zero and rest_identity:
        return (("expand", tuple(t.size())),)
    return None


def _detect_permute_unsqueeze_composite(base: torch.Tensor, t: torch.Tensor):
    """Try to express ``t`` as ``base.permute(perm).unsqueeze(...)``.

    Common in attention reshape (e.g., ``x.view(...).permute(...).unsqueeze(0)``)
    and rotary embedding output patterns. Detected by stripping size-1 dims
    from t and checking whether the remaining shape is a pure permute of
    base (using the same stride-pattern check as ``_detect_permute``).

    Returns recipe tuple ``[(permute, perm), (unsqueeze, dim), ...]`` or
    ``None``. Multiple unsqueezes are emitted in ascending dim order so
    each step's index is correct relative to the prior shape.
    """
    if t.dim() <= base.dim():
        return None
    if t.storage_offset() != base.storage_offset():
        return None
    t_stride = list(t.stride())
    if any(s == 0 for s in t_stride):
        return None  # expand path, not permute+unsqueeze
    # Identify size-1 dims in t (candidates for unsqueeze positions).
    one_dims = [i for i, sz in enumerate(t.size()) if sz == 1]
    if len(one_dims) != t.dim() - base.dim():
        return None  # extra size-1 count must match the unsqueeze count
    # Build the "stripped" view: remove size-1 dims from t.
    stripped_size = [t.size(i) for i in range(t.dim()) if t.size(i) != 1
                     or i not in one_dims]
    # The above is fragile; simpler: drop indices in `one_dims` (the dims we
    # plan to mark as unsqueezed), keeping the rest.
    drop = set(one_dims[: len(one_dims)])  # drop ALL size-1 candidates
    stripped_size = [t.size(i) for i in range(t.dim()) if i not in drop]
    stripped_stride = [t_stride[i] for i in range(t.dim()) if i not in drop]
    if len(stripped_size) != base.dim():
        return None
    n = len(stripped_size)
    if n == 0:
        return None
    # Run the same pure-permute detector against (stripped_size, stripped_stride)
    # vs base's contig strides.
    sorted_dims = sorted(range(n), key=lambda i: -stripped_stride[i])
    candidate_base_shape = tuple(stripped_size[d] for d in sorted_dims)
    if candidate_base_shape != tuple(base.size()):
        return None
    contig_stride = _contig_strides_for(candidate_base_shape)
    for k in range(n):
        if stripped_stride[sorted_dims[k]] != contig_stride[k]:
            return None
    perm = [0] * n
    for k, d in enumerate(sorted_dims):
        perm[d] = k
    # Build recipe: permute first (acting on base), then unsqueeze each one_dim
    # in ascending order so subsequent positions are valid relative to the
    # then-current shape.
    recipe = []
    if perm != list(range(n)):
        recipe.append(("permute", tuple(perm)))
    for un_dim in sorted(drop):
        recipe.append(("unsqueeze", un_dim))
    if not recipe:
        return None
    return tuple(recipe)


def _construct_synthetic_base(t: torch.Tensor):
    """Construct a contig+offset=0 view of ``t``'s storage, with shape derived
    purely from ``t``'s stride pattern. Used when ``t._base`` is itself a
    non-contig view (PyTorch's ``_base`` points to the storage owner, which
    may be another view, especially for autograd-disabled tensors and
    tensors built via composite reshape chains).

    Algorithm:
      - Drop dims with stride 0 (they're expand artifacts) and size-1 dims
        (they don't contribute to extent — appear as ``unsqueeze`` later).
      - Sort the remaining (stride, size) pairs by descending stride.
      - Verify the strides match contig strides for the resulting shape.
        If not, the storage isn't a simple contig+permute layout — return None.
      - Build via ``torch.as_strided`` (metadata-only, shares storage).
    """
    sizes = list(t.size())
    strides = list(t.stride())
    real = [(strides[i], sizes[i]) for i in range(t.dim()) if strides[i] > 0 and sizes[i] > 1]
    if not real:
        return None
    real.sort(key=lambda x: -x[0])
    canonical_strides = [st for st, _ in real]
    canonical_sizes = [sz for _, sz in real]
    expected = _contig_strides_for(canonical_sizes)
    if tuple(canonical_strides) != tuple(expected):
        return None
    return torch.as_strided(t, canonical_sizes, expected, 0)


# ============================================================================
# Generic view recipe detector (2026-04-30+, replaces 5-layer hard-coded path)
#
# Algorithm:
#   1. BFS over single-step view primitives (permute / narrow / expand /
#      select / squeeze / unsqueeze).
#   2. Each candidate step's effect is simulated on (shape, stride, offset)
#      tuples — no actual tensor allocation, just metadata math.
#   3. A recipe is accepted only when its full simulation produces metadata
#      identical to the target tensor's (shape, stride, offset). Wrong
#      recipes can never silently emit because verification is bit-exact.
#   4. Blacklisted patterns (rebel-backend bugs we've discovered) are
#      filtered out so the search picks an algebraically-equivalent
#      alternative when one exists.
#   5. Synthetic-base fallback (``_construct_synthetic_base``) handles
#      tensors whose ``_base`` is not contig+offset=0.
#
# Performance characteristics:
#   - First match wins (BFS by recipe length); pure permute/narrow/etc.
#     terminate after 1 step → ~5 µs.
#   - Each step's simulation is O(ndim) integer ops, no allocation.
#   - max_steps=4 caps the search depth (deeper chains are vanishingly
#     rare; if they happen, we fall back to synthetic base).
# ============================================================================


def _simulate_recipe(shape, stride, offset, recipe):
    """Apply each step in ``recipe`` to (shape, stride, offset) metadata
    and return the new ``(shape, stride, offset)`` tuple.

    Returns ``None`` on invalid step (e.g. ``expand`` on non-size-1 dim,
    ``narrow`` past dim end). Never raises — used in BFS where invalid
    candidates are silently skipped.
    """
    s = list(shape)
    st = list(stride)
    o = offset
    for step in recipe:
        op = step[0]
        if op == "permute":
            perm = step[1]
            if len(perm) != len(s) or sorted(perm) != list(range(len(s))):
                return None
            s = [s[i] for i in perm]
            st = [st[i] for i in perm]
        elif op == "narrow":
            if len(step) != 4:
                return None
            d, start, length = step[1], step[2], step[3]
            if d < 0 or d >= len(s) or start < 0 or length < 0 or start + length > s[d]:
                return None
            o = o + start * st[d]
            s[d] = length
        elif op == "expand":
            if len(step) != 3:
                return None
            d, new_size = step[1], step[2]
            if d < 0 or d >= len(s) or s[d] != 1 or new_size < 1:
                return None
            s[d] = new_size
            st[d] = 0
        elif op == "select":
            if len(step) != 3:
                return None
            d, idx = step[1], step[2]
            if d < 0 or d >= len(s) or idx < 0 or idx >= s[d]:
                return None
            o = o + idx * st[d]
            del s[d]
            del st[d]
        elif op == "squeeze":
            d = step[1]
            if d < 0 or d >= len(s) or s[d] != 1:
                return None
            del s[d]
            del st[d]
        elif op == "unsqueeze":
            d = step[1]
            if d < 0 or d > len(s):
                return None
            s.insert(d, 1)
            # PyTorch convention: ``unsqueeze(d)`` inserts a size-1 dim
            # with stride ``stride[d] * size[d]`` (extends the contig
            # convention from the dim now at position d). At the end,
            # use stride 1.
            if d < len(st):
                ref_stride = st[d] * s[d + 1]
            else:
                ref_stride = 1
            st.insert(d, ref_stride)
        else:
            return None
    return tuple(s), tuple(st), o


def _is_known_bad_pattern(recipe):
    """Reject recipes that match patterns rebel-backend cannot lower
    correctly. The BFS will then prefer an algebraically-equivalent
    alternative (e.g. ``narrow → permute`` instead of ``permute → narrow``).

    Add new entries here as new rebel bugs surface.
    """
    # 1. ``permute → narrow(non-leading dim)`` produces ``inf`` outputs from
    #    rebel-compiler. The equivalent ``narrow → permute`` works.
    #    (Observed 2026-04-30 with ``base.narrow(0,2,4).permute(2,0,1)``.)
    for i, step in enumerate(recipe):
        if step[0] == "narrow" and i > 0:
            prev = recipe[i - 1]
            if prev[0] == "permute" and step[1] > 0:
                return True

    # 2. Trailing ``narrow`` on a non-leading dim with non-zero start (or
    #    equivalently: any narrow on dim>0 where start>0) followed by an
    #    elementwise op (silu / sigmoid / etc.) aborts rebel-compiler at
    #    ``build_module.build_internal``. The same op on a contig input
    #    works, so the safe path is to materialize via ``.contiguous()``
    #    (return None from the detector — caller emits fallback warning).
    #    (Observed 2026-04-30 with ``silu(base.narrow(1, 4, 8))``.)
    for step in recipe:
        if step[0] == "narrow" and step[1] > 0 and step[2] > 0:
            return True

    # 3. ``expand`` followed by anything other than another ``expand`` —
    #    rebel-compiler rejects the resulting graph (e.g. ``compile error``
    #    for ``mul(expand+permute(t), other)``). Pure expand alone, or a
    #    chain of expand-on-different-dims (canonical broadcast lowering),
    #    is fine. Only ``expand → permute / narrow / select`` trips the bug.
    #    (Observed 2026-04-30 with ``base.expand(3,4,8).permute(2,0,1)``.)
    for i, step in enumerate(recipe):
        if step[0] == "expand" and i + 1 < len(recipe):
            if recipe[i + 1][0] != "expand":
                return True

    # 4. ``unsqueeze`` followed by anything other than another ``unsqueeze``
    #    or ``expand`` — rebel-compiler runtime fails (vmemory verify
    #    error) when unsqueeze precedes permute / narrow / etc. Legacy
    #    detector emits ``permute → unsqueeze`` (unsqueeze last) which
    #    works. ``unsqueeze → expand`` IS allowed: it's the canonical
    #    broadcast lowering ``(H,).expand(B,S,H)`` → which rebel handles
    #    fine (test_expand_view_via_binary_op exercises this).
    #    (Observed 2026-04-30 with ``base.permute(2,0,1).unsqueeze(0)``.)
    for i, step in enumerate(recipe):
        if step[0] == "unsqueeze" and i + 1 < len(recipe):
            next_op = recipe[i + 1][0]
            if next_op not in ("unsqueeze", "expand"):
                return True
    return False


def _gen_step_candidates(cur, target):
    """Yield single-step candidates that might bring ``cur`` toward ``target``.

    cur, target: (shape, stride, offset) tuples.

    Each yielded step is a tuple suitable for ``_simulate_recipe``. Order
    of yield favors patterns most likely to terminate early (permute first,
    then dim-changing ops).
    """
    cur_s, cur_st, cur_o = cur
    tgt_s, tgt_st, tgt_o = target

    # Permute. Two cases:
    #   (a) Same dim count: yield the unique permute (if any) whose
    #       resulting strides match target's.
    #   (b) Different dim count (cur < tgt): a later ``unsqueeze`` will
    #       add the missing dim(s); we don't know which permutation
    #       sets up the right post-unsqueeze shape, so enumerate all
    #       permutations. With ndim ≤ 5 the candidate count stays small
    #       (max 120). The simulation + final equality check filters
    #       wrong picks.
    if len(cur_s) == len(tgt_s):
        used = [False] * len(cur_st)
        perm = []
        for ts, tsz in zip(tgt_st, tgt_s):
            found = False
            for i, (cs, csz) in enumerate(zip(cur_st, cur_s)):
                if used[i]:
                    continue
                if cs == ts and (tsz != 1 or csz == 1):
                    perm.append(i)
                    used[i] = True
                    found = True
                    break
            if not found:
                perm = None
                break
        if perm is not None and perm != list(range(len(cur_st))):
            yield ("permute", tuple(perm))
    elif len(cur_s) < len(tgt_s) and len(cur_s) >= 2:
        # Enumerate all non-identity permutations.
        from itertools import permutations
        identity = tuple(range(len(cur_s)))
        for perm in permutations(range(len(cur_s))):
            if perm != identity:
                yield ("permute", perm)

    # Squeeze: cur has a size-1 dim and target has fewer dims.
    if len(cur_s) > len(tgt_s):
        for d, sz in enumerate(cur_s):
            if sz == 1:
                yield ("squeeze", d)

    # Unsqueeze: target has more dims. Try inserting a size-1 dim at every
    # possible position (the dim may be expanded later via expand). Don't
    # restrict to target dims that are size-1 — expand can grow a size-1
    # inserted dim into any size.
    if len(cur_s) < len(tgt_s):
        for d in range(len(cur_s) + 1):
            yield ("unsqueeze", d)

    # Narrow: any cur dim whose size could shrink to *some* target dim size.
    # Stride match isn't required because the dim may be permuted later
    # (e.g. ``base.narrow(0, ...).permute(...)`` form). The simulation +
    # final equality check filters wrong choices; we just need to enumerate
    # plausible candidates.
    target_sizes = set(tgt_s)
    for d in range(len(cur_s)):
        stride_d = cur_st[d]
        if stride_d == 0:
            continue
        offset_diff = tgt_o - cur_o
        for target_size in target_sizes:
            if 0 < target_size < cur_s[d]:
                max_start = cur_s[d] - target_size
                for start in range(max_start + 1):
                    contrib = start * stride_d
                    if 0 <= contrib <= offset_diff:
                        yield ("narrow", d, start, target_size)

    # Expand: any size-1 cur dim. The expanded dim may later be permuted
    # to a different index, so don't restrict to same-index target size.
    # Try every (cur size-1 dim) × (target expanded-dim size).
    if len(cur_s) == len(tgt_s):
        target_expand_sizes = {tgt_s[d] for d in range(len(tgt_s)) if tgt_st[d] == 0 and tgt_s[d] > 1}
        for d in range(len(cur_s)):
            if cur_s[d] == 1:
                for new_size in target_expand_sizes:
                    yield ("expand", d, new_size)

    # Select: cur has more dims, a non-size-1 dim is being indexed away.
    if len(cur_s) > len(tgt_s):
        for d in range(len(cur_s)):
            stride_d = cur_st[d]
            if stride_d <= 0:
                continue
            offset_diff = tgt_o - cur_o
            if offset_diff < 0:
                continue
            for idx in range(cur_s[d]):
                contrib = idx * stride_d
                if 0 <= contrib <= offset_diff:
                    yield ("select", d, idx)


def _bfs_search_recipe(base, t, max_steps=4):
    """BFS for a recipe that transforms ``base`` into ``t``. Returns the
    first non-blacklisted recipe whose simulation matches ``t``'s
    metadata, or ``None``.

    ``base`` must be contig+offset=0; ``t`` is the view target.
    """
    base_s = tuple(base.size())
    base_st = tuple(base.stride())
    target = (tuple(t.size()), tuple(t.stride()), t.storage_offset())
    start = (base_s, base_st, 0)

    if start == target:
        return ()

    visited = {start: ()}
    frontier = [start]

    for _ in range(max_steps):
        next_frontier = []
        for cur in frontier:
            cur_recipe = visited[cur]
            for step in _gen_step_candidates(cur, target):
                new_state = _simulate_recipe(cur[0], cur[1], cur[2], (step,))
                if new_state is None:
                    continue
                new_recipe = cur_recipe + (step,)
                if new_state == target:
                    # Reached the goal — accept if not blacklisted, else keep
                    # exploring other paths through OTHER intermediate states
                    # (do NOT add target to visited, so alternative-route
                    # arrivals can also be considered).
                    if not _is_known_bad_pattern(new_recipe):
                        return new_recipe
                    continue
                if new_state in visited:
                    continue
                visited[new_state] = new_recipe
                next_frontier.append(new_state)
        frontier = next_frontier
        if not frontier:
            break

    return None


def _detect_view_recipe_safe(t: torch.Tensor):
    """Generic view recipe detector — replaces 5-layer hard-coded path.

    Returns ``(base, recipe)`` where simulation of ``recipe`` on ``base``'s
    metadata produces ``t``'s metadata exactly. ``base`` is contig+offset=0.

    Returns ``None`` if no recipe found within ``max_steps=4`` (caller
    falls back to ``.contiguous()`` with a one-time warning).
    """
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    if t.is_contiguous() and t.storage_offset() == 0:
        return None
    if t.dim() == 0:
        return None

    # Prefer ``t._base`` when it's a clean contig+offset=0 anchor.
    base = getattr(t, "_base", None)
    if not (base is not None and base.is_contiguous() and base.storage_offset() == 0):
        # Synthesize a base from the tensor's storage stride pattern.
        base = _construct_synthetic_base(t)
        if base is None:
            return None

    # Reject 0-dim base: autograd's bprop emits ``scalar.expand(target_shape)``
    # for ``y.sum().backward()`` and similar reductions. Replaying this expand
    # inside a view-aware traced graph (``unsqueeze(scalar) + expand``) makes
    # rebel-compiler abort in ``build_internal`` for fp16 scalar+vector mul
    # (verified 2026-05-07 via cos backward fp16). Falling back to
    # ``.contiguous()`` materializes a real vector buffer so the compile path
    # sees vector × vector and skips the failing IR pattern.
    if base.dim() == 0:
        return None

    recipe = _bfs_search_recipe(base, t, max_steps=4)
    if recipe is None:
        return None
    return base, recipe


def _detect_view_recipe(t: torch.Tensor):
    """Detect the view recipe that transforms a contig base into ``t``.

    Returns ``(base, recipe)`` where:
      - ``base`` is a contig+offset=0 tensor sharing storage with ``t``.
      - ``recipe`` is a tuple of single-step view ops applied in order to
        ``base`` to reproduce ``t``.

    Returns ``None`` if ``t`` is contig (no recipe needed) or if the recipe
    cannot be reverse-engineered. ``None`` signals the caller to fall back
    to ``.contiguous()`` (with a one-time warning log).

    Implementation: delegates to ``_detect_view_recipe_safe`` (BFS + simulation
    + blacklist). The legacy 5-layer hard-coded helpers remain compiled but
    are no longer reached on the dispatch path.
    """
    return _detect_view_recipe_safe(t)


def _detect_view_recipe_legacy(t: torch.Tensor):
    """Legacy 5-layer hard-coded detector. Kept as a reference and for
    differential testing against the generic detector. NOT on the dispatch
    path — see ``_detect_view_recipe`` (which routes to
    ``_detect_view_recipe_safe``)."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    if t.is_contiguous() and t.storage_offset() == 0:
        return None
    if t.dim() == 0:
        return None

    # Layer 1: pure permute via stride alone (most reliable, no _base needed).
    permute = _detect_permute(t)
    if permute is not None:
        base_shape, perm = permute
        return _construct_permute_base(t, base_shape), (("permute", perm),)

    # Layers 2-5: pick the candidate base — prefer t._base when it's a
    # clean contig+offset=0 anchor; otherwise synthesize from the storage
    # via stride pattern. This matters for tensors that came through chains
    # like ``x.view(...).permute(...).unsqueeze(...)`` where ``t._base``
    # points to the immediate parent (still a non-contig view) instead of
    # the contig storage owner.
    base = getattr(t, "_base", None)
    if not (base is not None and base.is_contiguous() and base.storage_offset() == 0):
        base = _construct_synthetic_base(t)
        if base is None:
            return None

    # Layer 2: single-step from (base, t).
    step = _classify_single_step_view(base, t)
    if step is not None:
        return base, (step,)

    # Layer 3: permute + narrow composite.
    composite = _detect_permute_narrow_composite(base, t)
    if composite is not None:
        return base, composite

    # Layer 4: expand + permute composite (common case is just expand with
    # leading new dims).
    composite = _detect_expand_permute_composite(base, t)
    if composite is not None:
        return base, composite

    # Layer 5: permute + unsqueeze composite. Common in rotary embedding /
    # attention reshape (e.g., ``x.view(B,S,n_heads,d).permute(...).unsqueeze(0)``).
    composite = _detect_permute_unsqueeze_composite(base, t)
    if composite is not None:
        return base, composite

    return None


def _maybe_warn_view_fallback(t: torch.Tensor):
    """Increment fallback counter and emit one warning per (shape, stride)
    signature so we can debug new patterns without log spam.

    Also dumps ``t._base`` shape/stride if available, so we can identify
    which view chain produced the unrecognized pattern.
    """
    global _view_fallback_count
    _view_fallback_count += 1
    sig = (tuple(t.size()), tuple(t.stride()), t.storage_offset())
    if sig in _view_fallback_warn_once:
        return
    _view_fallback_warn_once.add(sig)
    base = getattr(t, "_base", None)
    if base is not None:
        base_info = (
            f"base.shape={tuple(base.size())} base.stride={tuple(base.stride())} "
            f"base.offset={base.storage_offset()} base.contig={base.is_contiguous()}"
        )
    else:
        base_info = "base=None"
    rbln_log_warn(
        f"view-on-device fallback (host materialize via .contiguous()): "
        f"shape={sig[0]} stride={sig[1]} offset={sig[2]} — "
        f"{base_info} — "
        f"unrecognized view pattern. Performance impact only; correctness "
        f"is preserved. Add a classifier in ops_utils._classify_single_step_view "
        f"if this pattern becomes hot."
    )


def _apply_view_step(t: torch.Tensor, step):
    """Apply a single recipe step to a tensor inside a wrapper OpModule.forward.

    This is the runtime counterpart to ``_classify_single_step_view``. Each
    branch maps to a single ``aten::*`` op so torch.compile captures it as
    an explicit FX node that rebel-backend lowers on device.
    """
    op = step[0]
    if op == "permute":
        return torch.permute(t, list(step[1]))
    if op == "expand":
        # Two recipe formats accepted:
        #   - generic detector: ``("expand", dim, new_size)`` — expand one
        #     dim only; full shape derived from current tensor.
        #   - legacy detector: ``("expand", full_shape_tuple)`` — full
        #     post-expand shape.
        if len(step) == 3:
            d, new_size = step[1], step[2]
            new_shape = list(t.shape)
            new_shape[d] = new_size
            return t.expand(*new_shape)
        return t.expand(*step[1])
    if op == "narrow":
        return t.narrow(step[1], step[2], step[3])
    if op == "select":
        return t.select(step[1], step[2])
    if op == "reshape":
        return t.reshape(*step[1])
    if op == "squeeze":
        return t.squeeze(step[1])
    if op == "unsqueeze":
        return t.unsqueeze(step[1])
    # Unknown step — pass through. Should never happen if classifier and
    # apply remain in sync.
    return t


def prepare_args_view_aware(args, kwargs_filtered):
    """View-aware variant of ``prepare_args_for_contiguous``.

    For each tensor arg:
    - contig + offset 0 → pass through (no recipe).
    - recognized view chain → replace with the contig base; record a recipe
      tuple of single-step view ops applied in order.
    - any other non-contig → ``.contiguous()`` (legacy fallback, one-time
      warning emitted via ``_maybe_warn_view_fallback``); no recipe.

    Returns:
        ((new_args, new_kwargs), view_recipes, changed)

    where ``view_recipes`` is a tuple aligned with the flat-arg sequence
    produced by ``tree_flatten((args, kwargs_filtered))``. Non-tensor
    positions get ``None``. Tensors with no view get ``None``. Tensors with
    a detected view get a tuple of recipe steps.
    """
    flat_args, args_spec = tree_flatten((args, kwargs_filtered))

    # Pre-pass: detect storage aliasing among tensor args. rebel-compiler's
    # graph optimizer rejects graphs where two inputs alias the same storage
    # (e.g. ``mm(a, a.T)`` where ``a`` and ``a.T`` are the same buffer with
    # different stride). Force ALL aliasing tensors with non-trivial views to
    # ``.contiguous()`` so each operand reaches the runtime as an independent
    # contig buffer. (Observed 2026-04-30 with ``test_cf16_to_cpu_remain_cf16``.)
    storage_ptrs = {}
    aliased_indices = set()
    for idx, a in enumerate(flat_args):
        if not isinstance(a, torch.Tensor) or a.numel() == 0:
            continue
        ptr = a.data_ptr()
        if ptr in storage_ptrs:
            aliased_indices.add(idx)
            aliased_indices.add(storage_ptrs[ptr])
        else:
            storage_ptrs[ptr] = idx

    new_flat = []
    recipes = []
    changed = False
    for idx, a in enumerate(flat_args):
        if not (isinstance(a, torch.Tensor) and a.numel() > 0):
            new_flat.append(a)
            recipes.append(None)
            continue
        if a.is_contiguous() and a.storage_offset() == 0:
            new_flat.append(a)
            recipes.append(None)
            continue
        # Aliased non-contig tensors: rebel can't compile graphs where two
        # inputs alias the same storage. Force ``.contiguous()`` for every
        # aliased operand (independent contig buffers are safe).
        if idx in aliased_indices:
            new_flat.append(a.contiguous())
            recipes.append(None)
            changed = True
            continue
        # Some contiguous-but-offset>0 tensors hit the cpu_fallback path
        # earlier (see is_cpu_fallback_cases storage_offset gate); if a
        # caller still reaches us with offset>0 + contig, we let .contiguous()
        # handle it. The view detector returns None in that case.
        detected = _detect_view_recipe(a)
        if detected is not None:
            base, recipe = detected
            new_flat.append(base)
            recipes.append(recipe)
            changed = True
            continue
        # Fallback: materialize via host (legacy behavior). Emit a warning
        # so unhandled view patterns surface in production logs.
        _maybe_warn_view_fallback(a)
        new_flat.append(a.contiguous())
        recipes.append(None)
        changed = True
    new_args, new_kwargs = tree_unflatten(new_flat, args_spec)
    return (new_args, new_kwargs), tuple(recipes), changed


# Cache of dynamically-built view-aware OpModules, keyed on
# (id(op_callable), view_recipes_tuple). Same key returns the same nn.Module
# instance so torch.compile's per-(model identity, options) cache in
# compile_rbln_cached reuses the same compiled callable across calls.
_view_op_module_cache: dict = {}


def get_view_op_module(op_callable_or_module, view_recipes):
    """Return an ``nn.Module`` whose forward applies the per-arg view recipes
    to its positional args, then invokes the underlying op.

    Accepts either a plain callable (e.g. ``torch.mul``) or a pre-built
    ``nn.Module`` (e.g. the custom-kernel modules defined in
    ``register_custom_ops.py``). With recipes that are all ``None``, returns
    the original module/callable directly (avoids unnecessary wrapping —
    preserves compile_rbln_cached cache hits for the existing module
    identity in ``register_custom_ops``). With non-trivial recipes, builds
    a wrapping module that applies each per-arg recipe step inside its
    forward, then calls the underlying op.

    Cache key is ``(id(callable), recipes)`` so distinct view patterns get
    distinct compiled graphs while a fresh pass-through doesn't churn the
    cache.
    """
    is_module = isinstance(op_callable_or_module, torch.nn.Module)
    op_callable = op_callable_or_module
    has_views = any(r is not None for r in view_recipes)

    if not has_views:
        if is_module:
            return op_callable_or_module  # reuse as-is, no wrap
        # Plain callable, no views: cache a pass-through nn.Module so
        # compile_rbln_cached's identity-keyed cache works the same as the
        # historical ``_<op>_op_module`` style.
        key = (id(op_callable), ())
        cached = _view_op_module_cache.get(key)
        if cached is not None:
            return cached

        class _PlainOp(torch.nn.Module):
            def forward(self, *fwd_args, **fwd_kwargs):
                return op_callable(*fwd_args, **fwd_kwargs)
        inst = _PlainOp().eval()
        _view_op_module_cache[key] = inst
        return inst

    # Non-trivial recipes: build (or reuse cached) view-applying wrapper.
    key = (id(op_callable), tuple(view_recipes))
    cached = _view_op_module_cache.get(key)
    if cached is not None:
        return cached
    captured_recipes = tuple(view_recipes)

    class _ViewOp(torch.nn.Module):
        def forward(self, *fwd_args, **fwd_kwargs):
            new_args = []
            for arg, recipe in zip(fwd_args, captured_recipes):
                if recipe is None or not isinstance(arg, torch.Tensor):
                    new_args.append(arg)
                    continue
                cur = arg
                for step in recipe:
                    cur = _apply_view_step(cur, step)
                new_args.append(cur)
            return op_callable(*new_args, **fwd_kwargs)
    inst = _ViewOp().eval()
    _view_op_module_cache[key] = inst
    return inst


def view_recipes_for_positional(view_recipes_flat, positional_count):
    """Slice the flat ``view_recipes`` tuple from ``prepare_args_view_aware``
    down to just the leading ``positional_count`` entries (matching the
    positional args of an OpModule.forward).
    """
    return view_recipes_flat[:positional_count]


def compile_and_run_view_aware(op_callable, op_name, args, kwargs_filtered, out_tensor):
    """Centralized view-aware rbln-compile dispatch.

    The single entry point used by all generated ``*_rbln`` handlers (via the
    codegen template) and by hand-written custom-kernel handlers in
    ``register_custom_ops.py``. Replaces the old verbose block of
    ``prepare_args_for_contiguous`` + ``compile_rbln_cached`` + warm-cache
    install with a one-liner call.

    Behavior:
    - Detects view recipes (permute / expand / narrow / select / reshape /
      composite) on each positional/keyword tensor arg via
      ``prepare_args_view_aware``. Recognized views are replaced with their
      contig base, and a recipe tuple captures the explicit aten-op chain
      that the wrapper OpModule will apply inside its forward.
    - Builds (or reuses) the wrapper OpModule via ``get_view_op_module``.
    - Compiles via ``compile_rbln_cached``.
    - Skips the C++ shim warm-cache install when any view recipe was applied,
      since the shim's pending key was built from the raw view tensor's shape,
      while the runtime is now compiled for the (different) base shape.
      Future calls re-enter Python and hit the compile_rbln_cached cache —
      ~50 µs/call overhead, still cheap relative to host materialization.

    Lazy imports: ``compile_rbln_cached``, ``warm_cache.install_pending``,
    ``env_utils.use_device_group_tensor_parallel_size``,
    ``device.context_holder.out_tensor_context`` are imported inside the
    function to avoid circular-import risk at module load time
    (``ops_utils`` is imported by everyone; if it imported them at top
    level it would deadlock the package init order).
    """
    from torch_rbln._internal.compile_cache import compile_rbln_cached
    from torch_rbln._internal.env_utils import use_device_group_tensor_parallel_size
    from torch_rbln._internal.warm_cache import install_pending as _install_warm_cache_pending
    from torch_rbln.device.context_holder import out_tensor_context

    # Device 64-elem-align fallback: when an input tensor's last-dim isn't a
    # multiple of 64 elements, the rebel-compiler pipeline wraps the device fn
    # with host-only `contrib_aligned_pad` (pre-pad to 64) + `contrib_aligned_pad`
    # (post-trim back to user shape) + `contrib_dummy_cast` ops that have no
    # RTOSA device-lowering. Each call pays the host-side ApplyTensorDeviceTransform
    # memcpy + a full base-buffer D2H + H2D round-trip
    # (~1MB / op for LLaMA-1B rotary). For these cases routing the whole op
    # through cpu_fallback_path is strictly cheaper: one D2H of the source
    # tensor + a host CPU op + one H2D of the result. Verified 2026-05-08 via
    # IR dump (rbln_tensor_debug.log) and RBLN_VERBOSE=0 trace counts.
    #
    # Condition: ANY tensor arg has last-dim % 64 != 0. Aligned cases keep
    # the view-on-device path (those don't trip the host-transform wrapping).
    # Tensor-only check (NOT list/tuple). Reason: for list-input ops (cat,
    # stack), the view-aware device path packs multiple unaligned tensors
    # into a single fused IR which the rebel-compiler pipeline handles
    # cheaper than per-tensor host fallback. Empirically (LLaMA-1B eager
    # prefill, 2026-05-08): TensorList recursion adds +12% prefill via
    # extra host-cat overhead. Tensor-only catches the dominant rotary
    # rotate_half regression (single-tensor neg/mul on unaligned views)
    # without penalizing TensorList ops.
    def _last_dim_unaligned(t):
        return (isinstance(t, torch.Tensor) and t.dim() > 0
                and t.shape[-1] % 64 != 0)
    if any(_last_dim_unaligned(a) for a in args) or any(
            _last_dim_unaligned(v) for v in kwargs_filtered.values()):
        return cpu_fallback_path(
            op_callable, args, result=out_tensor,
            op_name=op_name, **kwargs_filtered,
        )

    # fp16 div with rounding_mode trunc/floor: rebel-compiler emits IR for
    # the discontinuous rounding op that returns wrong values for entire
    # rows of the output (verified 2026-05-07 with sample 98 of trunc on
    # input shape (357,789): row 338 outputs all 0.0 / -0.0 instead of the
    # correct integer-bucketed quotient, despite both inputs being well
    # within the custom_float16-safe range — not a per-element ULP drift, the
    # device kernel returns wrong values for the affected rows). Route fp16
    # div with trunc/floor through cpu_fallback. Plain ``torch.div(x, y)``
    # (no rounding_mode) and fp32 div are unaffected and still flow through
    # the device path, so no forward perf regression on real workloads
    # that don't use the rounding modes.
    if op_name == "aten::div":
        rmode = kwargs_filtered.get("rounding_mode")
        if rmode in ("trunc", "floor"):
            for a in args:
                if isinstance(a, torch.Tensor) and a.dtype == torch.float16:
                    return cpu_fallback_path(
                        op_callable, args, result=out_tensor,
                        op_name=op_name, **kwargs_filtered,
                    )

    # Comparison ops (output dtype = bool) trip a rebel-compiler abort when
    # the view-aware path replays a narrow+select recipe inside the compiled
    # graph: the resulting IR is ``strided_slice + take + nn.pad + equal +
    # contrib_aligned_pad(bool, …, [-N])`` and ``build_internal`` aborts on
    # the negative-pad-on-bool sequence (verified 2026-05-07 with
    # base[:3,:,:,1] vs contig (3,5,5) eq, vs same shapes through ``add``
    # which compiles fine because the depad runs on fp16 rather than bool).
    # Materialize view inputs to ``.contiguous()`` so the comparison op
    # receives plain tensors and the recipe path is skipped.
    if op_name in {
        "aten::eq", "aten::ne", "aten::gt", "aten::ge", "aten::lt", "aten::le",
    }:
        args = tuple(
            a.contiguous() if isinstance(a, torch.Tensor) and not a.is_contiguous() else a
            for a in args
        )
        kwargs_filtered = {
            k: (v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v)
            for k, v in kwargs_filtered.items()
        }

    (view_args, view_kwargs), view_recipes, _ = prepare_args_view_aware(args, kwargs_filtered)
    has_views = any(r is not None for r in view_recipes)

    compile_options = {"disable_logger": True}
    if not use_device_group_tensor_parallel_size():
        compile_options["tensor_parallel_size"] = 1
    _runtime_holder = []
    compile_options["_runtime_holder"] = _runtime_holder

    if out_tensor is None:
        result_tensor = None
    else:
        if can_use_out_tensor_directly(view_args, dict(view_kwargs, out=out_tensor)):
            result_tensor = out_tensor
        else:
            result_tensor = None

    op_module = get_view_op_module(op_callable, view_recipes)

    with out_tensor_context(result_tensor):
        compiled = compile_rbln_cached(
            op_module,
            dynamic=False,
            options=compile_options,
            device_cache_key=extract_warm_cache_key(*view_args, **view_kwargs),
        )
        external_result = compiled(*view_args, **view_kwargs)
        if result_tensor is None:
            result_tensor = external_result
        elif isinstance(external_result, torch.Tensor) and (external_result.data_ptr() != result_tensor.data_ptr()):
            result_tensor.copy_(external_result)
        if not has_views:
            _install_warm_cache_pending(_runtime_holder, external_result)
    return result_tensor


_ALL_FALLBACK_CASES = frozenset({"dispatch_mode", "reentrant", "trace", "dtype", "scalar", "storage_offset", "nan_inf"})


@lru_cache(maxsize=1)
def _parse_disabled_fallback_cases() -> frozenset:
    """
    Parse `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` environment variable into a frozenset of disabled cases.
    """
    env = os.getenv("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK")
    if env is None:
        return frozenset()
    warnings.warn(
        "Enabling `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` may lead to unexpected behavior. Do NOT use in production."
    )
    cases = frozenset(c.strip() for c in env.split(",") if c.strip())
    return _ALL_FALLBACK_CASES if ("all" in cases) else (cases & _ALL_FALLBACK_CASES)


def is_cpu_fallback_cases(args):
    """
    Determines if a CPU fallback is necessary for an operation.

    This function checks for several conditions that would require an operation to fall back to the CPU:
    0a. **Python trace/debugger**: When a trace function is set (e.g. pdb, coverage, sys.settrace), we fall
        back to CPU to avoid running torch.compile under a tracer. Disable with
        TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=trace for debugging only.
    1. **TorchDispatchMode Active**: When a non-infra TorchDispatchMode is active (e.g. LoggingMode,
       CompositeCompliantTensorMode), we must not attempt torch.compile. Reason: with rbln tensors we do
       add_rbln -> torch.compile(torch.add) -> ...; Dynamo sees the dispatch mode and skips compilation,
       then runs the original torch.add eagerly; that eager call dispatches again and hits our add_rbln
       -> same path repeats -> infinite recursion. So when TorchDispatchMode is on, we fall back to CPU
       and never enter the compile path.
    2. **Data Type Mismatch:** If any of the input tensors are not of the `torch.float16` data type,
       which the `rbln` device is designed to handle.
    3. **Scalar Tensors**: If all input tensors are scalar tensors, rebel-compiler falls back to host ops.
    4. **Storage Offset**: If any tensor has `storage_offset != 0`, fall back to CPU.
    5. **NaN/Inf Values**: When not in deploy mode, if any input tensor contains NaN or Inf values,
       rbln cannot handle them and we need to fall back to CPU. This check converts tensors to CPU
       before checking to ensure accurate detection.
    6. **Reentrancy** (checked last): When we're already inside an RBLN op that uses torch.compile
       (thread-local depth > 0), any nested dispatch (e.g. compiled graph running torch.add -> add_rbln
       again, or print/repr triggering tensor ops) must use CPU fallback to avoid infinite recursion.
       This is an unexpected path; a warning is logged when it triggers. Disable with
       TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=reentrant for debugging only.

    Individual checks can be disabled via `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK`
    (comma-separated names or `"all"`).

    Args:
        args (tuple): A tuple of positional arguments for the operation, which may contain tensors.

    Returns:
        bool: True if a CPU fallback is necessary, False otherwise.
    """
    disabled_cases = _parse_disabled_fallback_cases()

    # 0a: Python trace/debugger – when a trace function is set (e.g. pdb, coverage, sys.settrace), fall back to CPU
    if "trace" not in disabled_cases and sys.gettrace() is not None:
        return True

    # 1: TorchDispatchMode active – run on CPU and do not attempt compile (cheap)
    if "dispatch_mode" not in disabled_cases:
        try:
            from torch.utils._python_dispatch import is_in_torch_dispatch_mode

            if is_in_torch_dispatch_mode(include_infra_modes=False):
                return True
        except ImportError:
            pass

    # Checks 2-5 require tensors; bail early if none
    tensor_args = extract_tensors(args)
    if not tensor_args:
        return False

    # 2: RBLN device can only handle float16 dtype
    if "dtype" not in disabled_cases:
        if any(a.dtype != torch.float16 for a in tensor_args):
            return True

    # 3: fall back to the CPU if all input tensors are scalar tensors
    if "scalar" not in disabled_cases:
        if all(a.ndim == 0 for a in tensor_args):
            return True

    # 4: fall back to the CPU if any contiguous tensor has non-zero storage offset
    if "storage_offset" not in disabled_cases:
        if any(a.is_contiguous() and a.storage_offset() != 0 for a in tensor_args):
            return True

    # 5: fall back to the CPU if any tensor contains NaN/Inf values (heavy: to_cpu + scan; non-deploy only)
    if "nan_inf" not in disabled_cases:
        try:
            from torch_rbln._internal.env_utils import is_rbln_deploy

            if not is_rbln_deploy() and has_invalid_tensor(to_cpu(args)):
                return True
        except ImportError:
            pass

    # 6: Reentrancy – already inside an RBLN compile op (e.g. from print/repr or from compiled graph).
    #     Unexpected path; log a warning when it triggers.
    if "reentrant" not in disabled_cases:
        from torch_rbln._internal.torch_compile_patch_helpers import get_rbln_compile_op_depth

        if get_rbln_compile_op_depth() > 0:
            rbln_log_warn("Unexpected CPU fallback: reentrant dispatch (already inside RBLN compile op)")
            return True

    return False


def cpu_fallback_path(
    target_ops, args, *, result: Optional[torch.Tensor] = None, op_name: Optional[str] = None, **op_kwargs
):
    """
    Perform CPU fallback for the given target operation.

    This function converts the input arguments and keyword arguments from their original device to CPU,
    executes the target operation on these converted arguments, and then converts the result back to the 'rbln'
    device.

    Args:
        target_ops (callable): The operation to be executed on the CPU.
        args (tuple): A tuple of positional arguments that need to be converted to CPU.
        result (Optional[torch.Tensor]): Optional pre-allocated output tensor. If provided and size matches,
            the result will be copied into this tensor.
        op_name (Optional[str]): Operator name like "aten::add" for logging purposes.
        **op_kwargs: Keyword arguments for the operation (e.g. dim=2, out=...). Passed as-is to target_ops.

    Returns:
        torch.Tensor: The result of the target operation, converted back to the 'rbln' device.
    """
    if op_name is not None:
        rbln_log_cpu_fallback(op_name)
    cpu_args = to_cpu(args)
    cpu_op_kwargs = to_cpu(op_kwargs)
    result_cpu = target_ops(*cpu_args, **cpu_op_kwargs)
    if result is not None and result_cpu.size() == result.size():
        result.copy_(result_cpu)
        return result

    # Get device_index from result tensor or from input args/op_kwargs
    # In this context, rbln tensors always have a device_index
    device_id = None
    if result is not None and isinstance(result, torch.Tensor) and result.device.type == "rbln":
        device_id = result.device.index
    else:
        # Find device_id from input tensors
        device_id = extract_device_id_from_inputs(*args, **op_kwargs)

    # Convert result back to rbln device with proper device_index
    # device_id should always be available when rbln tensors are present
    assert device_id is not None, "device_id should be found from rbln tensors"
    result = result_cpu.to(f"rbln:{device_id}")
    return result


def is_inplace_op(args, kwargs) -> bool:
    """
    Determine whether the current call is an in-place operation.

    The function scans all positional and keyword arguments to locate the
    tensor. It then checks whether this tensor shares the same storage
    (identical `data_ptr` on the same device) with any input tensor.

    Args:
        args (tuple): Positional arguments originally given to the operator.
        kwargs (dict): Keyword arguments originally given to the operator.

    Returns:
        bool: `True` if the `out_tensor` aliases the storage of any input
        tensor (in-place); otherwise `False`.
    """
    out_t = kwargs.get("out", None)
    if out_t is None or not torch.is_tensor(out_t):
        return False

    input_tensors = [t for t in args if torch.is_tensor(t)]
    input_tensors += [v for v in kwargs.values() if torch.is_tensor(v) and v is not out_t]

    for t in input_tensors:
        if (t is out_t) or (t.data_ptr() == out_t.data_ptr() and t.device == out_t.device):
            return True

    return False


def can_use_out_tensor_directly(args: tuple, kwargs: dict) -> bool:
    """
    Check if the out_tensor can be used directly by the compiler.

    This function checks several conditions to determine if the output tensor
    can be used directly without creating a temporary tensor:
    1. Not an in-place operation
    2. Tensor is neither empty nor scalar
    3. Tensor is contiguous
    4. Tensor has zero storage offset
    5. dtype is float16

    Args:
        args (tuple): Positional arguments for in-place operation check.
        kwargs (dict): Keyword arguments containing 'out' key with the output tensor to check.

    Returns:
        bool: True if the out_tensor can be used directly, False otherwise.
    """
    out_tensor = kwargs.get("out")
    if out_tensor is None or out_tensor.data_ptr() == 0:
        return False

    # Check conditions:
    # 1. Not inplace operation
    # 2. Neither empty nor scalar
    # 3. Contiguous
    # 4. Zero storage offset
    # 5. dtype is float16
    return (
        not is_inplace_op(args, kwargs)
        and ((out_tensor.numel() > 0) and len(out_tensor.size()) > 0)
        and out_tensor.is_contiguous()
        and (out_tensor.storage_offset() == 0)
        and (out_tensor.dtype == torch.float16)
    )


def _ceil_to_nearest_multiple_of_64(n):
    """
    Rounds up the given integer `n` to the nearest multiple of 64.

    Args:
        n (int): The integer to be rounded up.

    Returns:
        int: The smallest multiple of 64 that is greater than or equal to `n`.
    """
    return math.ceil(n / 64) * 64
