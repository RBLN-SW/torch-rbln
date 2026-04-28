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
    for a in args:
        if isinstance(a, torch.Tensor):  # torch.Tensor외에 list,tuple,dict tensor도 다뤄야할 수 있음
            ret = torch.empty(a.shape, dtype=a.dtype, device="rbln")
            return ret, ret.shape
    raise RuntimeError("Can't find reference tensor for out")


def broadcast_args_general(tensor_args, args):
    if _needs_broadcast(tensor_args):
        try:
            # broadcast_tensors returns the broadcasted result of the arguments
            broadcasted = torch.broadcast_tensors(*tensor_args)
            # if broadcast was successful, replace args with broadcasted
            new_args = []
            tensor_idx = 0
            for a in args:
                if isinstance(a, torch.Tensor):  # torch.Tensor외에 list,tuple,dict tensor도 다뤄야할 수 있음
                    new_args.append(broadcasted[tensor_idx])
                    tensor_idx += 1
                else:
                    new_args.append(a)
            return tuple(new_args)
        except RuntimeError as e:
            tensor_shapes = [tuple(t.shape) for t in tensor_args]
            raise RuntimeError(f"Broadcasting failed for tensor shapes={tensor_shapes}") from e
    return args


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
