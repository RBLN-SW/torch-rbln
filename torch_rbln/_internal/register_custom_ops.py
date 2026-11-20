import threading
from typing import cast

import torch

from torch_rbln._internal.ops_utils import (
    _detect_view_recipe,
    compile_and_run_view_aware,
    cpu_fallback_path,
    finalize_output_tensor,
    get_fallback_reason,
    handle_empty_binary,
    is_type_promotion_allowed,
)


def custom_softmax_out_rbln(self, dim: int, half_to_float: bool, *, out):  # noqa: ARG001, FBT001
    fallback_reason = get_fallback_reason(self)
    if fallback_reason:
        result_tensor = cpu_fallback_path(
            torch.softmax, (self,), result=out, op_name="aten::softmax", reason=fallback_reason, dim=dim
        )
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

    result_tensor = cast("torch.Tensor", result_tensor)
    finalize_output_tensor(out, result_tensor, result_tensor.shape)


def _is_integer_exponent(exponent) -> bool:
    """Device can only handle integer exponents (2, 2.0 OK; 2.5, inf, nan fall back)."""
    if isinstance(exponent, int):
        return True
    if isinstance(exponent, float):
        return exponent.is_integer()  # False for inf and nan, where round() would raise
    return False


def pow_tensor_scalar_out_rbln(self, exponent, *, out):
    # pow.Tensor_Scalar: exponent must be scalar (int/float), not tensor
    if isinstance(exponent, torch.Tensor):
        msg = "pow.Tensor_Scalar expects scalar exponent (int/float), not tensor."
        raise TypeError(msg)

    # Validate device/dtype match (preserved from original implementation)
    if self.device != out.device:
        msg = f"Input device {self.device} does not match output device {out.device}."
        raise RuntimeError(msg)
    if self.dtype != out.dtype and not is_type_promotion_allowed((self,), out):
        msg = f"Unsafe cast: input has dtype {self.dtype} but output tensor has dtype {out.dtype}."
        raise RuntimeError(msg)

    if self.numel() == 0:
        result_tensor, _ = handle_empty_binary((self,))
    elif not _is_integer_exponent(exponent):
        result_tensor = cpu_fallback_path(
            torch.pow, (self, exponent), result=out, op_name="aten::pow", reason="non_integer_exponent"
        )
    elif fallback_reason := get_fallback_reason((self, exponent)):
        result_tensor = cpu_fallback_path(
            torch.pow, (self, exponent), result=out, op_name="aten::pow", reason=fallback_reason
        )
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

    result_tensor = cast("torch.Tensor", result_tensor)
    finalize_output_tensor(out, result_tensor, result_tensor.shape)


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """The op under the view recipe: lay the replayed view out contiguously.

    ``torch.clone`` would be the obvious choice and is wrong here -- it preserves the
    source's memory format, so cloning a permuted view hands back the same strides, not
    the contiguous destination layout ``copy_`` was asked for.
    """
    return t.contiguous()


# Geometries (sizes, strides, storage offset, dtype, device) for which the compiled
# view copy has been shown to permute a buffer in place correctly. Writing the buffer
# a program reads is only correct when its schedule finishes reading a region before
# writing it -- a property of the compiler's tiling, not of anything here -- so the
# first in-place copy of a geometry is checked against a host reference and the
# verdict remembered.
_inplace_view_copy_verified: dict[tuple, bool] = {}
_inplace_view_copy_lock = threading.Lock()


def _inplace_view_copy_key(src: torch.Tensor) -> tuple:
    return (tuple(src.shape), tuple(src.stride()), src.storage_offset(), src.dtype, str(src.device))


def _run_view_copy(src: torch.Tensor, out: torch.Tensor, *, allow_alias: bool) -> None:
    result = compile_and_run_view_aware(
        _materialize, "torch_rbln::copy_strided_view", (src,), {}, out, allow_alias=allow_alias
    )
    if isinstance(result, torch.Tensor) and result.data_ptr() != out.data_ptr():
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


# ``inplace`` is positional because the op schema declares it so.
def copy_strided_view_rbln(src, out, inplace=False) -> bool:  # noqa: FBT002
    """``out.copy_(src)`` for a strided device ``src``, as one compiled device program.

    ``copy_``'s device->device path calls this instead of walking the copy one
    contiguous run at a time. The view chain is replayed *inside* the compiled graph by
    the same machinery every other view-aware op uses (``prepare_args_view_aware``
    classifies it, ``get_view_op_module`` replays it), so the device reads the base
    buffer and writes ``out`` in one program -- no descriptor per run, no host
    round-trip. Nothing here is specific to a permutation: whatever the detector can
    classify, this covers.

    ``inplace`` says ``out`` is the very buffer ``src`` views, so the program permutes
    it in its own storage and a caller that only needs the permuted layout holds one
    buffer instead of two. ``copy_`` never asks for it -- ATen refuses partially
    overlapping pairs before dispatch -- so an aliasing pair is always a caller's
    explicit intent. ``out`` must then be the whole storage ``src`` views: contiguous,
    storage offset 0, the same base address and byte size. Correctness rests on the
    compiled schedule finishing its reads of a region before writing it, which is the
    compiler's tiling rather than anything here, so the first call for a geometry is
    checked against a host reference and a geometry that fails raises.

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
        msg = (
            "copy_strided_view(inplace=True): out must be the whole buffer src views "
            "(contiguous, storage_offset 0, same base address and size)"
        )
        raise RuntimeError(msg)
    _verify_inplace_geometry(src, out)
    return True


# The schema is declared in RBLNRegisterOps.cpp, in the namespace this backend owns;
# only the implementation lives here, because it drives the compile path.
torch_rbln_impl = torch.library.Library("torch_rbln", "IMPL")
torch_rbln_impl.impl("copy_strided_view", copy_strided_view_rbln, "PrivateUse1")
