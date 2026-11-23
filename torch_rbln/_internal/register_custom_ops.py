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
            align_check="none",
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


def copy_strided_view_rbln(src, out) -> bool:
    """``out.copy_(src)`` for a strided device ``src``, as one compiled device program.

    ``copy_``'s device->device path calls this instead of walking the copy one
    contiguous run at a time. The view chain is replayed *inside* the compiled graph by
    the same machinery every other view-aware op uses (``prepare_args_view_aware``
    classifies it, ``get_view_op_module`` replays it), so the device reads the base
    buffer and writes ``out`` in one program -- no descriptor per run, no host
    round-trip. Nothing here is specific to a permutation: whatever the detector can
    classify, this covers.

    Returns False when the detector cannot classify ``src``, leaving ``copy_`` on its
    existing route. That check has to happen before ``compile_and_run_view_aware``:
    unclassified views take ``.contiguous()`` in there, which re-enters ``copy_``.
    """
    if _detect_view_recipe(src) is None:
        return False
    result = compile_and_run_view_aware(_materialize, "torch_rbln::copy_strided_view", (src,), {}, out)
    if isinstance(result, torch.Tensor) and result.data_ptr() != out.data_ptr():
        # The compile path only writes the caller's buffer for dtypes in
        # SupportedDtypes.dispatch; otherwise it hands back its own. Both are contiguous
        # here, so this lands on copy_'s direct memcpy and cannot recurse.
        out.copy_(result)
    return True


# The schema is declared in RBLNRegisterOps.cpp, in the namespace this backend owns;
# only the implementation lives here, because it drives the compile path.
torch_rbln_impl = torch.library.Library("torch_rbln", "IMPL")
torch_rbln_impl.impl("copy_strided_view", copy_strided_view_rbln, "PrivateUse1")
