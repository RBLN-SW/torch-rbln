import functools
from collections.abc import Callable
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


def _run_view_copy(
    src: torch.Tensor, out: torch.Tensor, *, check_alias: Callable[[Callable[[], None]], bool] | None = None
) -> None:
    result = compile_and_run_view_aware(
        _materialize, "torch_rbln::copy_strided_view", (src,), {}, out, check_alias=check_alias
    )
    if isinstance(result, torch.Tensor) and result.data_ptr() != out.data_ptr():
        # The compile path only writes the caller's buffer for dtypes in
        # SupportedDtypes.dispatch; otherwise it hands back its own. Both are contiguous
        # here, so this lands on copy_'s direct memcpy and cannot recurse.
        out.copy_(result)


def _copies_exactly(src: torch.Tensor, out: torch.Tensor) -> bool:
    """Whether the compiled program returns ``src``'s bytes unchanged in ``out``.

    The gate ``copy_`` applies before it reaches this op (``try_view_copy`` in
    RBLNCopy.cpp): the device rewrites every dtype but bf16 to a narrower float, and
    the compiler lowers a tensor to the device only when its last dim is a multiple of
    64 elements.
    """
    return src.dtype == torch.bfloat16 and out.dtype == src.dtype and src.size(-1) % 64 == 0 and out.size(-1) % 64 == 0


def _permutes_in_place(src: torch.Tensor, out: torch.Tensor, run: Callable[[], None]) -> bool:
    """Whether ``run``, the program bound to write ``out`` over ``src``, copies exactly.

    The compile path calls this once per compiled program, before it writes the
    caller's buffer in place. The check runs on random values of its own, not the
    caller's: a constant or uninitialized buffer proves little, and NaN bit patterns
    would fail a value comparison, so bytes are compared. The caller's bytes wait on
    the host meanwhile and go back before the real copy, which costs host copies
    rather than the second device buffer the in-place form exists to avoid.
    """
    saved = out.cpu()
    generator = torch.Generator().manual_seed(0)
    out.copy_(torch.randn(out.shape, generator=generator).to(out.dtype))
    reference = src.cpu().contiguous()
    run()
    verified = torch.equal(out.cpu().view(torch.uint8), reference.view(torch.uint8))
    out.copy_(saved)
    return verified


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
    storage offset 0, the same base address and byte size. A caller that keeps the
    program relies on it copying exactly, so an in-place copy the device would not
    return bit for bit (not bf16, or a last dim not a multiple of 64) is declined, the
    way ``copy_`` declines it. Correctness also rests on the compiled schedule finishing
    its reads of a region before writing it, which is the compiler's tiling rather than
    anything here, so the first in-place call of each compiled program is checked
    against a host reference and a program that fails raises.

    Returns False, leaving ``out`` untouched and the caller on its existing route, when
    the detector cannot classify ``src`` or, with ``inplace``, when the copy would not
    be exact. Raises RuntimeError when ``inplace`` is set and ``out`` is not ``src``'s
    whole storage, ``out`` cannot be bound as the program's output, or the program
    does not permute it in place correctly.
    """
    if _detect_view_recipe(src) is None:
        return False
    if not inplace:
        _run_view_copy(src, out)
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
    if not _copies_exactly(src, out):
        return False
    _run_view_copy(src, out, check_alias=functools.partial(_permutes_in_place, src, out))
    return True


# The schema is declared in RBLNRegisterOps.cpp, in the namespace this backend owns;
# only the implementation lives here, because it drives the compile path.
torch_rbln_impl = torch.library.Library("torch_rbln", "IMPL")
torch_rbln_impl.impl("copy_strided_view", copy_strided_view_rbln, "PrivateUse1")
