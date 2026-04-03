"""
Code templates for variant function code generation.

Contains all code templates used in variant function generation (out, inplace, normal).
"""

from typing import Optional


class VariantTemplates:
    """Templates for variant function code generation."""

    class FunctionSignature:
        """Templates for function signatures."""

        @staticmethod
        def signature(kernel_name: str) -> str:
            """Generate function signature."""
            return f"""
def {kernel_name}(*args, **kwargs):
"""

    class InputExtraction:
        """Templates for input extraction and processing."""

        @staticmethod
        def out_tensor(out_tensor_kwarg: str, out_values_kwarg: Optional[str] = None) -> str:
            """Generate out tensor extraction code."""
            if out_values_kwarg:
                return f"""    out_tensor = kwargs.get('{out_tensor_kwarg}')
    out_values_tensor = kwargs.get('{out_values_kwarg}')
"""
            else:
                return """    out_tensor = kwargs.get('out')
"""

        @staticmethod
        def initialization() -> str:
            """Generate input tracking variables initialization."""
            return """    input_dtype = None
    input_device = None
"""

        @staticmethod
        def initialization_device_only() -> str:
            """For compare out ops with self: only device is tracked."""
            return """    input_device = None
"""

        @staticmethod
        def with_self(self_arg_pos: int) -> str:
            """Generate input extraction when self argument exists."""
            return f"""    in_tensor = args[{self_arg_pos}] if {self_arg_pos} < len(args) else None
    if in_tensor is None:
        raise RuntimeError("No input tensor found.")
    if isinstance(in_tensor, torch.Tensor):
        input_dtype = in_tensor.dtype
        input_device = in_tensor.device
"""

        @staticmethod
        def with_self_device_only(self_arg_pos: int) -> str:
            """Like with_self but only device (compare out ops: dtype checked on output tensor)."""
            return f"""    in_tensor = args[{self_arg_pos}] if {self_arg_pos} < len(args) else None
    if in_tensor is None:
        raise RuntimeError("No input tensor found.")
    if isinstance(in_tensor, torch.Tensor):
        input_device = in_tensor.device
"""

        @staticmethod
        def array_start() -> str:
            """Generate start of array input extraction."""
            return """    if isinstance(args[0], (list, tuple)) and all(
        isinstance(t, torch.Tensor) for t in args[0] if t is not None
    ):
        for arg in args[0]:
"""

        @staticmethod
        def single_mem_loc_check() -> str:
            """Generate single memory location violation check."""
            return """            if (
                arg is not None
                and not out_tensor.data_ptr() == 0
                and arg.untyped_storage().data_ptr() == out_tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("unsupported operation: some elements of the input tensor and the written-to tensor "
                                   "refer to a single memory location.")
"""

        @staticmethod
        def dtype_promote() -> str:
            """Generate dtype extraction with promotion."""
            return """            input_dtype = (
                arg.dtype
                if input_dtype is None
                else torch.promote_types(input_dtype, arg.dtype)
            )
"""

        @staticmethod
        def dtype_strict() -> str:
            """Generate strict dtype extraction."""
            return """            if input_dtype is None:
                input_dtype = arg.dtype
            else:
                if input_dtype != arg.dtype:
                    raise RuntimeError(f"Unsafe cast: input has dtype {input_dtype} but other input tensor has dtype {arg.dtype}.")
"""

        @staticmethod
        def device_from_array() -> str:
            """Generate device extraction from array inputs."""
            return """            if input_device is None:
                input_device = arg.device
            else:
                if input_device != arg.device:
                    raise RuntimeError(f"Input device {input_device} does not match other input device {arg.device}.")
"""

    class Validation:
        """Templates for validation logic."""

        @staticmethod
        def device_with_kwarg() -> str:
            """Generate device validation when device kwarg exists."""
            return """    if input_device is not None and input_device != out_tensor.device:
        #if kwargs.get("device", None) is None:
        #    pass  # device is None → implicit to device may be acceptable → pass here
        #else:
        raise RuntimeError(f"Input device {input_device} does not match output device {out_tensor.device}.")
"""

        @staticmethod
        def device_default() -> str:
            """Generate default device validation."""
            return """    if input_device is not None and input_device != out_tensor.device:
        raise RuntimeError(f"Input device {input_device} does not match output device {out_tensor.device}.")
"""

        @staticmethod
        def dtype_reduction() -> str:
            """Generate dtype validation for reduction operations."""
            return """    if 'dtype' in kwargs and kwargs['dtype'] is not None:
        input_dtype = kwargs['dtype']
        if input_dtype != out_tensor.dtype and not is_type_promotion_allowed_dtype(input_dtype, out_tensor.dtype):
            raise RuntimeError(f"Unsafe cast: input has dtype {input_dtype} but output tensor has dtype {out_tensor.dtype}.")
"""

        @staticmethod
        def dtype_compare() -> str:
            """Generate dtype validation for comparison operations."""
            return """    if out_tensor.dtype != torch.bool:
        raise RuntimeError(
            f"Output tensor for comparison operator must have dtype torch.bool, but got {out_tensor.dtype}."
        )
"""

        @staticmethod
        def dtype_default() -> str:
            """Generate default dtype validation."""
            return """    if input_dtype is not None and input_dtype != out_tensor.dtype:
        if not is_type_promotion_allowed(args, out_tensor):
            raise RuntimeError(f"Unsafe cast: input has dtype {input_dtype} but output tensor has dtype {out_tensor.dtype}.")
"""

    class FunctionVariants:
        """Templates for different function variants."""

        @staticmethod
        def out_footer(general_kernel_name: str) -> str:
            """Generate footer for out functions."""
            return f"""    result, _ = {general_kernel_name}(*args, **kwargs)
    finalize_output_tensor(out_tensor, result, result.shape, args, kwargs)
"""

        @staticmethod
        def inplace(kernel_name: str, general_kernel_name: str) -> str:
            """Generate inplace function code."""
            return f"""
def {kernel_name}(*args, **kwargs):
    # TODO: try zero-copy by passing args[0] as out tensor (need to check for violations)
    result, _ = {general_kernel_name}(*args, **kwargs)
    args[0].copy_(result)
"""

        @staticmethod
        def normal(kernel_name: str, general_kernel_name: str) -> str:
            """Generate normal function code."""
            return f"""
def {kernel_name}(*args, **kwargs):
    result, _ = {general_kernel_name}(*args, **kwargs, out=None)
    return result
"""
