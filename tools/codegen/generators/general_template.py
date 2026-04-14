"""
Code templates for general function code generation.

Contains all code templates used in general function generation.
"""

from typing import Set  # noqa: UP035


class GeneralTemplates:
    """Templates for general function code generation."""

    class ArgsProcessing:
        """Templates for argument processing."""

        @staticmethod
        def kwargs_filter(exclude_kwargs: Set[str]) -> str:
            """Generate kwargs filter code."""
            if len(exclude_kwargs) == 1:
                # Single kwarg to exclude (default case: only 'out')
                kwarg = next(iter(exclude_kwargs))
                return f"""    kwargs_filtered = {{k: v for k, v in kwargs.items() if k != "{kwarg}"}}  # remove {kwarg} kwarg
"""
            else:
                # Multiple kwargs to exclude
                kwargs_str = "{" + ", ".join(f'"{k}"' for k in sorted(exclude_kwargs)) + "}"
                return f"""    kwargs_filtered = {{k: v for k, v in kwargs.items() if k not in {kwargs_str}}}
"""

        @staticmethod
        def tensor_args_extraction() -> str:
            """Generate tensor arguments extraction code."""
            return """    tensor_args = extract_tensors(args)
"""

        @staticmethod
        def contiguous_preparation_on_device() -> str:
            """Generate contiguous preparation for on-device operations."""
            return (
                "        (contig_args, contig_kwargs), changed_any = "
                "prepare_args_for_contiguous_on_device(args, kwargs_filtered)\n"
            )

        @staticmethod
        def contiguous_preparation_default() -> str:
            """Generate default contiguous preparation."""
            return (
                "        (contig_args, contig_kwargs), changed_any = "
                "prepare_args_for_contiguous(args, kwargs_filtered)\n"
            )

    class EmptyTensor:
        """Templates for empty tensor handling."""

        @staticmethod
        def reduction() -> str:
            """Generate empty tensor handling for reduction operations."""
            return """    if tensor_args and all(a.numel() == 0 for a in tensor_args):   # for handling empty tensor
        # Extract dim from args[1] if not in kwargs
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        # Extract keepdim from args[2] if not in kwargs
        keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
        return handle_empty_reduction(tensor_args[0], dim, keepdim)
"""

        @staticmethod
        def mm() -> str:
            """Generate empty tensor handling for matrix multiplication operations."""
            return """    if len(tensor_args) != 2:
        raise RuntimeError("mm requires 2 inputs")

    if tensor_args and (tensor_args[0].numel() == 0 or tensor_args[1].numel() == 0):   # for handling empty tensor
        return handle_empty_mm(tensor_args)
"""

        @staticmethod
        def where() -> str:
            """Generate empty tensor handling for where operations."""
            return """    if args[0].numel() == 0:
        return handle_empty_where(args)
"""

        @staticmethod
        def broadcastable() -> str:
            """Generate empty tensor handling for broadcastable operations."""
            return """    args = broadcast_args_general(tensor_args, args)
    if tensor_args and all(a.numel() == 0 for a in tensor_args):   # for handling empty tensor
        return handle_empty_binary(args)
"""

        @staticmethod
        def addmm() -> str:
            """Generate empty tensor handling for addmm operations."""
            return """    if len(tensor_args) != 3:
        raise RuntimeError("addmm requires three inputs")
    args = addmm_broadcast_args(tensor_args, args)
    if tensor_args and (tensor_args[1].numel() == 0 or tensor_args[2].numel() == 0):   # for handling empty tensors
        beta = kwargs.get("beta", 1)
        return handle_empty_addmm(tensor_args, beta)
"""

        @staticmethod
        def linear() -> str:
            """Generate default empty tensor handling."""
            # linear(input, weight, bias=None): if input is empty, output is empty
            # Unlike other ops, linear only needs input to be empty (not weight/bias)
            # because output shape depends on input's batch dimensions
            return """    if tensor_args and tensor_args[0].numel() == 0:   # for handling empty input tensor
        return handle_empty_linear(tensor_args)
"""

        @staticmethod
        def default() -> str:
            """Generate default empty tensor handling."""
            return """    if tensor_args and all(a.numel() == 0 for a in tensor_args):   # for handling empty tensor
        return handle_empty_tensor(tensor_args)
"""

    class FunctionBody:
        """Templates for function body generation."""

        @staticmethod
        def start(kernel_name: str) -> str:
            """Generate function body start."""
            return f"""
def {kernel_name}(*args, **kwargs):
"""

        @staticmethod
        def main(target: str, root_name: str, op_namespace: str) -> str:
            """Generate main function body logic."""
            return f"""
    # return values
    result_tensor = None

    out_tensor = kwargs.get('out', None)
    if is_cpu_fallback_cases(args):
        result = cpu_fallback_path({target}, args, result=out_tensor, op_name="{op_namespace}::{root_name}", **kwargs_filtered)
        result_tensor = result
    else:
        # device tensor handling
"""

        @staticmethod
        def op_module_definition(root_name: str, target: str) -> str:
            """Generate per-op nn.Module via make_op_module factory (opaque under torch.compile). Module-level."""
            var_name = f"_{root_name}_op_module"
            return f"""
{var_name} = make_op_module({target})
"""

        @staticmethod
        def compile_section(root_name: str, target: str) -> str:
            """Generate compilation and execution section (uses compile_and_execute utility)."""
            op_module_var = f"_{root_name}_op_module"
            return (
                f"        result_tensor = compile_and_execute(\n"
                f"            {op_module_var}, contig_args, contig_kwargs, out_tensor=out_tensor\n"
                f"        )\n"
                f"\n"
                f"    return result_tensor, result_tensor.shape\n"
            )
