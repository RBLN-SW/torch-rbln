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
        """Templates for function body generation.

        2026-04-30 update: handlers delegate the compile + dispatch step to
        ``compile_and_run_view_aware`` (in ops_utils.py) so view-on-device
        detection (permute/expand/narrow/select/composite) and the
        pre-existing contig + cpu-fallback + warm-cache-install logic all
        live in one place. The generated handler is now ~12 lines instead
        of ~50, and the per-op ``OpModule_<name>`` class is no longer
        emitted (the helper builds dynamic wrapper modules keyed on
        ``(op_callable, view_recipes)`` so identical view patterns reuse
        the same compiled callable).
        """

        @staticmethod
        def start(kernel_name: str) -> str:
            """Generate function body start. No need to import
            out_tensor_context — the helper does it lazily."""
            return f"""
def {kernel_name}(*args, **kwargs):
"""

        @staticmethod
        def main(target: str, root_name: str, op_namespace: str) -> str:
            """Generate main function body logic.

            Delegates the device-path dispatch to
            ``compile_and_run_view_aware`` (single helper call) and keeps
            the cpu-fallback gate inline since it short-circuits without
            going to the device.
            """
            return f"""
    out_tensor = kwargs.get('out', None)
    if is_cpu_fallback_cases(args):
        result_tensor = cpu_fallback_path(
            {target},
            args,
            result=out_tensor,
            op_name="{op_namespace}::{root_name}",
            **kwargs_filtered,
        )
    else:
        result_tensor = compile_and_run_view_aware(
            {target}, "{op_namespace}::{root_name}", args, kwargs_filtered, out_tensor,
        )

    return result_tensor, result_tensor.shape
"""

        @staticmethod
        def op_module_definition(root_name: str, target: str) -> str:
            """Per-op ``OpModule_<name>`` class is no longer emitted: the
            view-aware helper builds a wrapper OpModule dynamically and
            caches it per ``(op_callable, view_recipes)``. Returning the
            empty string keeps the rest of the generator untouched.
            """
            return ""

        @staticmethod
        def compile_section(root_name: str, target: str) -> str:
            """The legacy compile/dispatch block has been folded into the
            ``main`` template's call to ``compile_and_run_view_aware``.
            Keep the static method as a stub to preserve the call site
            in ``general.py``.
            """
            return ""
