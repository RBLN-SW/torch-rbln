"""
General function code generator.

Generates code for general (non-custom) function implementations.

Responsibilities:
- Generate general function body (kwargs filtering, empty tensor handling, compilation, etc.)

Generated code structure:
1. Function body:
   - kwargs filtering (remove 'out')
   - Tensor argument extraction
   - Empty tensor handling (operation type specific)
   - CPU fallback check
   - Contiguous preparation
   - Alignment fallback check
   - Compilation and execution (direct torch op compilation)

Templates: See general_template.py
"""

from typing import Callable, cast

from ..config import OpCategories
from .general_template import GeneralTemplates


class GeneralFunctionGenerator:
    """Generates code for general (non-custom) function implementations."""

    def __init__(self, op_categories: OpCategories):
        self.op_categories = op_categories
        self.templates = GeneralTemplates

    def generate_kwargs_filter(self, root_name: str) -> str:
        """Generate kwargs filtering code based on operation type."""
        exclude_kwargs = self.op_categories.get_kwargs_to_exclude(root_name)
        if exclude_kwargs is None:
            exclude_kwargs = {"out"}  # Default: only exclude 'out'
        return self.templates.ArgsProcessing.kwargs_filter(exclude_kwargs)

    def generate_empty_tensor_handling(self, root_name: str) -> str:
        """Generate empty tensor handling code based on operation type."""
        template_method_name = self.op_categories.get_empty_tensor_template_method(root_name)
        template_method = getattr(self.templates.EmptyTensor, template_method_name)
        # Cast to Callable[[], str] since all EmptyTensor methods return str
        method: Callable[[], str] = cast(Callable[[], str], template_method)
        return method()

    def generate_contiguous_preparation(self, root_name: str) -> str:
        """Generate contiguous tensor preparation code."""
        # TEMPORARY DISABLED DUE TO TRANSPOSE ISSUE IN RBLN
        # if self.op_categories.needs_contiguous_on_device(root_name):
        #     return self.templates.ArgsProcessing.contiguous_preparation_on_device()
        # else:
        return self.templates.ArgsProcessing.contiguous_preparation_default()

    def generate_function_body(self, kernel_name: str, root_name: str, python_module: str, op_namespace: str) -> str:
        """Generate the complete function body for a general operation."""
        if python_module == "nn":
            target = f"torch.nn.functional.{root_name}"
        else:
            target = f"torch.{root_name}"

        # Per-op module (class + instance) with literal forward for opaque torch.compile
        code = self.templates.FunctionBody.op_module_definition(root_name, target)
        code += self.templates.FunctionBody.start(kernel_name)
        code += self.generate_kwargs_filter(root_name)
        if self.op_categories.needs_tensor_args_extraction_before_empty_handling(root_name):
            code += self.templates.ArgsProcessing.tensor_args_extraction()
        code += self.generate_empty_tensor_handling(root_name)
        code += self.templates.FunctionBody.main(target, root_name, op_namespace)
        code += self.generate_contiguous_preparation(root_name)
        code += self.templates.FunctionBody.compile_section(root_name, target)

        return code

    def generate(self, kernel_name: str, root_name: str, python_module: str, op_namespace: str) -> str:
        """Generate complete code for a general function type."""
        code = self.generate_function_body(kernel_name, root_name, python_module, op_namespace)
        return code
