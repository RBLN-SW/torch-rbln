"""
Operation variant code generator.

Generates code for specific operation variants (out, inplace, normal).

Responsibilities:
- Out Function: Input validation + general function call + output processing
- Inplace Function: General function call + copy_
- Normal Function: General function call (out=None)

Generation flow:
1. Out Function:
   - Function signature
   - Out tensor extraction
   - Input initialization
   - Input extraction (self or array)
   - Device validation
   - Dtype validation
   - General function call + output processing

2. Inplace Function:
   - General function call
   - args[0].copy_(result)

3. Normal Function:
   - General function call (out=None)
   - Return result

Templates: See variants_template.py
"""

from typing import Optional

from ..analyzer import FunctionAnalyzer
from ..config import OpCategories
from .variants_template import VariantTemplates


class OpCodeGenerator:
    """Generates code for specific operation variants (out, inplace, normal)."""

    def __init__(self, op_categories: OpCategories, function_analyzer: FunctionAnalyzer):
        self.op_categories = op_categories
        self.analyzer = function_analyzer
        self.templates = VariantTemplates

    def generate_out_function(
        self,
        kernel_name: str,
        root_name: str,
        general_kernel_name: str,
        self_arg_pos: Optional[int],
        has_device_kwarg: bool,
    ) -> str:
        """Generate code for out variant of an operation."""
        code = self.templates.FunctionSignature.signature(kernel_name)

        # Extract out tensor
        out_tensor_kwarg, out_values_kwarg = self.op_categories.get_out_param_names(kernel_name)
        code += self.templates.InputExtraction.out_tensor(out_tensor_kwarg, out_values_kwarg)

        # Initialize input tracking variables (compare + self: dtype comes from out tensor checks)
        if self_arg_pos is not None and root_name in self.op_categories.COMPARE_OPS:
            code += self.templates.InputExtraction.initialization_device_only()
            code += self.templates.InputExtraction.with_self_device_only(self_arg_pos)
        elif self_arg_pos is not None:
            code += self.templates.InputExtraction.initialization()
            code += self.templates.InputExtraction.with_self(self_arg_pos)
        else:
            code += self.templates.InputExtraction.initialization()
            code += self._generate_input_extraction_from_array(root_name)

        # Validate device
        if has_device_kwarg:
            code += self.templates.Validation.device_with_kwarg()
        else:
            code += self.templates.Validation.device_default()

        # Validate dtype
        code += self._generate_dtype_validation(root_name)

        # Generate function footer
        code += self.templates.FunctionVariants.out_footer(general_kernel_name)

        return code

    def generate_inplace_function(self, kernel_name: str, general_kernel_name: str) -> str:
        """Generate code for inplace variant of an operation."""
        return self.templates.FunctionVariants.inplace(kernel_name, general_kernel_name)

    def generate_normal_function(self, kernel_name: str, general_kernel_name: str) -> str:
        """Generate code for normal (non-out, non-inplace) variant of an operation."""
        return self.templates.FunctionVariants.normal(kernel_name, general_kernel_name)

    # Helper methods for out function generation

    def _generate_input_extraction_from_array(self, root_name: str) -> str:
        """Generate code to extract input from array when no self argument."""
        code = self.templates.InputExtraction.array_start()

        # Single memory location violation check
        if self.op_categories.needs_single_mem_loc_check(root_name):
            code += self.templates.InputExtraction.single_mem_loc_check()

        # Dtype handling
        if self.op_categories.should_skip_dtype_check(root_name):
            code += self.templates.InputExtraction.dtype_promote()
        else:
            code += self.templates.InputExtraction.dtype_strict()

        # Device handling
        code += self.templates.InputExtraction.device_from_array()

        return code

    def _generate_dtype_validation(self, root_name: str) -> str:
        """Generate code for dtype validation based on operation type.

        Logic (matching original implementation):
        - If reduction op: add reduction dtype check (conditional, independent)
        - If compare op: add compare dtype check (conditional)
        - Else (not compare): add default dtype check

        Note: REDUCTION_OPS and COMPARE_OPS are mutually exclusive sets.
        """
        code = ""

        if root_name in self.op_categories.REDUCTION_OPS:
            code += self.templates.Validation.dtype_reduction()

        if root_name in self.op_categories.COMPARE_OPS:
            code += self.templates.Validation.dtype_compare()
        else:
            code += self.templates.Validation.dtype_default()

        return code
