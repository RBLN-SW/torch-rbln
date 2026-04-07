"""
Main code generator.

Orchestrates the generation of register_ops.py from parsed YAML data.

Code generation flow:
1. Generate header (imports, setup)
2. For each native function:
   a. Analyze function schema (FunctionAnalyzer)
   b. Determine operation type (custom/out/inplace/normal)
   c. Generate general function (GeneralFunctionGenerator) - once per root_name
   d. Generate variant function (OpCodeGenerator) - out/inplace/normal
   e. Add registration code
3. Append all registration code at the end of file

The main flow can be found in this file, while detailed implementations are separated into modules.
"""

from typing import Any, Dict, Tuple  # noqa: UP035

from torchgen.model import DispatchKey, SchemaKind  # type: ignore[import-untyped]

from .analyzer import FunctionAnalyzer
from .config import OpCategories
from .generators.general import GeneralFunctionGenerator
from .generators.variants import OpCodeGenerator


class CodeGenerator:
    """
    Main code generator that orchestrates the generation of register_ops.py.

    Manages the entire code generation process with the following steps:

    1. Initialization: Create required components
    2. Header generation: Import statements and basic setup
    3. Process each operation:
       - Custom Kernel: Only registration
       - Out Function: General function + Out wrapper
       - Inplace: General function + Inplace wrapper
       - Normal: General function + Normal wrapper
    4. Collect and append registration code
    """

    def __init__(self) -> None:
        """Initialize code generator with all required components."""
        self.op_categories = OpCategories()
        self.function_analyzer = FunctionAnalyzer()
        self.general_function_generator = GeneralFunctionGenerator(self.op_categories)
        self.op_code_generator = OpCodeGenerator(self.op_categories, self.function_analyzer)

    def generate_header(self) -> str:
        """Generate the header imports and setup code."""
        return """
from torch_rbln._internal.env_utils import *
from torch_rbln._internal.ops_utils import *
from torch_rbln._internal.profiling import wrap_registered_dispatch_functions
from torch_rbln._internal.register_custom_ops import *
from torch_rbln._internal.register_backward_ops import *
from torch_rbln._internal.kernels.custom_transpose import *
from torch_rbln._internal.kernels.sdpa import *

aten_impl = torch.library.Library('aten', 'IMPL')
DEBUG_MODE = torch.version.debug
"""

    def _get_or_generate_general_function(
        self,
        root_name: str,
        python_module: str,
        op_namespace: str,
        general_kernel_name_table: Dict[str, str],
    ) -> Tuple[str, str]:
        """
        Get or generate general function code for a root operation.

        If the general function has already been generated for this root_name,
        returns the existing kernel name and empty string.
        Otherwise, generates the general function code and returns the kernel name and code.

        Args:
            root_name: Root name of the operation
            python_module: Python module name (e.g., "nn")
            op_namespace: Operation namespace
            general_kernel_name_table: Dictionary mapping root_name to general kernel name

        Returns:
            Tuple of (general_kernel_name, function_code)
        """
        if root_name in general_kernel_name_table:
            return general_kernel_name_table[root_name], ""

        general_kernel_name = f"{root_name}_rbln"
        general_kernel_name_table[root_name] = general_kernel_name
        function_code = self.general_function_generator.generate(
            general_kernel_name, root_name, python_module, op_namespace
        )
        return general_kernel_name, function_code

    def generate_python_registration(self, parsed_yaml: Any) -> str:
        """
        Generate Python registration code for all operations.

        Overall flow:
        1. Generate header
        2. Iterate over each native function:
           - Extract function information (name, type, arguments, etc.)
           - Analyze function schema (self position, device kwarg, etc.)
           - Generate code based on operation type:
             * Custom Kernel: Registration only
             * Out Function: General function (once) + Out wrapper
             * Inplace: General function (once) + Inplace wrapper
             * Normal: General function (once) + Normal wrapper
           - Collect registration code
        3. Append all registration code at the end of file

        Args:
            parsed_yaml: Parsed YAML data from YamlParser

        Returns:
            Complete Python code string for register_ops.py
        """
        # Step 1: Generate header (imports, setup)
        code_blocks = self.generate_header()
        general_kernel_name_table: Dict[str, str] = {}
        operator_register_code = ""

        # Step 2: Iterate over all native functions
        for native_function in parsed_yaml.native_functions:
            # Extract function information
            python_module = native_function.python_module
            function_schema = native_function.func
            operator_name = function_schema.name
            root_name = native_function.root_name  # simple name of op (mm.out->mm, sum.intList_out->sum)
            schema_kind = function_schema.kind()
            is_out = schema_kind == SchemaKind.out
            is_inplace = schema_kind == SchemaKind.inplace
            arguments = function_schema.arguments
            function_code = ""  # init generated code body
            op_namespace = native_function.namespace

            # Analyze function schema
            is_custom_kernel = "use_custom_kernel_rbln" in native_function.tags
            kernel_name = None
            self_arg_pos = self.function_analyzer.get_self_arg_position_index(arguments)
            has_device_kwarg = self.function_analyzer.check_has_kwarg(arguments, "device")

            # Find the kernel name from backend indices
            backend_index = parsed_yaml.backend_indices.get(DispatchKey.PrivateUse1, None)
            if backend_index:
                metadata = backend_index.index.get(operator_name)
                if metadata:
                    kernel_name = metadata.kernel

            # If kernel name is not found, skip this function
            if not kernel_name:
                print(f"Skipping {operator_name}, no kernel name found.")
                continue

            # Step 3: Generate code based on operation type
            if is_custom_kernel:
                # Custom kernel - just register, no code generation needed
                # Detailed implementation: No need to reference generators/ directory
                operator_register_code += f"""
aten_impl.impl("{operator_name}", {kernel_name}, "PrivateUse1")"""

            elif is_out:
                # Out Function: Generate general function + Out wrapper
                general_kernel_name, function_code = self._get_or_generate_general_function(
                    root_name, python_module, op_namespace, general_kernel_name_table
                )

                # Generate Out function wrapper: Input validation + general function call + output processing
                function_code += self.op_code_generator.generate_out_function(
                    kernel_name, root_name, general_kernel_name, self_arg_pos, has_device_kwarg
                )
                operator_register_code += f"""
aten_impl.impl("{operator_name}", {kernel_name}, "PrivateUse1")"""

            elif is_inplace:
                # Inplace Function: Generate general function + Inplace wrapper
                general_kernel_name, function_code = self._get_or_generate_general_function(
                    root_name, python_module, op_namespace, general_kernel_name_table
                )

                # Generate Inplace function wrapper: General function call + copy_
                function_code += self.op_code_generator.generate_inplace_function(kernel_name, general_kernel_name)
                operator_register_code += f"""
aten_impl.impl("{operator_name}", {kernel_name}, "PrivateUse1")"""

            else:
                # Normal Function: Generate general function + Normal wrapper
                general_kernel_name, function_code = self._get_or_generate_general_function(
                    root_name, python_module, op_namespace, general_kernel_name_table
                )

                # Generate Normal function wrapper: General function call (out=None)
                normal_kernel_name = f"{general_kernel_name}_normal"
                function_code += self.op_code_generator.generate_normal_function(
                    normal_kernel_name, general_kernel_name
                )
                operator_register_code += f"""
aten_impl.impl("{operator_name}", {normal_kernel_name}, "PrivateUse1")"""

            code_blocks += function_code

        # Step 4: Place code to register operator at the bottom of register_ops.py
        code_blocks += """
wrap_registered_dispatch_functions(globals(), module_name=__name__)
"""
        code_blocks += operator_register_code
        return code_blocks
