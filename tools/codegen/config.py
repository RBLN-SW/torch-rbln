"""
Operation categories configuration.

Defines categories of operations for code generation logic.

Responsibilities:
- Categorize operations by type
- Each category determines specific code generation logic

Categories:
- REDUCTION_OPS: sum, mean, etc. (different empty tensor handling)
- BROADCASTABLE_OPS: add, sub, mul, etc. (support broadcasting)
- COMPARE_OPS: eq, ne, gt, etc. (output dtype is bool)
- SINGLE_MEM_LOC_OUT_VIOLATION_OPS: cat, etc. (memory location validation needed)
- SKIP_DTYPE_CHECK_OPS: cat, etc. (skip dtype check)
- CONTIGUOUS_ON_DEVICE_OPS: bmm, mm, etc. (need contiguous preparation on device)

Empty tensor handling:
- Defined in EMPTY_TENSOR_HANDLING_MAP mapping table
- Single operations (mm, where, addmm, linear) are defined inline as sets
- Category sets (REDUCTION_OPS, BROADCASTABLE_OPS) are reused from above

Usage examples:
- GeneralFunctionGenerator: Determine empty tensor handling approach
- OpCodeGenerator: Determine dtype validation approach
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple  # noqa: UP035


class OpCategories:
    """Manages categorization of operations for code generation logic."""

    # TODO: need to make automatically fill 'Ops' if it figure out
    REDUCTION_OPS: Set[str] = {"sum", "mean"}
    UNARY_OPS: Set[str] = {"silu", "rsqrt", "neg", "ceil", "abs", "log", "floor", "trunc"}
    BROADCASTABLE_OPS: Set[str] = {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "ne",
        "eq",
        "gt",
        "ge",
        "lt",
        "le",
    }

    COMPARE_OPS: Set[str] = {"logical_not", "ne", "eq", "gt", "ge", "lt", "le"}
    # TODO: for cat only, violation if in_tensor(array) and out tensor are same data_ptr.
    #       need to check if other OPs need this check too
    SINGLE_MEM_LOC_OUT_VIOLATION_OPS: Set[str] = {"cat"}
    SKIP_DTYPE_CHECK_OPS: Set[str] = {"cat"}

    # Operations that need special kwargs filtering (non-standard 'out' parameter)
    # Maps root_name -> set of kwargs to exclude from filtering
    SPECIAL_KWARGS_FILTER_OPS: Dict[str, Set[str]] = {
        "max": {"out", "max", "max_values"},
        "min": {"out", "min", "min_indices"},
    }

    # Operations that need contiguous preparation on device (instead of default)
    CONTIGUOUS_ON_DEVICE_OPS: Set[str] = {"bmm", "mm"}

    # EmptyTensor template names that skip `tensor_args = extract_tensors(args)` before the
    # empty-tensor branch (those templates only read raw positional args, e.g. args[0]).
    # Keep in sync when adding EmptyTensor methods with the same pattern.
    EMPTY_TENSOR_TEMPLATES_WITHOUT_PRIOR_TENSOR_ARGS: FrozenSet[str] = frozenset({"where"})

    # Empty tensor handling mapping: (category_set, template_method_name)
    # Maps category Set -> template method name in EmptyTensor class
    # Order matters: checked in sequence, first match wins
    EMPTY_TENSOR_HANDLING_MAP: List[Tuple[Set[str], str]] = [  # noqa: UP035
        (REDUCTION_OPS, "reduction"),
        (BROADCASTABLE_OPS, "broadcastable"),
        ({"mm"}, "mm"),
        ({"where"}, "where"),
        ({"addmm"}, "addmm"),
        ({"linear"}, "linear"),
    ]

    # Dtype validation mapping: (category_set, template_method_name)
    # Maps category Set -> template method name in Validation class
    # Order matters: checked in sequence, first match wins
    DTYPE_VALIDATION_MAP: List[Tuple[Set[str], str]] = [  # noqa: UP035
        (REDUCTION_OPS, "dtype_reduction"),
        (COMPARE_OPS, "dtype_compare"),
    ]

    # Special out parameter names for operations that don't use standard 'out' kwarg
    # Maps kernel_name pattern -> (out_tensor_kwarg, out_values_kwarg)
    SPECIAL_OUT_PARAMS: Dict[str, Tuple[str, Optional[str]]] = {  # noqa: UP035
        "max_dim_max_rbln": ("max", "max_values"),
        "min_dim_min_rbln": ("min", "min_indices"),
    }

    @classmethod
    def get_out_param_names(cls, kernel_name: str) -> Tuple[str, Optional[str]]:
        """
        Get out parameter names for a kernel.
        Returns (out_tensor_kwarg, out_values_kwarg) tuple.
        For standard ops, returns ('out', None).
        """
        if kernel_name in cls.SPECIAL_OUT_PARAMS:
            return cls.SPECIAL_OUT_PARAMS[kernel_name]
        return ("out", None)

    @classmethod
    def needs_single_mem_loc_check(cls, root_name: str) -> bool:
        """Check if operation needs single memory location violation check."""
        return root_name in cls.SINGLE_MEM_LOC_OUT_VIOLATION_OPS

    @classmethod
    def should_skip_dtype_check(cls, root_name: str) -> bool:
        """Check if operation should skip dtype checking."""
        return root_name in cls.SKIP_DTYPE_CHECK_OPS

    @classmethod
    def get_kwargs_to_exclude(cls, root_name: str) -> Optional[Set[str]]:
        """
        Get kwargs to exclude for an operation.
        Returns set of kwargs to exclude, or None for default (only 'out').
        """
        return cls.SPECIAL_KWARGS_FILTER_OPS.get(root_name)

    @classmethod
    def needs_contiguous_on_device(cls, root_name: str) -> bool:
        """Check if operation needs contiguous preparation on device."""
        return root_name in cls.CONTIGUOUS_ON_DEVICE_OPS

    @classmethod
    def _get_template_method_from_map(cls, root_name: str, mapping: List[Tuple[Set[str], str]], default: str) -> str:
        """
        Get template method name from a mapping table.

        Searches through the mapping table in order and returns the first matching
        template method name. If no match is found, returns the default value.

        Args:
            root_name: Root name of the operation
            mapping: List of tuples (category_set, template_method_name)
            default: Default template method name if no match found

        Returns:
            Template method name
        """
        for category_set, template_method_name in mapping:
            if root_name in category_set:
                return template_method_name
        return default

    @classmethod
    def get_empty_tensor_template_method(cls, root_name: str) -> str:
        """
        Get empty tensor template method name for an operation.
        Returns template method name, or 'default' if no match found.
        """
        return cls._get_template_method_from_map(root_name, cls.EMPTY_TENSOR_HANDLING_MAP, "default")

    @classmethod
    def needs_tensor_args_extraction_before_empty_handling(cls, root_name: str) -> bool:
        """Whether to emit `tensor_args = extract_tensors(args)` before EmptyTensor handling."""
        return (
            cls.get_empty_tensor_template_method(root_name) not in cls.EMPTY_TENSOR_TEMPLATES_WITHOUT_PRIOR_TENSOR_ARGS
        )
