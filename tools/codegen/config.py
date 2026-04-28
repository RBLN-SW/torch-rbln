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

    # Operations whose Python-level aten_impl.impl registration is replaced by a
    # C++ dispatch shim installed at module-init time. Codegen emits
    # `_register_cpp_shim("aten::<op>", <kernel>)` instead of
    # `aten_impl.impl(...)`.
    # See torch_rbln/csrc/rbln/DispatchShim.cpp for shim behavior: cheap
    # pre-check in C++, cpu_fallback_rbln on fail, Python callback on pass.
    # Compared against the string form of the operator overload name
    # (e.g. "add.out"), not the root name.
    #
    # Only pointwise/unary/compare out-variants are shimmed: their pre-check
    # (dtype==fp16, not-all-scalar, storage_offset) matches the generic shim's
    # quick_fallback_check. Matmul-family, reductions, where, masked_fill etc.
    # stay on the Python path because their checks/preprocessing differ.
    CPP_SHIM_OPS: Set[str] = {
        # BROADCASTABLE binary
        "add.out",
        "sub.out",
        "mul.out",
        "div.out",
        "div.out_mode",
        "pow.Tensor_Scalar_out",
        "maximum.out",
        "minimum.out",
        # UNARY
        "silu.out",
        "rsqrt.out",
        "neg.out",
        "abs.out",
        "ceil.out",
        "clamp.out",
        "log.out",
        "floor.out",
        "sigmoid.out",
        # COMPARE (bool output — checks are on inputs, so fine)
        "logical_not.out",
        "ne.Tensor_out",
        "eq.Tensor_out",
        "gt.Tensor_out",
        "ge.Tensor_out",
        "lt.Tensor_out",
        "le.Tensor_out",
        "ne.Scalar_out",
        "eq.Scalar_out",
        "gt.Scalar_out",
        "ge.Scalar_out",
        "lt.Scalar_out",
        "le.Scalar_out",
        # WHERE: arg 0 is bool cond — skip from dtype check (see CPP_SHIM_SKIP_DTYPE_ARGS)
        "where.self",
        "where.self_out",
        # REDUCTION .out variants: quick_fallback_check is compatible
        # (fp16 + not-all-scalar + no-contig-offset). Python wrappers keep
        # the empty-tensor handle_empty_reduction branch — it runs before the
        # shim decision on cold call, and on warm call our pre-check passes
        # and we call into Python which handles empty correctly.
        "mean.out",
        "max.unary_out",
        "min.unary_out",
        # MAX/MIN full-reduction: registered as Normal variants (no overload)
        # with single Tensor return. Still pointwise-compatible pre-check.
        "max",
        "min",
        # MATMUL family. Earlier benches showed a small device-path overhead
        # (~2% in pybind-heavy hot loops) which is now well below noise after
        # the warm-cache path skips Python entirely on warm hits. Coverage gain
        # — matmul is registered C++-side instead of Python-side — is the main
        # reason; perf is parity within noise on llama_1b/llama_8b 1L bench.
        "mm.out",
        "bmm.out",
        "addmm.out",
        "linear",
    }

    # Per-op overrides of the dispatch-shim dtype check: positional arg indices
    # whose dtype should NOT be compared to float16. The shim's own check still
    # applies to other args (storage offset, all-scalar, dtype).
    # Keyed by the same overload name used in CPP_SHIM_OPS.
    CPP_SHIM_SKIP_DTYPE_ARGS: Dict[str, List[int]] = {
        "where.self": [0],  # cond (bool)
        "where.self_out": [0],  # cond (bool)
    }

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
