"""Operation categories configuration.

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

Empty tensor handling:
- Defined in EMPTY_TENSOR_HANDLING_MAP mapping table
- Single operations (mm, where, addmm, linear) are defined inline as sets
- Category sets (REDUCTION_OPS, BROADCASTABLE_OPS) are reused from above

Usage examples:
- GeneralFunctionGenerator: Determine empty tensor handling approach
- OpCodeGenerator: Determine dtype validation approach
"""

from typing import ClassVar


class OpCategories:
    """Manages categorization of operations for code generation logic."""

    # TODO: need to make automatically fill 'Ops' if it figure out
    REDUCTION_OPS: ClassVar[set[str]] = {"sum", "mean"}
    BROADCASTABLE_OPS: ClassVar[set[str]] = {
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

    COMPARE_OPS: ClassVar[set[str]] = {"logical_not", "ne", "eq", "gt", "ge", "lt", "le"}
    # TODO: for cat only, violation if in_tensor(array) and out tensor are same
    #       data_ptr. need to check if other OPs need this check too
    SINGLE_MEM_LOC_OUT_VIOLATION_OPS: ClassVar[set[str]] = {"cat"}
    SKIP_DTYPE_CHECK_OPS: ClassVar[set[str]] = {"cat"}

    # Operations that need special kwargs filtering (non-standard 'out' parameter)
    # Maps root_name -> set of kwargs to exclude from filtering
    SPECIAL_KWARGS_FILTER_OPS: ClassVar[dict[str, set[str]]] = {
        "max": {"out", "max", "max_values"},
        "min": {"out", "min", "min_indices"},
    }

    # Operations whose Python-level aten_impl.impl registration is replaced by
    # a C++ dispatch shim installed at module-init time. Codegen emits
    # `_register_cpp_shim("aten::<op>", <kernel>)` instead of
    # `aten_impl.impl(...)`. See torch_rbln/csrc/rbln/DispatchShim.cpp for shim
    # behavior: cheap pre-check in C++, cpu_fallback_rbln on fail, Python
    # callback on pass. Compared against the operator overload name
    # (e.g. "add.out"), not the root name.
    #
    # Sub-set categories below partition the shim ops by the *shape* of their
    # pre-check and dispatch handling. Each sub-set maps 1:1 to a code path in
    # DispatchShim.cpp (e.g. SHIM_WHERE → skip_dtype_args for the bool cond,
    # SHIM_MATMUL → contiguous-on-device prep). Add an op by appending to the
    # right sub-set, not by editing the flat union below.
    SHIM_BINARY_POINTWISE: ClassVar[set[str]] = {
        "add.out",
        "sub.out",
        "mul.out",
        "div.out",
        "div.out_mode",
        "pow.Tensor_Scalar_out",
        "maximum.out",
        "minimum.out",
    }
    SHIM_UNARY_POINTWISE: ClassVar[set[str]] = {
        "silu.out",
        "rsqrt.out",
        "neg.out",
        "abs.out",
        "ceil.out",
        "clamp.out",
        "log.out",
        "floor.out",
        "sigmoid.out",
    }
    # Bool output — checks are on inputs, so pointwise pre-check still applies.
    SHIM_COMPARE: ClassVar[set[str]] = {
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
    }
    # Cond arg is bool — see PREDICATE_ARGS for the dtype-check override.
    SHIM_WHERE: ClassVar[set[str]] = {
        "where.self",
        "where.self_out",
    }
    # Reduction .out variants: quick_fallback_check is compatible
    # (fp16 + not-all-scalar + no-contig-offset). Python wrappers keep the empty-tensor
    # handle_empty_reduction branch — it runs before the shim decision on cold call; on
    # warm call our pre-check passes and we call into Python which handles empty
    # correctly.
    SHIM_REDUCTION: ClassVar[set[str]] = {
        "mean.out",
        "max.unary_out",
        "min.unary_out",
    }
    # Max/min full-reduction: Normal-variant (no overload) with single
    # Tensor return. Pointwise-compatible pre-check still applies.
    SHIM_REDUCTION_FULL: ClassVar[set[str]] = {
        "max",
        "min",
    }
    # Softmax .out: shape-preserving, with an int dim and a bool flag. The
    # pointwise pre-check applies unchanged (fp16, not-all-scalar, no contig
    # offset); the scalars ride in the warm-cache key like any other shim op.
    SHIM_SOFTMAX: ClassVar[set[str]] = {
        "_softmax.out",
    }
    # Matmul family. Earlier benches showed a small device-path overhead
    # (~2% in pybind-heavy hot loops) which is now well below noise after
    # the warm-cache path skips Python entirely on warm hits. Coverage gain
    # — matmul is registered C++-side instead of Python-side — is the main
    # reason; perf is parity within noise on llama_1b/llama_8b 1L bench.
    SHIM_MATMUL: ClassVar[set[str]] = {
        "mm.out",
        "bmm.out",
        "addmm.out",
        "linear",
    }

    # Flat union, derived from the sub-sets above. Codegen consumes this set.
    CPP_SHIM_OPS: ClassVar[set[str]] = (
        SHIM_BINARY_POINTWISE
        | SHIM_UNARY_POINTWISE
        | SHIM_COMPARE
        | SHIM_WHERE
        | SHIM_REDUCTION
        | SHIM_REDUCTION_FULL
        | SHIM_MATMUL
        | SHIM_SOFTMAX
    )

    # Shim ops whose unaligned device inputs the compiler pads itself, so the Python
    # path skips the 64-element alignment fallback for them.
    ALIGN_EXEMPT_OPS: ClassVar[set[str]] = {"add", "sub", "mul", "div", "neg", "where"} | {
        name.split(".")[0] for name in SHIM_MATMUL
    }

    # Broadcastable shim ops whose size-1 last-dim operand the device broadcasts, so
    # the alignment check reads the output's last dim. pow is left out: its
    # tensor-tensor overload takes the CPU fallback.
    ALIGN_CHECK_OUTPUT_OPS: ClassVar[set[str]] = (
        (BROADCASTABLE_OPS & {name.split(".")[0] for name in CPP_SHIM_OPS}) - {"pow"} - ALIGN_EXEMPT_OPS
    )

    # Positional predicate args (a bool cond or mask) per root op. A predicate is
    # typed independently of the data operands, so both dtype gates exempt it:
    # the C++ shim pre-check (``_register_cpp_shim(..., skip_dtype_args=)``) and
    # the Python gate (``get_fallback_reason(..., skip_dtype_args=)``). One
    # definition keeps the two gates from drifting apart. The other checks
    # (storage offset, all-scalar) still apply to the predicate.
    PREDICATE_ARGS: ClassVar[dict[str, list[int]]] = {
        "where": [0],  # cond (bool)
        "masked_fill": [1],  # mask (bool)
    }

    # In-place variants that offer ``self`` as the output to bind. The dispatch
    # helper binds it only when ``self`` reaches the compiler unchanged and no
    # other input overlaps it (``can_use_out_tensor_directly``); otherwise the
    # wrapper copies the result back.
    SELF_REUSE_INPLACE_OPS: frozenset[str] = frozenset({"masked_fill"})

    @classmethod
    def get_predicate_args(cls, root_name: str) -> list[int]:
        return cls.PREDICATE_ARGS.get(root_name, [])

    @classmethod
    def reuses_self_inplace(cls, root_name: str) -> bool:
        return root_name in cls.SELF_REUSE_INPLACE_OPS

    # EmptyTensor template names that skip `tensor_args = extract_tensors(args)` before
    # the empty-tensor branch (those templates only read raw positional args, e.g.
    # args[0]). Keep in sync when adding EmptyTensor methods with the same pattern.
    EMPTY_TENSOR_TEMPLATES_WITHOUT_PRIOR_TENSOR_ARGS: frozenset[str] = frozenset({"where", "masked_fill"})

    # TODO: remove with ``contiguous_for_scalar_and_strided`` (see its docstring).
    # Ops whose compile crashes on the pinned compiler when a 0-dim operand meets a
    # permuted one. The generated handler calls that helper before the dispatch gate.
    SCALAR_STRIDED_CONTIG_OPS: frozenset[str] = frozenset({"where"})

    @classmethod
    def needs_scalar_strided_contig(cls, root_name: str) -> bool:
        return root_name in cls.SCALAR_STRIDED_CONTIG_OPS

    # Empty tensor handling mapping: (category_set, template_method_name)
    # Maps category Set -> template method name in EmptyTensor class
    # Order matters: checked in sequence, first match wins
    EMPTY_TENSOR_HANDLING_MAP: ClassVar[list[tuple[set[str], str]]] = [
        (REDUCTION_OPS, "reduction"),
        (BROADCASTABLE_OPS, "broadcastable"),
        ({"mm"}, "mm"),
        ({"where"}, "where"),
        ({"masked_fill"}, "masked_fill"),
        ({"addmm"}, "addmm"),
        ({"linear"}, "linear"),
    ]

    # Special out parameter names for operations that don't use standard 'out' kwarg
    # Maps kernel_name pattern -> (out_tensor_kwarg, out_values_kwarg)
    SPECIAL_OUT_PARAMS: ClassVar[dict[str, tuple[str, str | None]]] = {
        "max_dim_max_rbln": ("max", "max_values"),
        "min_dim_min_rbln": ("min", "min_indices"),
    }

    @classmethod
    def get_out_param_names(cls, kernel_name: str) -> tuple[str, str | None]:
        """Get out parameter names for a kernel.
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
    def align_check(cls, root_name: str) -> str:
        """What the alignment fallback checks: "none", "output" or "inputs"."""
        if root_name in cls.ALIGN_EXEMPT_OPS:
            return "none"
        if root_name in cls.ALIGN_CHECK_OUTPUT_OPS:
            return "output"
        return "inputs"

    @classmethod
    def should_skip_dtype_check(cls, root_name: str) -> bool:
        """Check if operation should skip dtype checking."""
        return root_name in cls.SKIP_DTYPE_CHECK_OPS

    @classmethod
    def get_kwargs_to_exclude(cls, root_name: str) -> set[str] | None:
        """Get kwargs to exclude for an operation.
        Returns set of kwargs to exclude, or None for default (only 'out').
        """
        return cls.SPECIAL_KWARGS_FILTER_OPS.get(root_name)

    @classmethod
    def _get_template_method_from_map(cls, root_name: str, mapping: list[tuple[set[str], str]], default: str) -> str:
        """Get template method name from a mapping table.

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
        """Get empty tensor template method name for an operation.
        Returns template method name, or 'default' if no match found.
        """
        return cls._get_template_method_from_map(root_name, cls.EMPTY_TENSOR_HANDLING_MAP, "default")

    @classmethod
    def needs_tensor_args_extraction_before_empty_handling(cls, root_name: str) -> bool:
        """Whether to emit `tensor_args = extract_tensors(args)` before
        EmptyTensor handling.
        """
        return (
            cls.get_empty_tensor_template_method(root_name) not in cls.EMPTY_TENSOR_TEMPLATES_WITHOUT_PRIOR_TENSOR_ARGS
        )
