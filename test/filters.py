from functools import wraps
from inspect import signature
from pathlib import Path
from unittest import SkipTest

import torch
from tools.codegen.parser import YamlParser
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torchgen.model import dispatch_keys

from test.utils import SUPPORTED_DTYPES


__all__ = [
    "custom_instantiate_device_type_tests",
]

_skipped_tests = {
    "test_multiple_devices",
    "test_pointwise_tag_coverage",  # not working
    "test_numpy_ref",
    "test_python_ref_meta",
    "test_python_ref",
    "test_python_ref_torch_fallback",
    "test_python_ref_executor",
    "test_errors_sparse",  # not yet supported
    "test_python_ref_errors",
    "test_out_integral_dtype",
    "test_complex_half_reference_testing",
    "test_non_standard_bool_values",
    "test_dtypes",  # Exception: Caused by sample input at index 10
    "test_promotes_int_to_float",
    "test_cow_input",  # not yet supported
    "test_conj_view",
    "test_neg_view",
    "test_neg_conj_view",
    "test_fake_crossref_backward_no_amp",
    "test_fake_crossref_backward_amp",
    "test_tags",
}

# Operators whose RBLN implementation doesn't support forward AD
_ops_without_forward_ad_support = {
    "nn.functional.linear",  # torch.compile path loses forward AD context
}

_ops_with_related_runtime_tests = [
    "clone",
    "contiguous",
    "repeat",
    "tile",
    "to",
    "empty",
    "empty_strided",
    "resize_",
]
_ops_with_related_view_tests = [
    "view",
    "view_as",
    "view_copy",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "reshape",
    "reshape_as",
    "flatten",
    "transpose",
    "permute",
    "chunk",
    "unsafe_chunk",
]

_ops_with_cpu_fallback_tests = [
    "sum",
    "index_select",
    "masked_select",
    "fill",
    "normal",
    "uniform",
    "isnan",
    "randn",
    "argmax",
    "isin",
    "cumsum",
    "all",
    "any",
    "cos",
    "sin",
    "tril",
    "triu",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "logical_and",
    "logical_or",
    "logical_xor",
    "arange",
    # "scatter",  #  error handling is complicated
    # "gather",  #  error handling is complicated
    "index_copy",
    "index_add",
    "clamp",
    "sgn",
    "fmod",
    "nan_to_num",
    "flip",
    "unfold",
    "sign",
]

# Ops whose public API names differ from yaml function names (e.g., nn.functional.scaled_dot_product_attention
# vs _scaled_dot_product_fused_attention_overrideable). Added explicitly for test discovery.
_ops_with_public_api_name_mismatch = [
    "nn.functional.scaled_dot_product_attention",
]

# Assume this file is in `test/filters.py`.
_source_dir = Path(__file__).resolve().parent.parent
_native_functions_path = _source_dir / "aten/src/ATen/native/native_functions.yaml"
_tags_path = _source_dir / "aten/src/ATen/native/tags.yaml"

# Get all the base function names which has `PrivateUse1` dispatch.
_original_dispatch_keys = dispatch_keys.copy()
_parsed_yaml = YamlParser.parse_native_functions(_native_functions_path, _tags_path)

# Restore dispatch_keys if there are possible changes during parse_native_functions
dispatch_keys.clear()
dispatch_keys.extend(_original_dispatch_keys)

if _parsed_yaml is None:
    raise RuntimeError(
        f"Failed to parse ATen YAML metadata from {_native_functions_path} and {_tags_path}. "
        "Please verify both files are present and valid."
    )

# Get all the base function names and also nn.functional variants
_native_functions_ops = []
_native_functions_nn_ops = []
for native_function in _parsed_yaml.native_functions:
    _native_functions_ops.append(native_function.root_name)
    # For ops with python_module=nn, also add nn.functional.{root_name}
    py_module = native_function.python_module
    if py_module:
        # python_module can be a string or an object with .name attribute
        module_name = py_module.name if hasattr(py_module, "name") else str(py_module)
        if module_name == "nn":
            _native_functions_nn_ops.append(f"nn.functional.{native_function.root_name}")

_native_functions_ops_debug = [
    "add",  # BinaryUfuncInfo
    "rsqrt",  # UnaryUfuncInfo
    "mean",  # ReductionOpInfo
    "max",  # OpInfo
    "mm",  # OpInfo
]

_root_names = set()
if torch.version.debug:
    _root_names.update(_native_functions_ops_debug)
    _root_names.add(_ops_with_related_runtime_tests[0])
    _root_names.add(_ops_with_related_view_tests[0])
    _root_names.add(_ops_with_cpu_fallback_tests[0])
else:
    _root_names.update(_native_functions_ops)
    _root_names.update(_native_functions_nn_ops)
    _root_names.update(_ops_with_related_runtime_tests)
    _root_names.update(_ops_with_related_view_tests)
    _root_names.update(_ops_with_cpu_fallback_tests)
    _root_names.update(_ops_with_public_api_name_mismatch)

_target_ops = set()
for root_name in _root_names:
    _target_ops.add(root_name)
    _target_ops.add(f"_refs.{root_name}")

# Ops that test CPU fallback with non-float16 dtypes (e.g., SDPA falls back when dtype != float16).
_ops_with_cpu_fallback_dtype_tests = {"nn.functional.scaled_dot_product_attention"}


def _skip_if_not_implemented(fn):
    sig = signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Specific test restriction
        if fn.__name__ in _skipped_tests:
            raise SkipTest(f"Skipping {fn.__name__}")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Get the op name if available for CPU fallback dtype test exception
        op_name = None
        for name, value in bound_args.arguments.items():
            if name == "op" and hasattr(value, "name"):
                op_name = value.name
                break

        # Skip forward AD tests for operators without support
        if fn.__name__ == "test_forward_ad" and op_name in _ops_without_forward_ad_support:
            raise SkipTest(f"Skipping {fn.__name__} for {op_name}: forward AD not supported on RBLN")

        for name, value in bound_args.arguments.items():
            # `op` restriction
            if name == "op" and hasattr(value, "name") and value.name not in _target_ops:
                raise SkipTest(f"Skipping {value.name} because it is not in `native_functions.yaml`")
            # `dtype` restriction - SDPA is an exception to test CPU fallback with various dtypes
            if name == "dtype" and value not in SUPPORTED_DTYPES:
                if op_name not in _ops_with_cpu_fallback_dtype_tests:
                    raise SkipTest(f"Skipping {fn.__name__} because it is not supported dtype")

        return fn(*args, **kwargs)

    return wrapper


def custom_instantiate_device_type_tests(generic_test_class, scope, *args, **kwargs):
    """Wrap instantiate_device_type_tests() to skip tests not implemented on RBLN."""
    for test_case in dir(generic_test_class):
        if test_case.startswith("test_"):
            setattr(generic_test_class, test_case, _skip_if_not_implemented(getattr(generic_test_class, test_case)))
    return instantiate_device_type_tests(generic_test_class, scope, *args, **kwargs)
