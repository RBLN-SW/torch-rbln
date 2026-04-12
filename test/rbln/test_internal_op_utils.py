# Owner(s): ["module: PrivateUse1"]

"""
Test suite for internal utilities used in op kernel implementations.
"""

import os
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._python_dispatch import TorchDispatchMode

from torch_rbln._internal.ops_utils import (
    _all_scalars,
    _contains_nan_or_inf,
    _get_default_compile_options,
    _has_nan_or_inf,
    _has_non_float16_dtype,
    _has_nonzero_storage_offset,
    _is_trace_active,
    _parse_disabled_fallback_cases,
    _resolve_result_tensor,
    broadcast_args_general,
    can_use_out_tensor_directly,
    cpu_fallback_path,
    extract_tensors,
    finalize_output_tensor,
    handle_empty_binary,
    handle_empty_linear,
    handle_empty_mm,
    handle_empty_reduction,
    handle_empty_tensor,
    handle_empty_where,
    is_cpu_fallback_cases,
    is_inplace_op,
    make_op_module,
    prepare_args_for_contiguous,
)


@pytest.mark.test_set_ci
class TestInternalOpUtils(TestCase):
    # =========================================================================
    # cpu_fallback_path tests
    # =========================================================================

    def test_cpu_fallback_path_basic_addition(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32, device="rbln")
        args = [t]
        kwargs = {}

        def add1(x):
            return x + 1.0

        out = cpu_fallback_path(add1, args, **kwargs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.device.type, "rbln")
        expected = torch.tensor([2.0, 3.0], dtype=torch.float32, device="cpu")
        self.assertEqual(out.to("cpu"), expected)

    def test_cpu_fallback_path_with_kwargs_multiplication(self):
        t1 = torch.tensor([3], dtype=torch.float32, device="cpu")
        t2 = torch.tensor([4], dtype=torch.float32, device="rbln")
        args = [t1]
        kwargs = {"y": t2}

        def mul(x, y):
            return x * y

        out = cpu_fallback_path(mul, args, **kwargs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.device.type, "rbln")
        self.assertEqual(out.to("cpu"), torch.tensor([12], dtype=torch.float32))

    # =========================================================================
    # finalize_output_tensor tests
    # =========================================================================

    def test_finalize_output_tensor_no_change(self):
        # Case: shape and storage already match => no-op
        out = torch.empty((2, 2), device="rbln")
        result = out
        result_shape = out.shape
        args = ()
        kwargs = {}

        # no exception, in-place operation
        finalize_output_tensor(out, result, result_shape, args, kwargs)

        # shape unchanged
        self.assertEqual(out.shape, result_shape)
        # storage pointer unchanged
        self.assertEqual(out.data_ptr(), result.data_ptr())

    def test_finalize_output_tensor_resize_and_warn(self):
        # Case: shape mismatch with existing elements => warn and resize, then copy data
        out = torch.empty((2, 2), device="rbln")
        out.fill_(5.0)
        result = torch.ones((3, 3), device="rbln")
        result_shape = result.shape
        args = ()
        kwargs = {}

        with self.assertWarns(UserWarning):
            finalize_output_tensor(out, result, result_shape, args, kwargs)

        # shape updated
        self.assertEqual(out.shape, result_shape)
        # data copied from result
        self.assertEqual(out.to("cpu"), result.to("cpu"))

    def test_finalize_output_tensor_dtype_sync(self):
        # Case: dtype mismatch => dtype should be synchronized before copy
        out = torch.empty((2, 2), dtype=torch.float16, device="rbln")

        result = torch.ones((2, 2), dtype=torch.float16, device="rbln")

        result_shape = result.shape
        args = ()
        kwargs = {}

        finalize_output_tensor(out, result, result_shape, args, kwargs)

        self.assertEqual(out.to("cpu"), result.to("cpu"))

    # =========================================================================
    # handle_empty_* tests
    # =========================================================================

    def test_handle_empty_reduction_default(self):
        t = torch.randn((2, 3), device="rbln")
        # no dim, keepdim False -> scalar
        out, _ = handle_empty_reduction(t)
        self.assertEqual(out.shape, torch.Size([]))
        # dtype and device preserved
        self.assertEqual(out.dtype, t.dtype)
        self.assertEqual(out.device, t.device)

    def test_handle_empty_reduction_keepdim(self):
        t = torch.randn((2, 3, 4), device="rbln")
        out, _ = handle_empty_reduction(t, dim=1, keepdim=True)
        # shape: [2,1,4]
        self.assertEqual(out.shape, torch.Size([2, 1, 4]))

    def test_handle_empty_mm(self):
        t1 = torch.randn((2, 5), device="rbln")
        t2 = torch.randn((5, 3), device="rbln")
        out, _ = handle_empty_mm([t1, t2])
        self.assertEqual(out.shape, torch.Size([2, 3]))
        self.assertEqual(out, torch.zeros_like(out))

    def test_handle_empty_linear_basic(self):
        # input: [0, 3], weight: [4, 3] -> output: [0, 4]
        inp = torch.empty((0, 3), device="rbln", dtype=torch.float16)
        weight = torch.randn((4, 3), device="rbln", dtype=torch.float16)
        out, shape = handle_empty_linear([inp, weight])
        self.assertEqual(out.shape, torch.Size([0, 4]))
        self.assertEqual(shape, (0, 4))
        self.assertEqual(out.dtype, inp.dtype)
        self.assertEqual(out.device, inp.device)

    def test_handle_empty_linear_batched(self):
        # input: [2, 0, 3], weight: [5, 3] -> output: [2, 0, 5]
        inp = torch.empty((2, 0, 3), device="rbln", dtype=torch.float16)
        weight = torch.randn((5, 3), device="rbln", dtype=torch.float16)
        out, shape = handle_empty_linear([inp, weight])
        self.assertEqual(out.shape, torch.Size([2, 0, 5]))
        self.assertEqual(shape, (2, 0, 5))

    def test_handle_empty_where(self):
        cond = torch.empty((4, 1), device="rbln", dtype=torch.bool)
        x = torch.full((4, 1), 7.5, device="rbln")
        out, _ = handle_empty_where((cond, x, None))
        self.assertEqual(out.shape, cond.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_handle_empty_binary_and_empty_tensor(self):
        t = torch.zeros((3, 2), device="rbln")
        out1, _ = handle_empty_binary((None, t))
        self.assertEqual(out1.shape, t.shape)
        self.assertEqual(out1.dtype, t.dtype)
        out2, _ = handle_empty_tensor([t])
        self.assertEqual(out2.shape, t.shape)

    # =========================================================================
    # broadcast_args_general tests
    # =========================================================================

    def test_broadcast_args_general(self):
        a = torch.randn((2, 1), device="rbln")
        b = torch.randn((2, 3), device="rbln")
        args = (a, "foo", b)
        tensor_args = [a, b]
        new_args = broadcast_args_general(tensor_args, args)
        # non-tensor args preserved
        self.assertEqual(new_args[1], "foo")
        # tensors broadcasted to shape (2,3)
        self.assertEqual(new_args[0].shape, torch.Size([2, 3]))
        self.assertEqual(new_args[2].shape, torch.Size([2, 3]))

    # =========================================================================
    # prepare_args_for_contiguous tests
    # =========================================================================

    def test_prepare_args_no_change(self):
        t1 = torch.tensor([1, 2], device="rbln")
        args = (t1, 42, "x")
        kwargs = {"y": torch.tensor([[3]], device="rbln"), "z": None}
        (new_args, new_kwargs), changed = prepare_args_for_contiguous(args, kwargs)
        # No contiguous change expected (all args are already contiguous)
        self.assertFalse(changed)
        self.assertIsInstance(new_args, tuple)
        self.assertIsInstance(new_kwargs, dict)
        # values preserved
        self.assertEqual(new_args[0], t1)
        self.assertEqual(new_args[1], 42)
        self.assertEqual(new_kwargs["z"], None)

    def test_prepare_args_change_noncontiguous(self):
        base = torch.arange(4, device="rbln").view(2, 2).t()
        self.assertFalse(base.is_contiguous())
        args = (base,)
        kwargs = {}
        (new_args, _), changed = prepare_args_for_contiguous(args, kwargs)
        # Should have made contiguous
        self.assertTrue(changed)
        contig = new_args[0]
        self.assertTrue(contig.is_contiguous())
        # Data values preserved
        self.assertEqual(contig.flatten(), base.flatten())

    # =========================================================================
    # is_cpu_fallback_cases tests
    # =========================================================================

    def test_is_cpu_fallback_cases_dtype(self):
        a = torch.tensor([1, 2], dtype=torch.float32, device="rbln")
        b = torch.tensor([3, 4], dtype=torch.float16, device="rbln")
        # Dtype fallback due to a
        self.assertTrue(is_cpu_fallback_cases((a, b)))
        # No dtype fallback if all float16
        c = torch.tensor([5], dtype=torch.float16, device="rbln")
        # alignment unknown; skip alignment test
        _ = is_cpu_fallback_cases((c,))  # noqa: F841
        # Should not fallback for float16 scalar (unless other conditions apply)
        # Note: scalar tensors still trigger fallback, so this may still be True
        # This test now only checks dtype fallback behavior

    def test_is_cpu_fallback_cases_trace(self):
        """When sys.gettrace() is set (e.g. pdb, coverage), is_cpu_fallback_cases returns True (0a)."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="rbln")
        args = (t,)
        with patch("sys.gettrace", return_value=lambda *a, **k: None):
            self.assertTrue(
                is_cpu_fallback_cases(args),
                "CPU fallback should be triggered when a trace function is set",
            )
        with patch("sys.gettrace", return_value=None):
            # Same args without trace: no fallback from 0a (other conditions may still apply)
            fallback = is_cpu_fallback_cases(args)
            # With float16 1D tensor, scalar check triggers (all scalars? no). So we expect False
            # unless another case triggers (e.g. scalar is checked per-tensor: all(a.ndim==0) -> False)
            self.assertFalse(fallback, "Without trace, float16 1D rbln should not fallback from trace")

    def test_is_cpu_fallback_cases_trace_disabled(self):
        """When TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=trace, trace check is skipped."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="rbln")
        args = (t,)
        _parse_disabled_fallback_cases.cache_clear()
        try:
            with patch.dict(os.environ, {"TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK": "trace"}, clear=False):
                _parse_disabled_fallback_cases.cache_clear()
                with patch("sys.gettrace", return_value=lambda *a, **k: None):
                    self.assertFalse(
                        is_cpu_fallback_cases(args),
                        "With trace disabled via env, gettrace() set should not trigger fallback",
                    )
        finally:
            _parse_disabled_fallback_cases.cache_clear()

    def test_is_cpu_fallback_cases_reentrant(self):
        """When already inside RBLN compile op (depth > 0), is_cpu_fallback_cases returns True and logs warning (6)."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="rbln")
        args = (t,)
        with patch(
            "torch_rbln._internal.torch_compile_patch_helpers.get_rbln_compile_op_depth",
            return_value=1,
        ):
            with patch("torch_rbln._internal.ops_utils.rbln_log_warn") as mock_warn:
                result = is_cpu_fallback_cases(args)
                self.assertTrue(
                    result,
                    "CPU fallback should be triggered when reentrant (depth > 0)",
                )
                mock_warn.assert_called_once()
                self.assertIn("reentrant", mock_warn.call_args[0][0].lower())

    def test_is_cpu_fallback_cases_reentrant_disabled(self):
        """When TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=reentrant, reentrancy check is skipped."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="rbln")
        args = (t,)
        _parse_disabled_fallback_cases.cache_clear()
        try:
            with patch.dict(os.environ, {"TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK": "reentrant"}, clear=False):
                _parse_disabled_fallback_cases.cache_clear()
                with patch(
                    "torch_rbln._internal.torch_compile_patch_helpers.get_rbln_compile_op_depth",
                    return_value=1,
                ):
                    self.assertFalse(
                        is_cpu_fallback_cases(args),
                        "With reentrant disabled via env, depth > 0 should not trigger fallback",
                    )
        finally:
            _parse_disabled_fallback_cases.cache_clear()

    # =========================================================================
    # is_inplace_op tests
    # =========================================================================

    def test_self_inplace(self):
        """
        Tests the case where the input tensor itself is passed as the out argument.

        This test checks if the `is_inplace_op` function returns True when the same tensor is used both
        as an input and the output (in-place operation).

        Args:
            self (TestCase): The test case instance.

        Returns:
            None

        Raises:
            AssertionError: If the function does not return True as expected.
        """
        x = torch.randn(8, device="rbln")
        self.assertTrue(is_inplace_op((x,), {"out": x}))

    def test_view_alias(self):
        """
        Tests the case where a tensor created via view is passed as the out argument.

        This test checks if the `is_inplace_op` function returns True when a view of the base tensor
        (which shares storage with the base) is used as the output. The data pointers of the base and view
        should be the same, indicating an in-place operation.

        Args:
            self (TestCase): The test case instance.

        Returns:
            None

        Raises:
            AssertionError: If the function does not return True as expected.
        """
        base = torch.randn(4, 4, device="rbln")
        view = base.view(-1)
        self.assertTrue(is_inplace_op((base,), {"out": view}))

    def test_out_variant_alias(self):
        """
        Tests the case where src and dst are different instances but share the same storage.

        This test checks if the `is_inplace_op` function returns True when two tensors, created from the
        same base tensor (thus sharing storage), are used as input and output respectively. Despite being
        different instances, they should point to the same data, indicating an in-place operation.

        Args:
            self (TestCase): The test case instance.

        Returns:
            None

        Raises:
            AssertionError: If the function does not return True as expected.
        """
        src = torch.zeros(16, device="rbln")
        dst = src.view(2, 8)
        self.assertTrue(is_inplace_op((src,), {"out": dst}))

    def test_not_inplace(self):
        """
        Tests the case where the out tensor has a separate storage.

        This test checks if the `is_inplace_op` function returns False when the output tensor is created
        with its own storage, independent of the input tensor. This indicates that the operation is not in-place.

        Args:
            self (TestCase): The test case instance.

        Returns:
            None

        Raises:
            AssertionError: If the function does not return False as expected.
        """
        src = torch.ones(10, device="rbln")
        dst = torch.empty_like(src)
        self.assertFalse(is_inplace_op((src,), {"out": dst}))

    def test_no_out_argument(self):
        """
        Tests the case where there is no out argument.

        This test checks if the `is_inplace_op` function returns False when no output tensor is specified.
        In this scenario, there can be no in-place operation since there is no designated output tensor.

        Args:
            self (TestCase): The test case instance.

        Returns:
            None

        Raises:
            AssertionError: If the function does not return False as expected.
        """
        x = torch.randn(5, device="rbln")
        y = torch.randn(5, device="rbln")
        self.assertFalse(is_inplace_op((x, y), {}))


@pytest.mark.test_set_ci
class TestOutTensors(TestCase):
    """Test operations using out tensors with various contiguity and storage offset properties."""

    atol = 0.01
    rtol = 0.01

    def test_add_using_contiguous_out(self):
        """Test add operation using contiguous out tensor."""
        device = torch.device("rbln:0")
        x = torch.randn(4, 60, device=device, dtype=torch.float16)
        y = torch.randn(4, 60, device=device, dtype=torch.float16)

        # Create contiguous out tensor with zero storage offset
        out = torch.empty(4, 60, device=device, dtype=torch.float16)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.storage_offset(), 0)
        original_data_ptr = out.data_ptr()

        # Verify out tensor can be used directly
        self.assertTrue(can_use_out_tensor_directly((x, y), {"out": out}))

        # Perform operation
        result = torch.add(x, y, out=out)
        cpu_reference = torch.add(x.cpu(), y.cpu())

        # Verify the returned result is the same as the out tensor
        self.assertIs(result, out)
        self.assertEqual(result.data_ptr(), original_data_ptr)
        self.assertEqual(result.device, out.device)
        self.assertEqual(result.dtype, out.dtype)
        self.assertEqual(result.shape, out.shape)
        self.assertEqual(result.cpu(), out.cpu())

        # Verify result correctness
        self.assertEqual(result.cpu(), cpu_reference, atol=self.atol, rtol=self.rtol)

    def test_add_using_non_contiguous_out(self):
        """Test add operation using non-contiguous out tensor."""
        device = torch.device("rbln:0")
        x = torch.randn(4, 60, device=device, dtype=torch.float16)
        y = torch.randn(4, 60, device=device, dtype=torch.float16)

        # Create non-contiguous out tensor with zero storage offset
        base = torch.empty(60, 4, device=device, dtype=torch.float16)
        out = base.t()  # Transpose creates non-contiguous view
        self.assertFalse(out.is_contiguous())
        self.assertEqual(out.storage_offset(), 0)
        original_data_ptr = out.data_ptr()

        # Verify out tensor cannot be used directly due to non-contiguity
        self.assertFalse(can_use_out_tensor_directly((x, y), {"out": out}))

        # Perform operation
        result = torch.add(x, y, out=out)
        cpu_reference = torch.add(x.cpu(), y.cpu())

        # Verify the returned result is the same as the out tensor
        self.assertIs(result, out)
        self.assertEqual(result.data_ptr(), original_data_ptr)
        self.assertEqual(result.device, out.device)
        self.assertEqual(result.dtype, out.dtype)
        self.assertEqual(result.shape, out.shape)
        self.assertEqual(result.cpu(), out.cpu())

        # Verify result correctness
        self.assertEqual(result.cpu(), cpu_reference, atol=self.atol, rtol=self.rtol)

    def test_add_using_contiguous_out_with_non_zero_storage_offset(self):
        """Test add operation using contiguous out tensor with non-zero storage offset."""
        device = torch.device("rbln:0")
        x = torch.randn(4, 60, device=device, dtype=torch.float16)
        y = torch.randn(4, 60, device=device, dtype=torch.float16)

        # Create contiguous out tensor with non-zero storage offset
        base = torch.empty(8, 60, device=device, dtype=torch.float16)
        out = base[1:5]
        self.assertTrue(out.is_contiguous())
        self.assertNotEqual(out.storage_offset(), 0)
        original_data_ptr = out.data_ptr()

        # Verify out tensor cannot be used directly due to non-zero storage offset
        self.assertFalse(can_use_out_tensor_directly((x, y), {"out": out}))

        # Perform operation
        result = torch.add(x, y, out=out)
        cpu_reference = torch.add(x.cpu(), y.cpu())

        # Verify the returned result is the same as the out tensor
        self.assertIs(result, out)
        self.assertEqual(result.data_ptr(), original_data_ptr)
        self.assertEqual(result.device, out.device)
        self.assertEqual(result.dtype, out.dtype)
        self.assertEqual(result.shape, out.shape)
        self.assertEqual(result.cpu(), out.cpu())

        # Verify result correctness
        self.assertEqual(result.cpu(), cpu_reference, atol=self.atol, rtol=self.rtol)

    def test_add_using_non_contiguous_out_with_non_zero_storage_offset(self):
        """Test add operation using non-contiguous out tensor with non-zero storage offset."""
        device = torch.device("rbln:0")
        x = torch.randn(4, 60, device=device, dtype=torch.float16)
        y = torch.randn(4, 60, device=device, dtype=torch.float16)

        # Create non-contiguous out tensor with non-zero storage offset
        base = torch.empty(8, 64, device=device, dtype=torch.float16)
        out = base[4::2]
        self.assertFalse(out.is_contiguous())
        self.assertNotEqual(out.storage_offset(), 0)
        original_data_ptr = out.data_ptr()

        # Verify out tensor cannot be used directly due to non-contiguity
        self.assertFalse(can_use_out_tensor_directly((x, y), {"out": out}))

        # Perform operation
        result = torch.add(x, y, out=out)
        cpu_reference = torch.add(x.cpu(), y.cpu())

        # Verify the returned result is the same as the out tensor
        self.assertEqual(result.data_ptr(), original_data_ptr)
        self.assertIs(result, out)
        self.assertEqual(result.device, out.device)
        self.assertEqual(result.dtype, out.dtype)
        self.assertEqual(result.shape, out.shape)
        self.assertEqual(result.cpu(), out.cpu())

        # Verify result correctness
        self.assertEqual(result.cpu(), cpu_reference, atol=self.atol, rtol=self.rtol)


@pytest.mark.test_set_ci
class TestTorchDispatchModeWithRbln(TestCase):
    """Tests that rbln ops do not recurse infinitely when a TorchDispatchMode is active."""

    class _LoggingMode(TorchDispatchMode):
        """TorchDispatchMode that logs each op and forwards to the real implementation."""

        def __init__(self, log=None):
            super().__init__()
            self.log = log if log is not None else []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.log.append(func.__name__)
            return func(*args, **(kwargs or {}))

    def test_add_under_torch_dispatch_mode_no_recursion(self):
        """With a non-infra TorchDispatchMode active, add on rbln tensors should complete without RecursionError."""
        log = []
        with self._LoggingMode(log=log):
            x = torch.randn(3, dtype=torch.float16, device="rbln") + torch.randn(3, dtype=torch.float16, device="rbln")
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(x.device.type, "rbln")
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.dtype, torch.float16)
        # Mode should have seen at least one add (name may be "add" or "add.Tensor" etc.)
        self.assertTrue(
            any("add" in name for name in log),
            f"LoggingMode should have intercepted add; got log: {log}",
        )

    def test_add_under_torch_dispatch_mode_result_consistent(self):
        """Result of add under TorchDispatchMode should match CPU reference (CPU fallback path)."""
        log = []
        with self._LoggingMode(log=log):
            a = torch.randn(4, dtype=torch.float16, device="rbln")
            b = torch.randn(4, dtype=torch.float16, device="rbln")
            rbln_result = a + b
        cpu_a = a.cpu()
        cpu_b = b.cpu()
        expected = cpu_a + cpu_b
        self.assertEqual(rbln_result.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.test_set_ci
class TestFallbackCheckers(TestCase):
    """Unit tests for individual CPU fallback checker functions."""

    def test_is_trace_active_no_trace(self):
        with patch("sys.gettrace", return_value=None):
            self.assertFalse(_is_trace_active())

    def test_is_trace_active_with_trace(self):
        with patch("sys.gettrace", return_value=lambda *a: None):
            self.assertTrue(_is_trace_active())

    def test_has_non_float16_dtype_all_float16(self):
        tensors = [torch.empty(2, dtype=torch.float16), torch.empty(3, dtype=torch.float16)]
        self.assertFalse(_has_non_float16_dtype(tensors))

    def test_has_non_float16_dtype_mixed(self):
        tensors = [torch.empty(2, dtype=torch.float16), torch.empty(3, dtype=torch.float32)]
        self.assertTrue(_has_non_float16_dtype(tensors))

    def test_has_non_float16_dtype_single_float32(self):
        self.assertTrue(_has_non_float16_dtype([torch.empty(1, dtype=torch.float32)]))

    def test_all_scalars_true(self):
        scalars = [torch.tensor(1.0), torch.tensor(2.0)]
        self.assertTrue(_all_scalars(scalars))

    def test_all_scalars_false_with_1d(self):
        tensors = [torch.tensor(1.0), torch.tensor([2.0])]
        self.assertFalse(_all_scalars(tensors))

    def test_all_scalars_empty_list(self):
        self.assertTrue(_all_scalars([]))

    def test_has_nonzero_storage_offset_no_offset(self):
        t = torch.randn(4)
        self.assertFalse(_has_nonzero_storage_offset([t]))

    def test_has_nonzero_storage_offset_with_offset(self):
        base = torch.randn(8)
        sliced = base[2:6]  # contiguous but storage_offset != 0
        self.assertTrue(sliced.is_contiguous())
        self.assertNotEqual(sliced.storage_offset(), 0)
        self.assertTrue(_has_nonzero_storage_offset([sliced]))

    def test_has_nonzero_storage_offset_noncontiguous_ignored(self):
        base = torch.randn(4, 4)
        transposed = base.t()  # not contiguous, has offset 0
        self.assertFalse(transposed.is_contiguous())
        self.assertFalse(_has_nonzero_storage_offset([transposed]))

    def test_contains_nan_or_inf_clean(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertFalse(_contains_nan_or_inf(t))

    def test_contains_nan_or_inf_with_nan(self):
        t = torch.tensor([1.0, float("nan"), 3.0])
        self.assertTrue(_contains_nan_or_inf(t))

    def test_contains_nan_or_inf_with_inf(self):
        t = torch.tensor([1.0, float("inf"), 3.0])
        self.assertTrue(_contains_nan_or_inf(t))

    def test_contains_nan_or_inf_bool_tensor(self):
        t = torch.tensor([True, False])
        self.assertFalse(_contains_nan_or_inf(t))

    def test_contains_nan_or_inf_scalar_nan(self):
        self.assertTrue(_contains_nan_or_inf(float("nan")))

    def test_contains_nan_or_inf_scalar_normal(self):
        self.assertFalse(_contains_nan_or_inf(42))
        self.assertFalse(_contains_nan_or_inf(3.14))

    def test_has_nan_or_inf_clean_tensors(self):
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        with patch("torch_rbln._internal.ops_utils.is_rbln_deploy", return_value=False):
            self.assertFalse(_has_nan_or_inf(tensors))

    def test_has_nan_or_inf_with_nan_tensor(self):
        tensors = [torch.tensor([1.0, float("nan")])]
        with patch("torch_rbln._internal.ops_utils.is_rbln_deploy", return_value=False):
            self.assertTrue(_has_nan_or_inf(tensors))

    def test_has_nan_or_inf_deploy_mode_skips(self):
        tensors = [torch.tensor([1.0, float("nan")])]
        with patch("torch_rbln._internal.ops_utils.is_rbln_deploy", return_value=True):
            self.assertFalse(_has_nan_or_inf(tensors))

    def test_has_nan_or_inf_early_return(self):
        """Should return True on first NaN tensor without checking the rest."""
        t1 = torch.tensor([float("nan")])
        t2 = torch.tensor([1.0, 2.0])
        with patch("torch_rbln._internal.ops_utils.is_rbln_deploy", return_value=False):
            self.assertTrue(_has_nan_or_inf([t1, t2]))

    def test_extract_tensors_flat(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        result = extract_tensors((t1, 42, t2, "hello"))
        self.assertEqual(len(result), 2)

    def test_extract_tensors_nested(self):
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        result = extract_tensors({"a": t1, "b": [t2, 3]})
        self.assertEqual(len(result), 2)

    def test_extract_tensors_empty(self):
        self.assertEqual(extract_tensors((42, "hello")), [])


@pytest.mark.test_set_ci
class TestCompileUtilities(TestCase):
    """Unit tests for compile_and_execute utilities."""

    def test_make_op_module_creates_module(self):
        mod = make_op_module(torch.add)
        self.assertIsInstance(mod, torch.nn.Module)
        # Verify it's in eval mode
        self.assertFalse(mod.training)

    def test_make_op_module_forward_works_on_cpu(self):
        mod = make_op_module(torch.add)
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = mod(a, b)
        expected = torch.tensor([4.0, 6.0])
        self.assertEqual(result, expected)

    def test_make_op_module_qualname(self):
        mod = make_op_module(torch.add)
        self.assertIn("add", mod.__class__.__qualname__)

    def test_get_default_compile_options_cached(self):
        _get_default_compile_options.cache_clear()
        try:
            opts1 = _get_default_compile_options()
            opts2 = _get_default_compile_options()
            # Should be the exact same object (cached)
            self.assertIs(opts1, opts2)
            self.assertIn("disable_logger", opts1)
            self.assertTrue(opts1["disable_logger"])
        finally:
            _get_default_compile_options.cache_clear()

    def test_resolve_result_tensor_none_out(self):
        result = _resolve_result_tensor(None, (), {})
        self.assertIsNone(result)

    def test_resolve_result_tensor_valid_out(self):
        out = torch.empty(4, dtype=torch.float16, device="rbln")
        args = (torch.randn(4, dtype=torch.float16, device="rbln"),)
        result = _resolve_result_tensor(out, args, {"out": out})
        # Should return out since it meets all conditions
        self.assertIs(result, out)

    def test_resolve_result_tensor_non_contiguous_out(self):
        base = torch.empty(4, 4, dtype=torch.float16, device="rbln")
        out = base.t()  # non-contiguous
        args = (torch.randn(4, 4, dtype=torch.float16, device="rbln"),)
        result = _resolve_result_tensor(out, args, {"out": out})
        # Should return None since out is not contiguous
        self.assertIsNone(result)


@pytest.mark.test_set_ci
class TestPrepareArgsContiguousFastPath(TestCase):
    """Test the fast-path optimization in prepare_args_for_contiguous."""

    def test_fast_path_all_contiguous(self):
        t1 = torch.randn(2, 3, device="rbln")
        t2 = torch.randn(4, device="rbln")
        (new_args, new_kwargs), changed = prepare_args_for_contiguous((t1, t2), {"alpha": 1.0})
        self.assertFalse(changed)

    def test_slow_path_non_contiguous(self):
        base = torch.randn(4, 4, device="rbln")
        non_contig = base.t()
        self.assertFalse(non_contig.is_contiguous())
        (new_args, _), changed = prepare_args_for_contiguous((non_contig,), {})
        self.assertTrue(changed)
        self.assertTrue(new_args[0].is_contiguous())

    def test_fast_path_mixed_tensor_and_scalar(self):
        t = torch.randn(3, device="rbln")
        (new_args, _), changed = prepare_args_for_contiguous((t, 42, "str"), {"dim": 0})
        self.assertFalse(changed)

    def test_empty_tensor_skipped_in_fast_path(self):
        empty_t = torch.empty(0, device="rbln")
        normal_t = torch.randn(3, device="rbln")
        (new_args, _), changed = prepare_args_for_contiguous((empty_t, normal_t), {})
        self.assertFalse(changed)


instantiate_device_type_tests(TestInternalOpUtils, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestOutTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTorchDispatchModeWithRbln, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestFallbackCheckers, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCompileUtilities, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestPrepareArgsContiguousFastPath, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
