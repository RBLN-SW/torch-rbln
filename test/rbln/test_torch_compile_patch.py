# Owner(s): ["module: PrivateUse1"]

"""
Test suite for torch_compile_patch_helpers module.
"""

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import requires_logical_devices

import torch_rbln._internal.register_custom_ops as register_custom_ops
from torch_rbln._internal.compile_cache import clear_rbln_compile_cache, compile_rbln_cached
from torch_rbln._internal.monkey_patches import patch_torch_compile, remove_all_patches
from torch_rbln._internal.ops_utils import extract_device_id_from_inputs
from torch_rbln._internal.torch_compile_patch_helpers import (
    _convert_result_to_device,
    attempt_cpu_fallback,
    auto_determine_tp_if_needed,
    CompiledFunctionWrapper,
    extract_device_from_inputs,
    get_tensor_parallel_size_from_options,
    handle_tp_failover,
    is_rbln_backend,
    recompile_with_tp_size,
    should_attempt_failover,
)


@pytest.mark.test_set_ci
class TestTorchCompilePatchHelpers(TestCase):
    """Tests for torch_compile_patch_helpers module."""

    def test_extract_device_from_inputs_with_tensor_args(self):
        """Test extracting device from tensor arguments."""
        t1 = torch.tensor([1, 2, 3], device="rbln")
        t2 = torch.tensor([4, 5], device="rbln")
        device = extract_device_from_inputs(t1, 42, t2)
        self.assertEqual(device.type, "rbln")

    def test_extract_device_from_inputs_raises_with_cpu_tensor_kwargs(self):
        """Test that CPU-only tensor inputs are rejected."""
        t = torch.tensor([1, 2], device="cpu")

        with self.assertRaises(RuntimeError) as ctx:
            extract_device_from_inputs(x=t, y="hello")
        self.assertIn("requires at least one RBLN tensor input", str(ctx.exception))

    def test_extract_device_from_inputs_raises_without_tensors(self):
        """Test that calls without tensor inputs are rejected."""
        with self.assertRaises(RuntimeError) as ctx:
            extract_device_from_inputs(42, "hello", [1, 2, 3])
        self.assertIn("requires at least one RBLN tensor input", str(ctx.exception))

    def test_extract_device_from_inputs_prefers_rbln_over_cpu(self):
        """Test that the first RBLN tensor wins even if a CPU tensor appears earlier."""
        cpu_tensor = torch.tensor([1, 2], device="cpu")
        rbln_tensor = torch.tensor([3, 4], device="rbln:0")

        device = extract_device_from_inputs(cpu_tensor, rbln_tensor)

        self.assertEqual(device.type, "rbln")
        self.assertEqual(device.index, 0)

    def test_extract_device_from_inputs_prefers_rbln_in_kwargs(self):
        """Test that RBLN tensors in kwargs are preferred over CPU tensors in args."""
        cpu_tensor = torch.tensor([1], device="cpu")
        rbln_tensor = torch.tensor([2], device="rbln:0")

        device = extract_device_from_inputs(cpu_tensor, tensor=rbln_tensor)

        self.assertEqual(device.type, "rbln")
        self.assertEqual(device.index, 0)

    def test_extract_device_id_from_inputs_with_rbln_tensor(self):
        """Test extracting device ID from RBLN tensor."""
        t = torch.tensor([1, 2, 3], device="rbln:0")
        device_id = extract_device_id_from_inputs(t)
        self.assertEqual(device_id, 0)

    @requires_logical_devices(2)
    def test_extract_device_id_from_inputs_with_multiple_tensors(self):
        """Test extracting device ID from multiple tensors."""
        t1 = torch.tensor([1], device="cpu")
        t2 = torch.tensor([2], device="rbln:1")
        device_id = extract_device_id_from_inputs(t1, t2)
        self.assertEqual(device_id, 1)

    @requires_logical_devices(3)
    def test_extract_device_id_from_inputs_with_kwargs(self):
        """Test extracting device ID from kwargs."""
        t = torch.tensor([1], device="rbln:2")
        device_id = extract_device_id_from_inputs(x=42, tensor=t)
        self.assertEqual(device_id, 2)

    def test_extract_device_id_from_inputs_no_rbln_tensors(self):
        """Test that None is returned when no RBLN tensors present."""
        t = torch.tensor([1], device="cpu")
        device_id = extract_device_id_from_inputs(t, 42, "hello")
        self.assertIsNone(device_id)

    def test_get_tensor_parallel_size_from_options_with_tp_size(self):
        """Test extracting tensor_parallel_size from compile_kwargs."""
        compile_kwargs = {"options": {"tensor_parallel_size": 4}}
        tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
        self.assertEqual(tp_size, 4)

    def test_get_tensor_parallel_size_from_options_without_tp_size(self):
        """Test that None is returned when tp_size is not set."""
        compile_kwargs = {"options": {}}
        tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
        self.assertIsNone(tp_size)

    def test_get_tensor_parallel_size_from_options_no_options(self):
        """Test that None is returned when options key is missing."""
        compile_kwargs = {}
        tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
        self.assertIsNone(tp_size)

    def test_get_tensor_parallel_size_from_options_non_dict_options(self):
        """Test that None is returned when options is not a dict."""
        compile_kwargs = {"options": "not_a_dict"}
        tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
        self.assertIsNone(tp_size)

    def test_convert_result_to_device_single_tensor(self):
        """Test converting single tensor result to target device."""
        result = torch.tensor([1, 2, 3], device="cpu")
        converted = _convert_result_to_device(result, torch.device("rbln"))
        self.assertEqual(converted.device.type, "rbln")
        self.assertEqual(converted.tolist(), [1, 2, 3])

    def test_convert_result_to_device_tuple(self):
        """Test converting tuple of tensors to target device."""
        t1 = torch.tensor([1, 2], device="cpu")
        t2 = torch.tensor([3, 4], device="cpu")
        result = (t1, "hello", t2)
        converted = _convert_result_to_device(result, torch.device("rbln"))

        self.assertIsInstance(converted, tuple)
        self.assertEqual(converted[0].device.type, "rbln")
        self.assertEqual(converted[1], "hello")
        self.assertEqual(converted[2].device.type, "rbln")

    def test_convert_result_to_device_list(self):
        """Test converting list of tensors to target device."""
        t1 = torch.tensor([1], device="cpu")
        t2 = torch.tensor([2], device="cpu")
        result = [t1, 99, t2]
        converted = _convert_result_to_device(result, torch.device("rbln"))

        self.assertIsInstance(converted, list)
        self.assertEqual(converted[0].device.type, "rbln")
        self.assertEqual(converted[1], 99)
        self.assertEqual(converted[2].device.type, "rbln")

    def test_convert_result_to_device_nested_containers(self):
        """Test recursively converting nested list/tuple/dict containers."""
        result = {
            "tuple": (
                torch.tensor([1], device="cpu"),
                [torch.tensor([2], device="cpu")],
            ),
            "list": [torch.tensor([3], device="cpu")],
        }

        converted = _convert_result_to_device(result, torch.device("rbln:0"))

        self.assertEqual(converted["tuple"][0].device.type, "rbln")
        self.assertEqual(converted["tuple"][0].device.index, 0)
        self.assertEqual(converted["tuple"][1][0].device.type, "rbln")
        self.assertEqual(converted["list"][0].device.type, "rbln")

    def test_convert_result_to_device_non_tensor(self):
        """Test that non-tensor results pass through unchanged."""
        result = "hello world"
        converted = _convert_result_to_device(result, torch.device("rbln"))
        self.assertEqual(converted, result)

    def test_attempt_cpu_fallback_basic(self):
        """Test CPU fallback with basic operation."""

        def add_one(x):
            return x + 1

        t = torch.tensor([1.0, 2.0], device="rbln")
        result = attempt_cpu_fallback(add_one, (t,), {}, torch.device("rbln"))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "rbln")
        self.assertEqual(result.tolist(), [2.0, 3.0])

    def test_attempt_cpu_fallback_with_kwargs(self):
        """Test CPU fallback with keyword arguments."""

        def multiply(x, y):
            return x * y

        t1 = torch.tensor([2.0], device="rbln")
        t2 = torch.tensor([3.0], device="rbln")
        result = attempt_cpu_fallback(multiply, (t1,), {"y": t2}, torch.device("rbln"))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "rbln")
        self.assertEqual(result.item(), 6.0)

    def test_attempt_cpu_fallback_cpu_device(self):
        """Test CPU fallback when original device is CPU."""

        def add_two(x):
            return x + 2

        t = torch.tensor([5.0], device="cpu")
        result = attempt_cpu_fallback(add_two, (t,), {}, torch.device("cpu"))

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.item(), 7.0)

    def test_attempt_cpu_fallback_no_original_fn(self):
        """Test that error is raised when original_fn is None."""
        t = torch.tensor([1.0], device="rbln")

        with self.assertRaises(ValueError) as ctx:
            attempt_cpu_fallback(None, (t,), {}, torch.device("rbln"))
        self.assertIn("original_fn is not provided", str(ctx.exception))

    def test_is_rbln_backend_string(self):
        """Test backend detection with string 'rbln'."""
        self.assertTrue(is_rbln_backend("rbln"))

    def test_is_rbln_backend_callable(self):
        """Test backend detection with callable named 'rbln_backend'."""

        def rbln_backend():
            pass

        self.assertTrue(is_rbln_backend(rbln_backend))

    def test_is_rbln_backend_other_backend(self):
        """Test that other backends are not detected as RBLN."""
        self.assertFalse(is_rbln_backend("inductor"))
        self.assertFalse(is_rbln_backend("aot_eager"))

        def other_backend():
            pass

        self.assertFalse(is_rbln_backend(other_backend))


@pytest.mark.test_set_ci
class TestTensorParallelFunctions(TestCase):
    """Tests for tensor parallel related functions."""

    def test_recompile_with_tp_size(self):
        """Test recompile_with_tp_size updates compile_kwargs correctly."""
        mock_compile_fn = Mock(return_value="recompiled_fn")
        mock_model = Mock()
        compile_kwargs = {"backend": "rbln", "options": {"some_option": "value"}}

        result = recompile_with_tp_size(mock_model, compile_kwargs, tp_size=4, original_compile_fn=mock_compile_fn)

        # Verify compile function was called with updated options
        mock_compile_fn.assert_called_once()
        call_args = mock_compile_fn.call_args
        self.assertEqual(call_args[0][0], mock_model)
        self.assertEqual(call_args[1]["backend"], "rbln")
        self.assertEqual(call_args[1]["options"]["tensor_parallel_size"], 4)
        self.assertEqual(call_args[1]["options"]["some_option"], "value")
        self.assertEqual(result, "recompiled_fn")

    def test_recompile_with_tp_size_no_existing_options(self):
        """Test recompile_with_tp_size with no existing options."""
        mock_compile_fn = Mock(return_value="recompiled_fn")
        mock_model = Mock()
        compile_kwargs = {"backend": "rbln"}

        recompile_with_tp_size(mock_model, compile_kwargs, tp_size=2, original_compile_fn=mock_compile_fn)

        call_args = mock_compile_fn.call_args
        self.assertEqual(call_args[1]["options"]["tensor_parallel_size"], 2)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tensor_parallel_size")
    def test_auto_determine_tp_if_needed_when_not_set(self, mock_auto_tp):
        """Test auto_determine_tp_if_needed when tp_size is not set."""
        mock_auto_tp.return_value = 4
        mock_compile_fn = Mock(return_value="recompiled_fn")
        mock_model = Mock()
        compile_kwargs = {"backend": "rbln"}

        result = auto_determine_tp_if_needed(
            mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn
        )

        mock_auto_tp.assert_called_once_with(0)
        self.assertEqual(result, "recompiled_fn")

    def test_auto_determine_tp_if_needed_when_already_set(self):
        """Test auto_determine_tp_if_needed when tp_size is already set."""
        mock_compile_fn = Mock()
        mock_model = Mock()
        compile_kwargs = {"options": {"tensor_parallel_size": 2}}

        result = auto_determine_tp_if_needed(
            mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn
        )

        # Should return None without calling compile function
        mock_compile_fn.assert_not_called()
        self.assertIsNone(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tensor_parallel_size")
    def test_auto_determine_tp_if_needed_when_auto_tp_fails(self, mock_auto_tp):
        """Test auto_determine_tp_if_needed when auto-determination fails."""
        mock_auto_tp.return_value = None
        mock_compile_fn = Mock()
        mock_model = Mock()
        compile_kwargs = {}

        result = auto_determine_tp_if_needed(
            mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn
        )

        mock_compile_fn.assert_not_called()
        self.assertIsNone(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_should_attempt_failover_enabled(self, mock_enable_failover):
        """Test should_attempt_failover when failover is enabled."""
        mock_enable_failover.return_value = True
        compile_kwargs = {}

        result = should_attempt_failover(device_id=0, compile_kwargs=compile_kwargs, current_tp=4)

        self.assertTrue(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_should_attempt_failover_disabled(self, mock_enable_failover):
        """Test should_attempt_failover when failover is disabled."""
        mock_enable_failover.return_value = False
        compile_kwargs = {}

        result = should_attempt_failover(device_id=0, compile_kwargs=compile_kwargs, current_tp=4)

        self.assertFalse(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_should_attempt_failover_tp_is_one(self, mock_enable_failover):
        """Test should_attempt_failover when current_tp is 1."""
        mock_enable_failover.return_value = True
        compile_kwargs = {}

        result = should_attempt_failover(device_id=0, compile_kwargs=compile_kwargs, current_tp=1)

        self.assertFalse(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_should_attempt_failover_explicit_tp_one(self, mock_enable_failover):
        """Test should_attempt_failover when tp_size=1 is explicitly set."""
        mock_enable_failover.return_value = True
        compile_kwargs = {"options": {"tensor_parallel_size": 1}}

        result = should_attempt_failover(device_id=0, compile_kwargs=compile_kwargs, current_tp=4)

        self.assertFalse(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_should_attempt_failover_explicit_tp_gt_one(self, mock_enable_failover):
        """Test should_attempt_failover when tp_size>1 is explicitly set."""
        mock_enable_failover.return_value = True
        compile_kwargs = {"options": {"tensor_parallel_size": 2}}

        result = should_attempt_failover(device_id=0, compile_kwargs=compile_kwargs, current_tp=2)

        self.assertFalse(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.get_physical_device_ids")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tensor_parallel_size")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_handle_tp_failover_success(self, mock_enable_failover, mock_auto_tp, mock_get_devices):
        """Test handle_tp_failover successfully recompiles with tp=1."""
        mock_enable_failover.return_value = True
        mock_auto_tp.return_value = 4
        mock_get_devices.return_value = [0, 1, 2, 3]
        mock_compile_fn = Mock(return_value="recompiled_fn_tp1")
        mock_model = Mock()
        mock_model.__name__ = "TestModel"
        compile_kwargs = {}

        result = handle_tp_failover(mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn)

        self.assertEqual(result, "recompiled_fn_tp1")
        mock_compile_fn.assert_called_once()

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_handle_tp_failover_not_applicable(self, mock_enable_failover):
        """Test handle_tp_failover when failover is not applicable."""
        mock_enable_failover.return_value = False
        mock_compile_fn = Mock()
        mock_model = Mock()
        compile_kwargs = {}

        result = handle_tp_failover(mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn)

        mock_compile_fn.assert_not_called()
        self.assertIsNone(result)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_handle_tp_failover_respects_explicit_tp_size(self, mock_enable_failover):
        """Test handle_tp_failover does not override explicit caller TP settings."""
        mock_enable_failover.return_value = True
        mock_compile_fn = Mock()
        mock_model = Mock()
        compile_kwargs = {"options": {"tensor_parallel_size": 4}}

        result = handle_tp_failover(mock_model, compile_kwargs, device_id=0, original_compile_fn=mock_compile_fn)

        mock_compile_fn.assert_not_called()
        self.assertIsNone(result)


@pytest.mark.test_set_ci
class TestCompiledFunctionWrapper(TestCase):
    """Tests for CompiledFunctionWrapper class."""

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_wrapper_successful_execution(self, mock_auto_determine):
        """Test that wrapper correctly executes compiled function on success."""
        mock_auto_determine.return_value = None  # No recompilation needed

        def mock_compiled_fn(x):
            return x * 2

        def original_fn(x):
            return x * 2

        mock_original_compile_fn = Mock()
        t = torch.tensor([5], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)
        result = wrapper(t)
        self.assertEqual(result.item(), 10)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_wrapper_cpu_fallback_on_error(self, mock_enable_failover, mock_auto_determine):
        """Test that wrapper falls back to CPU on error."""
        mock_enable_failover.return_value = False
        mock_auto_determine.return_value = None

        def mock_compiled_fn(x):
            raise RuntimeError("Compilation error")

        def original_fn(x):
            return x + 1

        mock_original_compile_fn = Mock()
        t = torch.tensor([1.0, 2.0], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        with patch("torch_rbln._internal.torch_compile_patch_helpers.is_fallback_disabled", return_value=False):
            result = wrapper(t)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.device.type, "rbln")
            self.assertEqual(result.tolist(), [2.0, 3.0])

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_wrapper_cpu_fallback_handles_cpu_first_inputs_and_nested_outputs(
        self,
        mock_enable_failover,
        mock_auto_determine,
    ):
        """Test fallback chooses the RBLN device and restores nested outputs recursively."""
        mock_enable_failover.return_value = False
        mock_auto_determine.return_value = None

        def mock_compiled_fn(cpu_tensor, rbln_tensor):
            raise RuntimeError("Compilation error")

        def original_fn(cpu_tensor, rbln_tensor):
            return {
                "result": (
                    cpu_tensor.to(torch.float32) + rbln_tensor.to(torch.float32),
                    [rbln_tensor.clone()],
                )
            }

        cpu_tensor = torch.tensor([1.0, 2.0], device="cpu")
        rbln_tensor = torch.tensor([3.0, 4.0], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, Mock())

        with patch("torch_rbln._internal.torch_compile_patch_helpers.is_fallback_disabled", return_value=False):
            result = wrapper(cpu_tensor, rbln_tensor)

        self.assertEqual(result["result"][0].device.type, "rbln")
        self.assertEqual(result["result"][0].device.index, 0)
        self.assertEqual(result["result"][1][0].device.type, "rbln")
        self.assertEqual(result["result"][0].cpu().tolist(), [4.0, 6.0])

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_wrapper_cpu_fallback_raises_without_rbln_inputs(
        self,
        mock_enable_failover,
        mock_auto_determine,
    ):
        """Test that RBLN fallback refuses CPU-only tensor inputs."""
        mock_enable_failover.return_value = False
        mock_auto_determine.return_value = None

        def mock_compiled_fn(cpu_tensor):
            raise RuntimeError("Compilation error")

        def original_fn(cpu_tensor):
            return cpu_tensor + 1

        cpu_tensor = torch.tensor([1.0, 2.0], device="cpu")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, Mock())

        with patch("torch_rbln._internal.torch_compile_patch_helpers.is_fallback_disabled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                wrapper(cpu_tensor)

        self.assertIn("requires at least one RBLN tensor input", str(ctx.exception))

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    @patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover")
    def test_wrapper_raises_when_compile_error_fallback_disabled(self, mock_enable_failover, mock_auto_determine):
        """Test that wrapper raises exception when 'compile_error' fallback is disabled."""
        mock_enable_failover.return_value = False
        mock_auto_determine.return_value = None

        def mock_compiled_fn(x):
            raise RuntimeError("Compilation error")

        def original_fn(x):
            return x + 1

        mock_original_compile_fn = Mock()
        t = torch.tensor([1.0], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        with patch("torch_rbln._internal.torch_compile_patch_helpers.is_fallback_disabled", return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                wrapper(t)
            self.assertIn("Compilation error", str(ctx.exception))

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_wrapper_recompile_retry_success(self, mock_auto_determine):
        """Test that wrapper retries on recompile limit and succeeds."""
        try:
            from torch._dynamo.exc import FailOnRecompileLimitHit
        except ImportError:
            self.skipTest("torch._dynamo.exc not available")

        mock_auto_determine.return_value = None
        call_count = [0]

        def mock_compiled_fn(x):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FailOnRecompileLimitHit("Recompile limit hit")
            return x * 3

        def original_fn(x):
            return x * 3

        mock_original_compile_fn = Mock()
        t = torch.tensor([4], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        with patch("torch._dynamo.reset") as mock_reset:
            result = wrapper(t)
            self.assertEqual(result.item(), 12)
            mock_reset.assert_called_once()
            self.assertEqual(call_count[0], 2)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_wrapper_recompile_retry_then_fallback(self, mock_auto_determine):
        """Test that wrapper falls back to CPU after max retries."""
        try:
            from torch._dynamo.exc import FailOnRecompileLimitHit
        except ImportError:
            self.skipTest("torch._dynamo.exc not available")

        mock_auto_determine.return_value = None

        def mock_compiled_fn(x):
            raise FailOnRecompileLimitHit("Recompile limit hit")

        def original_fn(x):
            return x + 10

        mock_original_compile_fn = Mock()
        t = torch.tensor([5.0], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        with patch("torch._dynamo.reset") as mock_reset:
            with patch("torch_rbln._internal.torch_compile_patch_helpers.is_fallback_disabled", return_value=False):
                result = wrapper(t)
                # Should have tried max_retries + 1 times (initial + 1 retry)
                self.assertEqual(mock_reset.call_count, 2)
                # Should fall back to CPU
                self.assertEqual(result.item(), 15.0)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_wrapper_auto_determine_tp_on_first_call(self, mock_auto_determine):
        """Test that TP auto-determination happens only once on first call."""

        def recompiled_fn_with_tp(x):
            return x * 2

        mock_auto_determine.return_value = recompiled_fn_with_tp

        def mock_compiled_fn(x):
            return x * 2

        def original_fn(x):
            return x * 2

        mock_original_compile_fn = Mock()
        t = torch.tensor([5], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn, compile_kwargs={})

        # First call should trigger auto-determination
        wrapper(t)
        mock_auto_determine.assert_called_once()

        # Second call should not trigger auto-determination again
        wrapper(t)
        mock_auto_determine.assert_called_once()  # Still only once

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_wrapper_tp_failover_on_runtime_error(self, mock_auto_determine):
        """Test that TP failover is triggered on RuntimeError."""
        mock_auto_determine.return_value = None

        call_count = [0]
        recompiled_called = [False]

        def mock_compiled_fn(x):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("TP execution error")
            return x * 3

        def recompiled_fn_tp1(x):
            recompiled_called[0] = True
            return x * 3

        def original_fn(x):
            return x * 3

        mock_original_compile_fn = Mock()
        t = torch.tensor([4], device="rbln:0")
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        # Mock _try_tp_failover with side_effect to update _compiled_fn
        def mock_failover_side_effect(device_id):
            wrapper._compiled_fn = recompiled_fn_tp1
            wrapper._failover_attempted = True
            return recompiled_fn_tp1

        with patch("torch_rbln._internal.torch_compile_patch_helpers.use_tp_failover", return_value=True):
            with patch.object(wrapper, "_try_tp_failover", side_effect=mock_failover_side_effect) as mock_failover:
                # Should trigger failover and retry
                result = wrapper(t)
                mock_failover.assert_called_once()
                self.assertEqual(result.item(), 12)
                self.assertTrue(recompiled_called[0])


@pytest.mark.test_set_ci
class TestTorchCompileMonkeyPatch(TestCase):
    """Tests for torch.compile monkey patching."""

    def setUp(self):
        """Store original torch.compile and torch._dynamo.reset before tests."""
        self._original_compile = torch.compile
        self._original_dynamo_reset = torch._dynamo.reset
        clear_rbln_compile_cache()

    def tearDown(self):
        """Restore original torch.compile and torch._dynamo.reset after tests."""
        torch.compile = self._original_compile
        torch._dynamo.reset = self._original_dynamo_reset
        clear_rbln_compile_cache()
        # Reset patch state
        import torch_rbln._internal.monkey_patches as mp

        mp._torch_compile_patched = False
        mp._torch_dynamo_reset_patched = False
        mp._rbln_backend_registered = False

    def _make_paged_attn_inputs(self):
        q = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        k = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        attn_mask = torch.zeros((1, 1, 1), device="rbln:0", dtype=torch.float16)
        k_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        seq = torch.zeros((1, 1), dtype=torch.int32)
        scale = torch.tensor(1.0)
        block_table = torch.zeros((1, 1), dtype=torch.int32)
        block_size = 16
        return (q, k, v, attn_mask, k_cache, v_cache, seq, scale, block_table, block_size)

    def _make_paged_causal_attn_prefill_inputs(self):
        q = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        k = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        k_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        seq = torch.zeros((1, 1), dtype=torch.int32)
        scale = torch.tensor(1.0)
        block_table = torch.zeros((1, 1), dtype=torch.int32)
        block_size = 16
        is_bidirectional = False
        return (q, k, v, k_cache, v_cache, seq, scale, block_table, block_size, is_bidirectional)

    def _make_paged_causal_attn_decode_inputs(self):
        q = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        k = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v = torch.zeros((1, 1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        k_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        v_cache = torch.zeros((1, 1, 1, 64), device="rbln:0", dtype=torch.float16)
        seq = torch.zeros((1, 1), dtype=torch.int32)
        scale = torch.tensor(1.0)
        block_table = torch.zeros((1, 1), dtype=torch.int32)
        block_size = 16
        return (q, k, v, k_cache, v_cache, seq, scale, block_table, block_size)

    def test_patch_torch_compile_idempotent(self):
        """Test that patching multiple times is safe."""
        patch_torch_compile()
        first_compile = torch.compile

        patch_torch_compile()
        second_compile = torch.compile

        self.assertIs(first_compile, second_compile)

    def test_remove_all_patches_restores_original_functions(self):
        """remove_all_patches should restore both torch.compile and torch._dynamo.reset."""
        patch_torch_compile()

        self.assertIsNot(torch.compile, self._original_compile)
        self.assertIsNot(torch._dynamo.reset, self._original_dynamo_reset)

        remove_all_patches()

        self.assertIs(torch.compile, self._original_compile)
        self.assertIs(torch._dynamo.reset, self._original_dynamo_reset)

        import torch_rbln._internal.monkey_patches as mp

        self.assertFalse(mp._torch_compile_patched)
        self.assertFalse(mp._torch_dynamo_reset_patched)

    def test_patch_torch_compile_non_rbln_backend(self):
        """Test that non-RBLN backends pass through normally."""
        patch_torch_compile()

        # Mock a simple function
        def simple_fn(x):
            return x + 1

        # Test that the wrapper correctly detects non-RBLN backend
        # We can't actually compile with inductor in tests, so we test the detection logic
        from torch_rbln._internal.torch_compile_patch_helpers import is_rbln_backend

        # Verify that non-RBLN backends are correctly identified
        self.assertFalse(is_rbln_backend("inductor"))
        self.assertFalse(is_rbln_backend("aot_eager"))

        # The patched torch.compile should still be callable
        self.assertTrue(callable(torch.compile))

    def test_patch_torch_compile_applies_correctly(self):
        """Test that patch_torch_compile correctly modifies torch.compile."""
        # Reset patch state to ensure clean test
        import torch_rbln._internal.monkey_patches as mp

        mp._torch_compile_patched = False

        # Restore original torch.compile
        torch.compile = self._original_compile

        # Store original
        original = torch.compile

        # Apply patch
        patch_torch_compile()
        patched = torch.compile

        # Should be different (wrapped)
        self.assertIsNot(original, patched)

        # Applying again should be idempotent
        patch_torch_compile()
        self.assertIs(patched, torch.compile)

    def test_compile_rbln_cached_reuses_compiled_callable_for_steady_state(self):
        """Repeated eager-op compilation should hit the Python-level cache after the first call."""
        model = Mock()
        compiled_fn = Mock(name="compiled_fn")

        with patch("torch_rbln._internal.compile_cache.torch.compile", return_value=compiled_fn) as mock_compile:
            first = compile_rbln_cached(
                model,
                dynamic=False,
                options={"disable_logger": True, "tensor_parallel_size": 1},
                device_cache_key=0,
            )
            second = compile_rbln_cached(
                model,
                dynamic=False,
                options={"disable_logger": True, "tensor_parallel_size": 1},
                device_cache_key=0,
            )

        self.assertIs(first, compiled_fn)
        self.assertIs(second, compiled_fn)
        self.assertEqual(mock_compile.call_count, 1)

    def test_compile_rbln_cached_does_not_alias_distinct_models_with_same_raw_id(self):
        """Cache keys should distinguish different model objects even if raw id() collides."""
        model_a = Mock(name="model_a")
        model_b = Mock(name="model_b")
        compiled_a = Mock(name="compiled_a")
        compiled_b = Mock(name="compiled_b")

        with patch("torch_rbln._internal.compile_cache.id", return_value=7, create=True):
            with patch("torch_rbln._internal.compile_cache.torch.compile", side_effect=[compiled_a, compiled_b]) as mock_compile:
                first = compile_rbln_cached(model_a, dynamic=False, options={"disable_logger": True}, device_cache_key=0)
                second = compile_rbln_cached(model_b, dynamic=False, options={"disable_logger": True}, device_cache_key=0)

        self.assertIs(first, compiled_a)
        self.assertIs(second, compiled_b)
        self.assertEqual(mock_compile.call_count, 2)

    @requires_logical_devices(1)
    def test_paged_attn_custom_kernel_paths_reuse_singleton_modules(self):
        """Custom paged-attn kernels should pass stable module singletons into the compile cache."""
        cases = [
            (
                register_custom_ops.paged_attn_prefill_rbln,
                register_custom_ops._paged_attn_prefill_op_module,
                self._make_paged_attn_inputs,
            ),
            (
                register_custom_ops.paged_attn_decode_rbln,
                register_custom_ops._paged_attn_decode_op_module,
                self._make_paged_attn_inputs,
            ),
            (
                register_custom_ops.paged_causal_attn_prefill_rbln,
                register_custom_ops._paged_causal_attn_prefill_op_module,
                self._make_paged_causal_attn_prefill_inputs,
            ),
            (
                register_custom_ops.paged_causal_attn_decode_rbln,
                register_custom_ops._paged_causal_attn_decode_op_module,
                self._make_paged_causal_attn_decode_inputs,
            ),
        ]

        def fake_out_tensor_context(out_tensor=None):
            return nullcontext()

        for kernel_fn, expected_module, make_inputs in cases:
            seen_models = []

            def fake_compile(model, **kwargs):
                seen_models.append(model)
                return lambda *args, **inner_kwargs: args[0].clone()

            with patch.object(register_custom_ops, "compile_rbln_cached", side_effect=fake_compile):
                with patch("torch_rbln.device.context_holder.out_tensor_context", new=fake_out_tensor_context):
                    kernel_fn(*make_inputs())
                    kernel_fn(*make_inputs())

            self.assertEqual(len(seen_models), 2)
            self.assertIs(seen_models[0], expected_module)
            self.assertIs(seen_models[1], expected_module)

    def test_compile_rbln_cached_recompiles_after_dynamo_reset(self):
        """torch._dynamo.reset should also clear the RBLN eager compile cache."""
        patch_torch_compile()
        model = Mock()

        with patch("torch_rbln._internal.compile_cache.torch.compile", side_effect=[Mock(), Mock()]) as mock_compile:
            compile_rbln_cached(
                model,
                dynamic=False,
                options={"disable_logger": True},
                device_cache_key=0,
            )
            compile_rbln_cached(
                model,
                dynamic=False,
                options={"disable_logger": True},
                device_cache_key=0,
            )
            self.assertEqual(mock_compile.call_count, 1)

            torch._dynamo.reset()

            compile_rbln_cached(
                model,
                dynamic=False,
                options={"disable_logger": True},
                device_cache_key=0,
            )

        self.assertEqual(mock_compile.call_count, 2)

    @patch("torch_rbln._internal.torch_compile_patch_helpers.auto_determine_tp_if_needed")
    def test_patch_torch_compile_wrapper_integration(self, mock_auto_determine):
        """Test CompiledFunctionWrapper integration with torch.compile."""
        mock_auto_determine.return_value = None

        # This tests that the wrapper can be instantiated and called correctly
        def mock_compiled_fn(x):
            return x * 2

        def original_fn(x):
            return x * 2

        mock_original_compile_fn = Mock()
        # Create wrapper directly (simulating what patch does)
        wrapper = CompiledFunctionWrapper(mock_compiled_fn, original_fn, mock_original_compile_fn)

        # Test with tensor
        t = torch.tensor([1, 2, 3], device="rbln:0")
        result = wrapper(t)
        self.assertEqual(result.tolist(), [2, 4, 6])


instantiate_device_type_tests(TestTorchCompilePatchHelpers, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTensorParallelFunctions, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestCompiledFunctionWrapper, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestTorchCompileMonkeyPatch, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
