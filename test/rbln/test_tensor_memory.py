# Owner(s): ["module: PrivateUse1"]

"""
Test suite for tensor memory correctness.

Validates that device operations produce correct results under non-trivial memory scenarios:
- Storage aliasing: multiple tensors sharing the same underlying storage
- Input/output memory independence: ensuring that input and output tensors do not share overlapping memory
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import run_in_isolated_process, SUPPORTED_DTYPES


# Tolerance for numerical comparisons
ATOL = 0.01
RTOL = 0.01


@pytest.mark.test_set_ci
class TestAliasedTensors(TestCase):
    binary_ops = [torch.add]
    rbln_device = torch.device("rbln:0")
    shapes = [(2, 16), (2, 64)]

    def _run_binary_op(self, binary_op, rbln_input, rbln_other):
        self.assertEqual(rbln_input.device.type, "rbln")
        self.assertEqual(rbln_other.device.type, "rbln")

        rbln_out = binary_op(rbln_input, rbln_other)

        cpu_input = rbln_input.cpu()
        cpu_other = rbln_other.cpu()
        cpu_out = binary_op(cpu_input, cpu_other)
        self.assertEqual(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        input = x_base
        other = x_base
        self.assertIs(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_base_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        input = x_base
        other = y_base
        self.assertIsNot(input, other)
        self.assertNotEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_same_view_tensor_from_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())

        input = x_view
        other = x_view
        self.assertIs(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_view_tensors_from_same_base_tensor(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())
        new_x_view = x_base.view(shape)
        self.assertEqual(new_x_view.size(), x_base.size())
        self.assertEqual(new_x_view.stride(), x_base.stride())
        self.assertEqual(new_x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(new_x_view.data_ptr(), x_base.data_ptr())

        input = x_view
        other = new_x_view
        self.assertIsNot(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_different_view_tensors_from_different_base_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())
        y_view = y_base.view(shape)
        self.assertEqual(y_view.size(), y_base.size())
        self.assertEqual(y_view.stride(), y_base.stride())
        self.assertEqual(y_view.storage_offset(), y_base.storage_offset())
        self.assertEqual(y_view.data_ptr(), y_base.data_ptr())

        input = x_view
        other = y_view
        self.assertIsNot(input, other)
        self.assertNotEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("binary_op", binary_ops)
    @parametrize("shape", shapes)
    def test_mixed_base_and_view_tensors(self, dtype, binary_op, shape):
        x_base = torch.randn(shape, dtype=dtype, device=self.rbln_device)

        x_view = x_base.view(shape)
        self.assertEqual(x_view.size(), x_base.size())
        self.assertEqual(x_view.stride(), x_base.stride())
        self.assertEqual(x_view.storage_offset(), x_base.storage_offset())
        self.assertEqual(x_view.data_ptr(), x_base.data_ptr())

        input = x_base
        other = x_view
        self.assertIsNot(input, other)
        self.assertEqual(input.data_ptr(), other.data_ptr())
        self._run_binary_op(binary_op, input, other)


def _input_output_tensor_memory_independence_worker(rbln_device, dtype):
    """Worker function that runs in a spawned subprocess to ensure a clean data_ptr counter."""
    # Create a seed input tensor whose internal key value may collide with output tensor memory keys.
    seed_tensor = torch.randn([2, 2], dtype=dtype, device=rbln_device)
    seed_data_ptr = seed_tensor.data_ptr()

    rbln_out = torch.abs(seed_tensor)  # First output tensor allocation.
    cpu_out = torch.abs(seed_tensor.cpu())
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)

    # Burn through (seed_data_ptr - 1) output tensor allocations so the next output tensor receives a key equal to
    # seed_data_ptr, forcing a key collision.
    for i in range(seed_data_ptr - 1):
        # Vary the size of the tensor to ensure unique memory allocation.
        t = torch.randn([i], dtype=dtype, device=rbln_device)
        rbln_out = torch.abs(t)
        cpu_out = torch.abs(t.cpu())
        torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)
    # After the loop, the next output tensor key may collide with the seed tensor key.

    x = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)
    y = torch.randn([2, 2, 4], dtype=dtype, device=rbln_device)

    # Use trunc-mode division to amplify numerical divergence if the backend silently returns wrong data.
    rbln_out = torch.div(x, y, rounding_mode="trunc")
    cpu_out = torch.div(x.cpu(), y.cpu(), rounding_mode="trunc")
    torch.testing.assert_close(rbln_out.cpu(), cpu_out, atol=ATOL, rtol=RTOL)


@pytest.mark.test_set_ci
class TestInputOutputTensors(TestCase):
    """
    Regression tests for input and output tensor memory collisions.

    The RBLN backend may confuse an output tensor with an already-allocated
    input tensor, silently producing incorrect results. This test reproduces
    such a pattern and verifies that the results remain correct.

    The test runs in a spawned subprocess so the data_ptr counter starts from
    a clean state regardless of how many other tests have run before it. This
    is because when data_ptr is too large, it takes a long time to reach the
    collision point, making the test impractically slow.
    """

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_input_output_tensor_memory_independence(self, dtype):
        run_in_isolated_process(_input_output_tensor_memory_independence_worker, self.rbln_device, dtype)


instantiate_device_type_tests(TestAliasedTensors, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestInputOutputTensors, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
