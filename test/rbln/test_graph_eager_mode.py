# Owner(s): ["module: PrivateUse1"]

"""
Test suite for graph mode and eager mode compatibility.
"""

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import GRAPH_EAGER_ATOL, GRAPH_EAGER_RTOL, SUPPORTED_DTYPES


ATOL = GRAPH_EAGER_ATOL
RTOL = GRAPH_EAGER_RTOL


@pytest.mark.test_set_ci
class TestGraphMode(TestCase):
    """Test cases for graph mode with RBLN backend."""

    rbln_device = torch.device("rbln:0")

    def _run_eager_and_graph_mode(self, fn, args):
        # Run eager mode
        eager_out = fn(*args)

        # Run graph mode
        compiled_fn = torch.compile(fn, backend="rbln", dynamic=False)
        graph_out = compiled_fn(*args)

        self.assertEqual(eager_out.cpu(), graph_out.cpu(), atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_add(self, dtype):
        """Test that graph mode and eager mode produce same results for add operation."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        self._run_eager_and_graph_mode(torch.add, (x, y))

    @dtypes(*SUPPORTED_DTYPES)
    def test_to_device(self, dtype):
        """Test that graph mode and eager mode produce same results for to.device operation."""

        class Net(nn.Module):
            def forward(self, x):
                x = x.to(torch.device("cpu"))  # ← to.device
                return x * 2

        model = Net().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x,))

    @dtypes(*SUPPORTED_DTYPES)
    def test_to_other(self, dtype):
        """Test that graph mode and eager mode produce same results for to.other operation."""

        class Net(nn.Module):
            def forward(self, x):
                ref = torch.empty((), dtype=dtype)  # Reference tensor
                return (x + 1).to(ref)  # ← to.other

        model = Net().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([5, 5], dtype=dtype, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x,))

    @dtypes(*SUPPORTED_DTYPES)
    def test_matmul_bias(self, dtype):
        """Test that graph mode and eager mode produce same results for matmul with bias."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(24))

            def forward(self, x, y):
                return torch.matmul(x, y) + self.bias

        model = Net().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([8, 24], dtype=dtype, device=self.rbln_device)
        y = torch.randn([24, 24], dtype=dtype, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x, y))

    @dtypes(*SUPPORTED_DTYPES)
    def test_linear(self, dtype):
        """Test that graph mode and eager mode produce same results for linear layer."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 32)

            def forward(self, x):
                return self.linear(x)

        model = Net().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([8, 16], dtype=dtype, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x,))

    def test_linear_to_linear(self):
        """Test that graph mode and eager mode produce same results for complex linear layers."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(16, 32)
                self.relu = nn.ReLU()
                self.l2 = nn.Linear(32, 10).to(torch.float16)

            def forward(self, x):
                x = self.l1(x)
                x = self.relu(x)
                x = x.to(dtype=torch.float16)
                return self.l2(x)

        model = Net().to(self.rbln_device)
        x = torch.randn([8, 16], dtype=torch.float32, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x,))

    def test_layernorm_mixed_precision(self):
        """Test that graph mode and eager mode produce same results for layernorm with mixed precision."""

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(32)

            def forward(self, x):
                x = x.to(torch.float16)
                x = self.ln(x.to(torch.float32))
                return x.to(torch.float16)

        model = Net().to(self.rbln_device)
        x = torch.randn([6, 32], dtype=torch.float16, device=self.rbln_device)

        self._run_eager_and_graph_mode(model, (x,))


@pytest.mark.test_set_ci
class TestGraphEagerMode(TestCase):
    """Test cases for graph mode and eager mode compatibility with rbln backend."""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    def test_mixed_graph_eager_operations(self, dtype):
        """Test mixing graph mode and eager mode operations in sequence."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        # Compile some operations
        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)
        compiled_mul = torch.compile(torch.mul, backend="rbln", dynamic=False)

        # Mix graph and eager operations
        result1 = compiled_add(x, y)  # Graph mode
        result2 = torch.mul(result1, 2.0)  # Eager mode
        result3 = compiled_mul(result2, y)  # Graph mode
        result4 = torch.add(result3, x)  # Eager mode

        # Verify the result is valid
        self.assertIsInstance(result4, torch.Tensor)
        self.assertEqual(result4.device.type, "rbln")
        self.assertEqual(result4.shape, x.shape)

    @dtypes(*SUPPORTED_DTYPES)
    def test_nested_graph_eager_calls(self, dtype):
        """Test nested calls where graph mode calls eager mode operations."""

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                # This will be compiled as graph mode
                out = self.linear(x)
                return out

        model = MixedModel().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([8, 16], dtype=dtype, device=self.rbln_device)

        # Graph mode compilation
        compiled_model = torch.compile(model, backend="rbln", dynamic=False)
        graph_out = compiled_model(x)

        # Eager mode for comparison
        eager_out = model(x)

        self.assertEqual(eager_out.cpu(), graph_out.cpu(), atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_multiple_graph_compilations(self, dtype):
        """Test that multiple graph mode compilations work correctly."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        # Compile multiple operations
        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)
        compiled_sub = torch.compile(torch.sub, backend="rbln", dynamic=False)
        compiled_mul = torch.compile(torch.mul, backend="rbln", dynamic=False)

        # Use all compiled operations
        result1 = compiled_add(x, y)
        result2 = compiled_sub(x, y)
        result3 = compiled_mul(result1, result2)

        # Verify results
        self.assertIsInstance(result3, torch.Tensor)
        self.assertEqual(result3.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_graph_mode_with_eager_input_preparation(self, dtype):
        """Test graph mode with inputs prepared in eager mode."""

        # Prepare inputs in eager mode
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        z = torch.add(x, y)  # Eager operation

        # Use graph mode with eager-prepared inputs
        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)
        result = compiled_add(z, x)  # Graph mode with eager-prepared input

        # Verify result
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_eager_mode_with_graph_output(self, dtype):
        """Test eager mode operations using outputs from graph mode."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        # Graph mode operation
        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)
        graph_output = compiled_add(x, y)

        # Use graph output in eager mode
        eager_result1 = torch.mul(graph_output, 2.0)
        eager_result2 = torch.add(eager_result1, x)

        # Verify results
        self.assertIsInstance(eager_result2, torch.Tensor)
        self.assertEqual(eager_result2.device.type, "rbln")

    @dtypes(*SUPPORTED_DTYPES)
    def test_sequential_graph_eager_alternating(self, dtype):
        """Test alternating between graph mode and eager mode operations."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)

        # Alternate between graph and eager
        result = x
        for i in range(5):
            if i % 2 == 0:
                result = compiled_add(result, y)  # Graph mode
            else:
                result = torch.mul(result, 1.1)  # Eager mode

        # Verify final result
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "rbln")
        self.assertEqual(result.shape, x.shape)

    @dtypes(*SUPPORTED_DTYPES)
    def test_model_with_both_modes(self, dtype):
        """Test a model that uses both graph and eager operations."""

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 32)
                self.linear2 = nn.Linear(32, 16)

            def forward(self, x):
                # These will be compiled as graph mode
                out = self.linear1(x)
                out = self.linear2(out)
                return out

        model = HybridModel().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([8, 16], dtype=dtype, device=self.rbln_device)

        # Compile model (graph mode)
        compiled_model = torch.compile(model, backend="rbln", dynamic=False)

        # Run in graph mode
        graph_out = compiled_model(x)

        # Run same operations in eager mode for comparison
        eager_out = model(x)

        self.assertEqual(eager_out.cpu(), graph_out.cpu(), atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_functional_ops_graph_eager_mix(self, dtype):
        """Test mixing functional operations in graph and eager modes."""
        x = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)

        # Compile some functional ops
        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)
        compiled_softmax = torch.compile(torch.softmax, backend="rbln", dynamic=False)

        # Mix operations
        result1 = compiled_add(x, y)  # Graph
        result3 = compiled_softmax(result1, dim=-1)  # Graph
        result4 = torch.log(result3 + 1e-8)  # Eager

        # Verify result
        self.assertIsInstance(result4, torch.Tensor)
        self.assertEqual(result4.device.type, "rbln")
        self.assertEqual(result4.shape, x.shape)

    @dtypes(*SUPPORTED_DTYPES)
    def test_reuse_compiled_function(self, dtype):
        """Test reusing a compiled function multiple times with different inputs."""

        compiled_add = torch.compile(torch.add, backend="rbln", dynamic=False)

        # Use compiled function multiple times
        x1 = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y1 = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        result1 = compiled_add(x1, y1)

        x2 = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        y2 = torch.randn([4, 4], dtype=dtype, device=self.rbln_device)
        result2 = compiled_add(x2, y2)

        # Compare with eager mode
        eager_result1 = torch.add(x1, y1)
        eager_result2 = torch.add(x2, y2)

        self.assertEqual(eager_result1.cpu(), result1.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(eager_result2.cpu(), result2.cpu(), atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    def test_model_with_many_ops_graph_vs_eager(self, dtype):
        """Test a model with many operations comparing graph mode and eager mode."""

        class MultiOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 64)
                self.linear2 = nn.Linear(64, 32)
                self.linear3 = nn.Linear(32, 16)

            def forward(self, x, y):
                # Element-wise operations
                out = torch.add(x, y)  # add
                out = torch.mul(out, 0.5)  # mul
                out = torch.sub(out, x * 0.1)  # sub
                out = torch.div(out, 2.0)  # div

                # Linear transformation
                out = self.linear1(out)

                # Activation functions
                out = torch.sigmoid(out)  # sigmoid

                # Mathematical operations
                out = torch.exp(out * 0.1)  # exp
                out = torch.log(out + 1.0)  # log
                out = torch.abs(out)  # abs

                # Linear transformation
                out = self.linear2(out)

                # Reduction operations
                out_mean = torch.mean(out, dim=-1, keepdim=True)  # mean
                out_sum = torch.sum(out, dim=-1, keepdim=True)  # sum
                out = torch.add(out, out_mean)
                out = torch.mul(out, out_sum * 0.01)

                # Softmax
                out = torch.softmax(out, dim=-1)  # softmax

                # Reshape operations
                out = torch.reshape(out, (out.shape[0], -1))  # reshape

                # Final linear transformation
                out = self.linear3(out)

                # Clamp
                out = torch.clamp(out, min=-2.0, max=2.0)  # clamp

                return out

        model = MultiOpModel().to(device=self.rbln_device, dtype=dtype)
        x = torch.randn([8, 32], dtype=dtype, device=self.rbln_device)
        y = torch.randn([8, 32], dtype=dtype, device=self.rbln_device)

        # Eager mode
        eager_out = model(x, y)

        # Graph mode
        compiled_model = torch.compile(model, backend="rbln", dynamic=False)
        graph_out = compiled_model(x, y)

        self.assertEqual(eager_out.cpu(), graph_out.cpu(), atol=ATOL, rtol=RTOL)


instantiate_device_type_tests(TestGraphMode, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestGraphEagerMode, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
