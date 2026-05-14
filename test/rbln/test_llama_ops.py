# Owner(s): ["module: PrivateUse1"]

"""
Test suite for operations used in LLaMA model implementations.

This test suite covers the following operations and functions:
1. Indexing, slicing, joining, mutating ops: torch.cat
2. Pointwise ops: torch.add, torch.mul, torch.neg, torch.pow, torch.rsqrt
3. Reduction ops: torch.mean
4. BLAS and LAPACK ops: torch.bmm, torch.matmul
5. Non-linear activation functions: torch.nn.functional.softmax
6. Linear functions: torch.nn.functional.linear
"""

from typing import Any

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


def convert_tensor(arg: Any) -> Any:
    if isinstance(arg, list):
        return [convert_tensor(item) for item in arg]
    elif isinstance(arg, tuple):
        return tuple(convert_tensor(item) for item in arg)
    elif isinstance(arg, torch.Tensor):
        return arg.to("rbln")
    else:
        return arg


def assert_equal_where_specials_match(a: torch.Tensor, b: torch.Tensor, *, atol=1e-5, rtol=1e-3):
    a = a.cpu()
    b = b.cpu()
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} != {b.shape}")

    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)

    # 1. check finite values
    both_finite = a_finite & b_finite
    if both_finite.any():
        torch.testing.assert_close(a[both_finite], b[both_finite], atol=atol, rtol=rtol, check_dtype=False)

    # 2. check non-finite values
    mismatch = a_finite ^ b_finite
    if mismatch.any():
        idx = torch.nonzero(mismatch, as_tuple=False)
        raise AssertionError(f"Mismatch at finite/special boundary: {idx}")


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestLlamaOps(TestCase):
    batch_sizes = [1, 2]

    def _run_op(self, op, *args, **kwargs):
        cpu_out = op(*args, **kwargs)
        rbln_args = convert_tensor(args)
        rbln_out = op(*rbln_args, **kwargs)
        self.assertEqual(cpu_out.shape, rbln_out.shape)

        atol = 0.016
        rtol = 0.03
        op_name = op.__name__
        ignore_inf_nan = False

        if op_name == "rsqrt" or op_name == "pow":
            ignore_inf_nan = True
        elif op_name == "linear" or op_name == "matmul":
            atol = 2
            rtol = 0.05

        if ignore_inf_nan:
            assert_equal_where_specials_match(cpu_out, rbln_out, atol=atol, rtol=rtol)
        else:
            self.assertEqual(cpu_out, rbln_out.cpu(), atol=atol, rtol=rtol)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(8, 1), (8, 2048), (1, 8, 8), (8, 8, 64), (32, 8, 8), (32, 8, 64)])
    def test_llama_add(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        other = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.add, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("n,m,p", [(32, 1, 8), (8, 64, 8)])
    def test_llama_bmm(self, dtype, batch_size, n, m, p):
        input = torch.randn([batch_size, n, m], dtype=dtype)
        mat2 = torch.randn([batch_size, m, p], dtype=dtype)
        self._run_op(torch.bmm, input, mat2)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("b,n,m,p", [(32, 8, 8, 64)])
    def test_llama_bmm_without_batch(self, dtype, b, n, m, p):
        input = torch.randn([b, n, m], dtype=dtype)
        mat2 = torch.randn([b, m, p], dtype=dtype)
        self._run_op(torch.bmm, input, mat2)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    def test_llama_matmul(self, dtype, batch_size):
        input = torch.randn([batch_size, 128, 12, 128], dtype=dtype)
        # Get original stride and reshape to (batch, 12, 128, 128)
        input = input.transpose(-2, -3)
        mat2 = torch.randn([batch_size, 12, 128, 128], dtype=dtype)
        # Get original stride and reshape to (batch, 12, 128, 128) with transposed stride
        mat2 = mat2.transpose(-1, -2)
        self._run_op(torch.matmul, input, mat2)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("tensor_shape,kwargs", [((8, 32), {"dim": 2}), ((32, 8, 32), {"dim": 3})])
    def test_llama_cat(self, dtype, batch_size, tensor_shape, kwargs):
        tensors = (
            torch.randn([batch_size, *tensor_shape], dtype=dtype),
            torch.randn([batch_size, *tensor_shape], dtype=dtype),
        )
        self._run_op(torch.cat, tensors, **kwargs)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape,kwargs", [((8, 2048), {"dim": 2, "keepdim": True})])
    def test_llama_mean(self, dtype, batch_size, input_shape, kwargs):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.mean, input, **kwargs)

    @dtypes(*SUPPORTED_DTYPES)
    def test_llama_mean_specific(self, dtype):
        # Test case from test_mean_simple.py with specific tensor values
        input = torch.tensor(
            [
                [[[5.9844, 2.0117]], [[-2.7344, -3.4375]]],
                [[[8.6406, -2.5664]], [[-7.7891, -0.0352]]],
                [[[0.1494, -8.1641]], [[-6.0117, 6.5664]]],
            ],
            dtype=dtype,
        )
        dim = [2]
        keepdim = True
        self._run_op(torch.mean, input, dim=dim, keepdim=keepdim)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(8, 64), (8, 2048), (8, 8, 64), (32, 8, 8), (32, 8, 64), (8, 8192)])
    def test_llama_mul(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        other = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.mul, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("input_shape", [(8, 8)])
    def test_llama_mul_without_batch(self, dtype, input_shape):
        input = torch.randn(*input_shape, dtype=dtype)
        other = torch.randn(*input_shape, dtype=dtype)
        self._run_op(torch.mul, input, other)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(8, 8, 64), (32, 8, 32)])
    def test_llama_neg(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.neg, input)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("in_features,out_features", [(2048, 512), (2048, 2048), (2048, 8192), (8192, 2048), (2048, 128256)])
    def test_llama_nn_functional_linear(self, dtype, batch_size, in_features, out_features):
        input = torch.randn([batch_size, 8, in_features], dtype=dtype)
        weight = torch.randn([out_features, in_features], dtype=dtype)
        self._run_op(torch.nn.functional.linear, input, weight)

    # 3D input × bias-less linear where the compiler generates a post-GEMM host
    # op sequence whose reshape and transpose ranks disagree. rebel-compiler
    # PR #10429 (GEMM primitive interface change, released in dev281) made
    # WrapHostOps emit a 3D reshape followed by a 4D transpose for the output,
    # tripping `transpose_op.axis_order().size() == curr_shape.size()` at
    # runtime/vmemory/transform/calculate.cc:250. Fixed by rebel-compiler
    # PR #10476 (dev top).
    #
    # Shapes come from real compiled graphs captured during perf testing: the
    # (1, 16, 1024) × (640, 1024) case is the one reported on Slack; the (192)
    # and (768) siblings were in the same trace and were empirically verified
    # to reproduce the identical runtime assert on rebel-compiler==dev281, but
    # not on dev307+.
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "batch_size,seq_len,in_features,out_features",
        [
            (1, 16, 1024, 192),
            (1, 16, 1024, 640),
            (1, 16, 1024, 768),
        ],
    )
    def test_llama_nn_functional_linear_host_op_seq_regression(
        self, dtype, batch_size, seq_len, in_features, out_features
    ):
        input = torch.randn([batch_size, seq_len, in_features], dtype=dtype)
        weight = torch.randn([out_features, in_features], dtype=dtype)
        self._run_op(torch.nn.functional.linear, input, weight, None)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(8, 2048)])
    def test_llama_pow(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        exponent = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.pow, input, exponent)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(8, 1)])
    def test_llama_rsqrt(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.rsqrt, input)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size", batch_sizes)
    @parametrize("input_shape", [(32, 8, 8)])
    def test_llama_nn_functional_softmax(self, dtype, batch_size, input_shape):
        input = torch.randn([batch_size, *input_shape], dtype=dtype)
        self._run_op(torch.nn.functional.softmax, input)


instantiate_device_type_tests(TestLlamaOps, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
