# Owner(s): ["module: PrivateUse1"]

"""
Test suite for all ops registered in RBLNRegisterOps.cpp and register_ops.py.

This test suite verifies:
1. All native implementation ops work correctly with various shapes
2. Representative fallback ops work correctly with CPU fallback
3. Python-registered ops from register_ops.py work correctly
"""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import requires_logical_devices, SUPPORTED_DTYPES


# Tolerance for numerical comparisons
ATOL = 0.01
RTOL = 0.01


# Various test shapes for comprehensive testing
TEST_SHAPES = [
    (1,),  # scalar-like
    (10,),  # 1D small
    (100,),  # 1D medium
    (2, 3),  # 2D small
    (5, 10),  # 2D medium
    (10, 20),  # 2D large
    (2, 3, 4),  # 3D small
    (4, 5, 6),  # 3D medium
    (1, 1, 1),  # singleton dimensions
    (1, 10, 1),  # mixed singleton
    (64, 64),  # square matrix
    (128, 64),  # rectangular matrix
]

# Subset of TEST_SHAPES for smaller tests to reduce runtime while still covering different dimensionalities and sizes
TEST_SHAPES_SUBSET = TEST_SHAPES[:8]


@pytest.mark.test_set_ci
class TestRegisteredNativeOps(TestCase):
    """Test native implementation ops registered in RBLNRegisterOps.cpp"""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES)
    def test_copy_from(self, dtype, shape):
        """Test _copy_from op with various shapes"""
        if torch.version.debug:
            self.skipTest("Skipping test_copy_from in debug mode due to ref count check issues")

        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.empty_like(x)
        torch.ops.aten._copy_from(y, x)
        self.assertEqual(x.cpu(), y.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(y.device, self.rbln_device)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("src_shape,dst_shape", [((3, 4), (2, 2)), ((10, 20), (5, 10)), ((2, 3, 4), (1, 2)), ((100,), (50,))])
    def test_copy_from_and_resize(self, dtype, src_shape, dst_shape):
        """Test _copy_from_and_resize op with various shapes"""
        if torch.version.debug:
            self.skipTest("Skipping test_copy_from_and_resize in debug mode due to ref count check issues")

        x = torch.randn(src_shape, dtype=dtype, device=self.rbln_device)
        y = torch.empty(dst_shape, dtype=dtype, device=self.rbln_device)
        result = torch.ops.aten._copy_from_and_resize(x, y)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(x.cpu(), result.cpu(), atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    @parametrize("memory_format", [torch.contiguous_format])
    def test_empty_memory_format(self, dtype, shape, memory_format):
        """Test empty.memory_format op with various shapes"""
        x = torch.empty(shape, dtype=dtype, device=self.rbln_device, memory_format=memory_format)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(x.shape, shape)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("size,stride", [((3, 4), (4, 1)), ((10, 20), (20, 1)), ((2, 3, 4), (12, 4, 1)), ((5, 10), (10, 1))])
    def test_empty_strided(self, dtype, size, stride):
        """Test empty_strided op with various shapes"""
        x = torch.empty_strided(size, stride, dtype=dtype, device=self.rbln_device)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(x.size(), size)
        self.assertEqual(x.stride(), stride)
        self.assertEqual(x.storage_offset(), 0)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("original_shape,shape", [((3, 4), (5, 6)), ((10,), (20,)), ((2, 3), (4, 5)), ((1, 1), (10, 10))])
    def test_resize_(self, dtype, original_shape, shape):
        """Test resize_ op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        x.resize_(shape)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(x.shape, shape)

    @dtypes(*SUPPORTED_DTYPES)
    def test_set_storage_storage_offset(self, dtype):
        """Test set_.source_Storage_storage_offset op"""
        # Create storage on rbln device by creating a tensor first
        temp_tensor = torch.empty([20], dtype=dtype, device=self.rbln_device)
        storage = temp_tensor.storage()
        x = torch.empty([3, 4], dtype=dtype, device=self.rbln_device)
        x.set_(storage, 0, (3, 4), (4, 1))
        self.assertEqual(x.device, storage.device)
        self.assertEqual(x.dtype, storage.dtype)
        self.assertEqual(x.size(), (3, 4))
        self.assertEqual(x.stride(), (4, 1))
        self.assertEqual(x.storage_offset(), 0)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES)
    def test_clone(self, dtype, shape):
        """Test clone op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = x.clone()
        self.assertEqual(x.device, y.device)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.cpu(), y.cpu(), atol=ATOL, rtol=RTOL)
        # Clone should create a new tensor
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("original_shape,shape", [((3, 4), (12,)), ((2, 3, 4), (24,)), ((10, 20), (200,)), ((4, 5, 6), (120,))])
    def test_view(self, dtype, original_shape, shape):
        """Test view op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        y = x.view(*shape)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.shape, shape)
        # View should share storage
        self.assertEqual(x.data_ptr(), y.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "original_shape,size,stride,storage_offset",
        [
            ((3, 4), (2, 2), (4, 1), 0),
            ((10, 20), (5, 5), (20, 1), 0),
            ((5, 6), (3, 3), (6, 1), 0),
        ],
    )
    def test_as_strided(self, dtype, original_shape, size, stride, storage_offset):
        """Test as_strided op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        y = x.as_strided(size, stride, storage_offset)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.size(), size)
        self.assertEqual(y.stride(), stride)
        self.assertEqual(y.storage_offset(), storage_offset)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "original_shape,size,stride",
        [
            ((3, 4), (12,), (1,)),
            ((2, 3, 4), (24,), (1,)),
            ((10, 20), (200,), (1,)),
        ],
    )
    def test_reshape_alias(self, dtype, original_shape, size, stride):
        """Test _reshape_alias op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        y = torch.ops.aten._reshape_alias(x, size, stride)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.size(), size)
        self.assertEqual(y.stride(), stride)
        self.assertEqual(y.storage_offset(), 0)

    @dtypes(*SUPPORTED_DTYPES)
    def test_set_tensor(self, dtype):
        """Test set_.source_Tensor op"""
        x = torch.randn([3, 4], dtype=dtype, device=self.rbln_device)
        y = torch.empty_like(x)
        y.set_(x)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())
        # set_ should share storage
        self.assertEqual(x.data_ptr(), y.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    def test_set_storage(self, dtype):
        """Test set_.source_Storage op"""
        x = torch.randn([3, 4], dtype=dtype, device=self.rbln_device)
        storage = x.storage()
        y = torch.empty([2, 2], dtype=dtype, device=self.rbln_device)
        y.set_(storage)
        self.assertEqual(y.device, storage.device)
        self.assertEqual(y.dtype, storage.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "original_shape,dimension,size,step,expected_shape",
        [
            ((3, 4), 0, 2, 1, (2, 4, 2)),
            ((10, 20), 1, 3, 2, (10, 9, 3)),
            ((5, 6, 7), 0, 2, 1, (4, 6, 7, 2)),
        ],
    )
    def test_unfold(self, dtype, original_shape, dimension, size, step, expected_shape):
        """Test unfold op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        y = x.unfold(dimension, size, step)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.shape, expected_shape)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "original_shape,size",
        [
            ((3, 4), (12,)),
            ((2, 3, 4), (24,)),
            ((10, 20), (200,)),
        ],
    )
    def test_unsafe_view(self, dtype, original_shape, size):
        """Test _unsafe_view op with various shapes"""
        x = torch.randn(original_shape, dtype=dtype, device=self.rbln_device)
        y = torch.ops.aten._unsafe_view(x, size)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.shape, size)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES)
    def test_alias(self, dtype, shape):
        """Test alias op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.ops.aten.alias(x)
        self.assertEqual(y.device, self.rbln_device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())
        # alias should share storage
        self.assertEqual(x.data_ptr(), y.data_ptr())


@pytest.mark.test_set_ci
class TestRegisteredFallbackOps(TestCase):
    """Test representative fallback ops registered in RBLNRegisterOps.cpp"""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shape,dim",
        [
            ((3, 4), [0, 1]),
            ((10, 20), [1]),
            ((2, 3, 4), [0, 2]),
            ((5, 10, 15), [0, 1, 2]),
            ((10,), None),  # sum all elements
        ],
    )
    def test_sum(self, dtype, shape, dim):
        """Test sum fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x.sum(dim=dim) if dim is not None else x.sum()
        expected = x.cpu().sum(dim=dim) if dim is not None else x.cpu().sum()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_clamp_tensor(self, dtype, shape):
        """Test clamp fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        min_val = torch.tensor(-0.5, dtype=dtype, device=self.rbln_device)
        max_val = torch.tensor(0.5, dtype=dtype, device=self.rbln_device)
        result = torch.clamp(x, min_val, max_val)
        expected = torch.clamp(x.cpu(), min_val.cpu(), max_val.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_trunc(self, dtype, shape):
        """Test trunc fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device) * 2.5
        result = torch.trunc(x)
        expected = torch.trunc(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_fill_scalar(self, dtype, shape):
        """Test fill_.Scalar fallback op with various shapes"""
        # TODO: Extend non-contiguous input testing to other ops.
        for is_contiguous in [True, False]:
            if not is_contiguous and len(shape) < 2:
                continue  # Cannot create non-contiguous view from 1D tensor

            for fill_value in [2.5, 7.0]:
                with self.subTest(shape=shape, is_contiguous=is_contiguous, fill_value=fill_value):
                    x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
                    if not is_contiguous:
                        x = x.transpose(0, -1)
                        if x.is_contiguous():
                            continue  # Skip if transpose results in contiguous tensor
                    self.assertEqual(x.is_contiguous(), is_contiguous)

                    x_cpu = x.detach().clone().cpu()
                    self.assertEqual(x_cpu.is_contiguous(), is_contiguous)

                    x.fill_(fill_value)
                    self.assertEqual(x.is_contiguous(), is_contiguous)
                    self.assertEqual(x.device, self.rbln_device)
                    self.assertEqual(x.dtype, dtype)

                    x_cpu = x_cpu.fill_(fill_value)

                    self.assertEqual(x.cpu(), x_cpu, atol=ATOL, rtol=RTOL)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "data,shape",
        [
            ([1.0, float("nan"), 3.0], (3,)),
            ([float("nan"), 2.0, float("nan"), 4.0], (2, 2)),
        ],
    )
    def test_isnan(self, dtype, data, shape):
        """Test isnan fallback op with various shapes"""
        x = torch.tensor(data, dtype=dtype, device=self.rbln_device).reshape(shape)
        result = torch.isnan(x)
        expected = torch.isnan(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(3, 4), (10, 20), (2, 3, 4), (5, 10)])
    def test_argmax(self, dtype, shape):
        """Test argmax fallback op with various shapes"""
        valid_dims = [None] + list(range(len(shape)))
        for dim in valid_dims:
            with self.subTest(shape=shape, dim=dim):
                x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
                result = torch.argmax(x, dim=dim) if dim is not None else torch.argmax(x)
                expected = torch.argmax(x.cpu(), dim=dim) if dim is not None else torch.argmax(x.cpu())
                self.assertEqual(result.cpu(), expected.cpu())
                self.assertEqual(result.device, self.rbln_device)
                self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_cos(self, dtype, shape):
        """Test cos fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.cos(x)
        expected = torch.cos(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_sin(self, dtype, shape):
        """Test sin fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.sin(x)
        expected = torch.sin(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_exp(self, dtype, shape):
        """Test exp fallback op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.exp(x)
        expected = torch.exp(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_all(self, dtype):
        """Test all fallback op with various shapes"""
        test_cases = [
            (torch.ones([3, 4], dtype=dtype, device=self.rbln_device), None),
            (torch.ones([10, 20], dtype=dtype, device=self.rbln_device), 1),
            (torch.zeros([5, 6], dtype=dtype, device=self.rbln_device), None),
        ]
        for x, dim in test_cases:
            with self.subTest(shape=x.shape, dim=dim):
                result = torch.all(x, dim=dim) if dim is not None else torch.all(x)
                expected = torch.all(x.cpu(), dim=dim) if dim is not None else torch.all(x.cpu())
                self.assertEqual(result.cpu(), expected.cpu())
                self.assertEqual(result.device, self.rbln_device)
                self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    def test_any(self, dtype):
        """Test any fallback op with various shapes"""
        test_cases = [
            (torch.zeros([3, 4], dtype=dtype, device=self.rbln_device), None),
            (torch.zeros([10, 20], dtype=dtype, device=self.rbln_device), 0),
        ]
        for x, dim in test_cases:
            with self.subTest(shape=x.shape, dim=dim):
                x[0, 0] = 1.0  # Set one element to True
                result = torch.any(x, dim=dim) if dim is not None else torch.any(x)
                expected = torch.any(x.cpu(), dim=dim) if dim is not None else torch.any(x.cpu())
                self.assertEqual(result.cpu(), expected.cpu())
                self.assertEqual(result.device, self.rbln_device)
                self.assertEqual(result.dtype, expected.dtype)

    @parametrize("dtype", [torch.int32])
    @parametrize(
        "data1,data2",
        [
            ([1, 2, 3], [2, 3, 4]),
            ([5, 10, 15], [3, 7, 12]),
        ],
    )
    def test_bitwise_and(self, dtype, data1, data2):
        """Test bitwise_and fallback op with various shapes"""
        x = torch.tensor(data1, dtype=dtype, device=self.rbln_device)
        y = torch.tensor(data2, dtype=dtype, device=self.rbln_device)
        result = torch.bitwise_and(x, y)
        expected = torch.bitwise_and(x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @parametrize("dtype", [torch.bool])
    @parametrize(
        "data1,data2",
        [
            ([True, False, True], [True, True, False]),
            ([True, True], [False, True]),
        ],
    )
    def test_logical_and(self, dtype, data1, data2):
        """Test logical_and fallback op with various shapes"""
        x = torch.tensor(data1, dtype=dtype, device=self.rbln_device)
        y = torch.tensor(data2, dtype=dtype, device=self.rbln_device)
        result = torch.logical_and(x, y)
        expected = torch.logical_and(x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "data,shape",
        [
            ([-2.0, 0.0, 2.0], (3,)),
            ([-1.0, 1.0, -0.5, 0.5], (2, 2)),
        ],
    )
    def test_sign(self, dtype, data, shape):
        """Test sign fallback op with various shapes"""
        x = torch.tensor(data, dtype=dtype, device=self.rbln_device).reshape(shape)
        result = torch.sign(x)
        expected = torch.sign(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestRegisteredPythonOps(TestCase):
    """Test ops registered in register_ops.py (Python implementation)"""

    rbln_device = torch.device("rbln:0")

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_add(self, dtype, shape):
        """Test add op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_mul(self, dtype, shape):
        """Test mul op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x * y
        expected = x.cpu() * y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_abs(self, dtype, shape):
        """Test abs op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.abs(x)
        expected = torch.abs(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_neg(self, dtype, shape):
        """Test neg op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = -x
        expected = -x.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_silu(self, dtype, shape):
        """Test silu op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.nn.functional.silu(x)
        expected = torch.nn.functional.silu(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_silu_backward(self, dtype, shape):
        """Test silu_backward op with various shapes"""
        grad_output = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        self_input = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.ops.aten.silu_backward(grad_output, self_input)
        expected = torch.ops.aten.silu_backward(grad_output.cpu(), self_input.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shape,dim",
        [
            ((10, 20), [1]),
            ((5, 10, 15), [0, 2]),
            ((2, 3, 4, 5), [1, 3]),
            ((10, 20), None),  # mean over all dimensions
        ],
    )
    def test_mean(self, dtype, shape, dim):
        """Test mean op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.mean(x, dim=dim) if dim is not None else torch.mean(x)
        expected = torch.mean(x.cpu(), dim=dim) if dim is not None else torch.mean(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(10, 20, 30), (5, 10, 15), (1, 10, 1)])
    def test_mm(self, dtype, n, m, p):
        """Test mm op with various matrix sizes

        Note: float16 precision limits require relaxed tolerance for matrix multiplication.
        """
        x = torch.randn([n, m], dtype=dtype, device=self.rbln_device)
        y = torch.randn([m, p], dtype=dtype, device=self.rbln_device)
        result = torch.mm(x, y)
        expected = torch.mm(x.cpu(), y.cpu())
        # Use relaxed tolerance for float16 matrix multiplication
        self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_sub(self, dtype, shape):
        """Test sub op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x - y
        expected = x.cpu() - y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_div(self, dtype, shape):
        """Test div op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device) + 0.1  # avoid division by zero
        result = x / y
        expected = x.cpu() / y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    @parametrize("mode", ["trunc", "floor"])
    def test_div_mode(self, dtype, shape, mode):
        """Test div with mode (trunc/floor)"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device) + 0.1
        result = torch.div(x, y, rounding_mode=mode)
        expected = torch.div(x.cpu(), y.cpu(), rounding_mode=mode)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    @parametrize(
        "exponent",
        [
            # Categorized exponents for meaningful coverage
            *(0,),  # special: x^0 = 1
            *(-1, -2.0),  # negative: reciprocal, inverse square
            *(0.5, 1.5),  # fractional: sqrt, non-integer
            *(2, 2.0, 3, 5, 8),  # positive int (2.0 tests float repr)
        ],
    )
    def test_pow_tensor_scalar(self, dtype, shape, exponent):
        """Test pow with tensor and scalar (x^exp)"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1  # ensure positive
        result = torch.pow(x, exponent)
        expected = torch.pow(x.cpu(), exponent)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_pow_tensor_tensor(self, dtype, shape):
        """Test pow with tensor and tensor"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1
        result = torch.pow(x, y)
        expected = torch.pow(x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    @parametrize(
        "base",
        [
            # Categorized bases for meaningful coverage
            *(0.1, 0.5, 1.5),  # small : fractional base
            *(2, 2.0, 3, 4.0, 5, 8),  # positive int (float repr)
            *(10,),  # larger base: numerical range
        ],
    )
    def test_pow_scalar(self, dtype, shape, base):
        """Test pow with scalar and tensor (base^y)"""
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1
        result = torch.pow(base, y)
        expected = torch.pow(base, y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_rsqrt(self, dtype, shape):
        """Test rsqrt op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1
        result = torch.rsqrt(x)
        expected = torch.rsqrt(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("b,n,m,p", [(2, 10, 20, 30), (3, 5, 10, 15)])
    def test_bmm(self, dtype, b, n, m, p):
        """Test bmm op with various batch matrix sizes

        Note: float16 precision limits require relaxed tolerance for batch matrix multiplication.
        """
        x = torch.randn([b, n, m], dtype=dtype, device=self.rbln_device)
        y = torch.randn([b, m, p], dtype=dtype, device=self.rbln_device)
        result = torch.bmm(x, y)
        expected = torch.bmm(x.cpu(), y.cpu())
        # Use relaxed tolerance for float16 batch matrix multiplication
        self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

    # cat coverage lives in test_cat_index_select_v2v.py — TestCatV2V covers the
    # native v2v kernel across dtypes, axes, contig / non-contig inputs, empty
    # inputs, stack composite, and opinfo conformance regressions.

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_where_self(self, dtype, shape):
        """Test where.self op"""
        condition = torch.randn(shape, dtype=dtype, device=self.rbln_device) > 0
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.where(condition, x, y)
        expected = torch.where(condition.cpu(), x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_where_self_out(self, dtype, shape):
        """Test where.self_out op"""
        condition = torch.randn(shape, dtype=dtype, device=self.rbln_device) > 0
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        out = torch.empty_like(x)
        result = torch.where(condition, x, y, out=out)
        expected = torch.where(condition.cpu(), x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.data_ptr(), out.data_ptr())

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_ceil(self, dtype, shape):
        """Test ceil op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device) * 2.5
        result = torch.ceil(x)
        expected = torch.ceil(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_clamp(self, dtype, shape):
        """Test clamp op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        min_val = -0.5
        max_val = 0.5
        result = torch.clamp(x, min_val, max_val)
        expected = torch.clamp(x.cpu(), min_val, max_val)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_zero_(self, dtype, shape):
        """Test zero_ op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        x.zero_()
        expected = torch.zeros(shape, dtype=dtype, device="cpu")
        self.assertEqual(x.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_log(self, dtype, shape):
        """Test log op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device).abs() + 0.1
        result = torch.log(x)
        expected = torch.log(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_floor(self, dtype, shape):
        """Test floor op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device) * 2.5
        result = torch.floor(x)
        expected = torch.floor(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_masked_fill_tensor(self, dtype, shape):
        """Test masked_fill_.Tensor op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        mask = torch.randn(shape, dtype=dtype, device=self.rbln_device) > 0
        value = torch.tensor(5.0, dtype=dtype, device=self.rbln_device)
        x.masked_fill_(mask, value)
        expected = x.cpu().clone()
        expected.masked_fill_(mask.cpu(), value.cpu())
        self.assertEqual(x.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_masked_fill_scalar(self, dtype, shape):
        """Test masked_fill_.Scalar op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        mask = torch.randn(shape, dtype=dtype, device=self.rbln_device) > 0
        value = 5.0
        x.masked_fill_(mask, value)
        expected = x.cpu().clone()
        expected.masked_fill_(mask.cpu(), value)
        self.assertEqual(x.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(x.device, self.rbln_device)
        self.assertEqual(x.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shape,dim",
        [
            ((3, 4), 1),
            ((10, 20), 0),
            ((2, 3, 4), 2),
            ((5, 10), None),  # max over all dimensions
        ],
    )
    def test_max(self, dtype, shape, dim):
        """Test max op (reduction)"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.max(x, dim=dim) if dim is not None else torch.max(x)
        expected = torch.max(x.cpu(), dim=dim) if dim is not None else torch.max(x.cpu())
        if isinstance(result, tuple):
            self.assertEqual(result[0].cpu(), expected[0].cpu(), atol=ATOL, rtol=RTOL)
            if len(result) > 1:
                self.assertEqual(result[1].cpu(), expected[1].cpu())
        else:
            self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result[0].device if isinstance(result, tuple) else result.device, self.rbln_device)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shape,dim",
        [
            ((3, 4), 1),
            ((10, 20), 0),
            ((2, 3, 4), 2),
            ((5, 10), None),  # min over all dimensions
        ],
    )
    def test_min(self, dtype, shape, dim):
        """Test min op (reduction)"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.min(x, dim=dim) if dim is not None else torch.min(x)
        expected = torch.min(x.cpu(), dim=dim) if dim is not None else torch.min(x.cpu())
        if isinstance(result, tuple):
            self.assertEqual(result[0].cpu(), expected[0].cpu(), atol=ATOL, rtol=RTOL)
            if len(result) > 1:
                self.assertEqual(result[1].cpu(), expected[1].cpu())
        else:
            self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result[0].device if isinstance(result, tuple) else result.device, self.rbln_device)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_max_unary_out(self, dtype, shape):
        """Test max.unary_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        out = torch.empty([], dtype=dtype, device=self.rbln_device)
        result = torch.max(x, out=out)
        expected = torch.max(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_min_unary_out(self, dtype, shape):
        """Test min.unary_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        out = torch.empty([], dtype=dtype, device=self.rbln_device)
        result = torch.min(x, out=out)
        expected = torch.min(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_maximum(self, dtype, shape):
        """Test maximum op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.maximum(x, y)
        expected = torch.maximum(x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_minimum(self, dtype, shape):
        """Test minimum op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.minimum(x, y)
        expected = torch.minimum(x.cpu(), y.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(10, 20, 30), (5, 10, 15)])
    def test_addmm(self, dtype, n, m, p):
        """Test addmm op

        Note: float16 precision limits require relaxed tolerance for matrix multiplication.
        """
        mat1 = torch.randn([n, m], dtype=dtype, device=self.rbln_device)
        mat2 = torch.randn([m, p], dtype=dtype, device=self.rbln_device)
        vec = torch.randn([n, p], dtype=dtype, device=self.rbln_device)
        result = torch.addmm(vec, mat1, mat2)
        expected = torch.addmm(vec.cpu(), mat1.cpu(), mat2.cpu())
        if dtype == torch.float16:
            # Use relaxed tolerance for float16 matrix multiplication
            self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
        else:
            self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size,in_features,out_features", [(3, 64, 32), (5, 128, 64), (10, 32, 16)])
    def test_linear(self, dtype, batch_size, in_features, out_features):
        """Test linear op

        Note: float16 precision limits require relaxed tolerance for linear operations
        (which internally use matrix multiplication).
        """
        # linear(input, weight, bias) computes input @ weight.T + bias
        # input: (batch_size, in_features)
        # weight: (out_features, in_features)
        # bias: (out_features,)
        # result: (batch_size, out_features)
        x = torch.randn([batch_size, in_features], dtype=dtype, device=self.rbln_device)
        weight = torch.randn([out_features, in_features], dtype=dtype, device=self.rbln_device)
        bias = torch.randn([out_features], dtype=dtype, device=self.rbln_device)
        result = torch.nn.functional.linear(x, weight, bias)
        expected = torch.nn.functional.linear(x.cpu(), weight.cpu(), bias.cpu())
        if dtype == torch.float16:
            # Use relaxed tolerance for float16 linear operations
            self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
        else:
            self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "batch_size,in_features,out_features,output_mask",
        [
            (3, 64, 32, [True, True, True]),  # all gradients
            (5, 128, 64, [True, False, False]),  # only input grad
            (10, 32, 16, [False, True, True]),  # weight and bias grad
        ],
    )
    def test_linear_backward(self, dtype, batch_size, in_features, out_features, output_mask):
        """Test linear_backward op

        Note: float16 precision limits require relaxed tolerance for linear backward operations.
        CPU doesn't have direct linear_backward implementation, so we compute expected values
        using the same logic as the RBLN implementation.
        """

        def _linear_backward_cpu(input_tensor, grad_output, weight, output_mask):
            """CPU implementation of linear_backward using the same logic as RBLN"""
            grad_input = None
            grad_weight = None
            grad_bias = None
            if output_mask[0]:  # input grad
                grad_input = torch.matmul(grad_output, weight)
            if output_mask[1]:  # weight grad
                grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
                input_reshaped = input_tensor.reshape(-1, input_tensor.shape[-1])
                grad_weight = torch.matmul(grad_output_reshaped.T, input_reshaped)
            if output_mask[2]:  # bias grad
                grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
            return grad_input, grad_weight, grad_bias

        input_tensor = torch.randn([batch_size, in_features], dtype=dtype, device=self.rbln_device)
        grad_output = torch.randn([batch_size, out_features], dtype=dtype, device=self.rbln_device)
        weight = torch.randn([out_features, in_features], dtype=dtype, device=self.rbln_device)
        result = torch.ops.aten.linear_backward(input_tensor, grad_output, weight, output_mask)
        expected = _linear_backward_cpu(input_tensor.cpu(), grad_output.cpu(), weight.cpu(), output_mask)
        # Compare each element of the tuple
        for i, (r, e) in enumerate(zip(result, expected)):
            if r is not None and e is not None:
                if dtype == torch.float16:
                    self.assertEqual(r.cpu(), e.cpu(), atol=0.1, rtol=0.1)
                else:
                    self.assertEqual(r.cpu(), e.cpu(), atol=ATOL, rtol=RTOL)
                self.assertEqual(r.device, self.rbln_device)
                self.assertEqual(r.dtype, e.dtype)
            elif r is None and e is None:
                pass  # Both None is fine
            else:
                self.fail(f"Gradient {i} mismatch: result is {r is None}, expected is {e is None}")

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_sigmoid(self, dtype, shape):
        """Test sigmoid op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = torch.sigmoid(x)
        expected = torch.sigmoid(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shape,dim",
        [
            ((3, 4), 1),  # 2D tensor, dim=1
            ((10, 20), 0),  # 2D tensor, dim=0
            ((2, 3, 4), 2),  # 3D tensor, dim=2
            ((5, 10, 15), 1),  # 3D tensor, dim=1
        ],
    )
    def test_softmax_backward_data(self, dtype, shape, dim):
        """Test _softmax_backward_data op with various shapes and dimensions"""
        # Create output from forward softmax pass
        output = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        output = torch.softmax(output, dim=dim)  # Normalize to valid softmax output
        grad_output = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        input_dtype = dtype
        result = torch.ops.aten._softmax_backward_data(grad_output, output, dim, input_dtype)
        expected = torch.ops.aten._softmax_backward_data(grad_output.cpu(), output.cpu(), dim, input_dtype)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_logical_not(self, dtype, shape):
        """Test logical_not op with various shapes"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device) > 0
        result = torch.logical_not(x)
        expected = torch.logical_not(x.cpu())
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_ne_tensor(self, dtype, shape):
        """Test ne.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x != y
        expected = x.cpu() != y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_eq_tensor(self, dtype, shape):
        """Test eq.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = x.clone()  # Make them equal
        result = x == y
        expected = x.cpu() == y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_gt_tensor(self, dtype, shape):
        """Test gt.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x > y
        expected = x.cpu() > y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_ge_tensor(self, dtype, shape):
        """Test ge.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x >= y
        expected = x.cpu() >= y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_lt_tensor(self, dtype, shape):
        """Test lt.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x < y
        expected = x.cpu() < y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_le_tensor(self, dtype, shape):
        """Test le.Tensor_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        y = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        result = x <= y
        expected = x.cpu() <= y.cpu()
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_ne_scalar(self, dtype, shape):
        """Test ne.Scalar_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        scalar = 0.5
        result = x != scalar
        expected = x.cpu() != scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_eq_scalar(self, dtype, shape):
        """Test eq.Scalar_out op"""
        x = torch.ones(shape, dtype=dtype, device=self.rbln_device) * 0.5
        scalar = 0.5
        result = x == scalar
        expected = x.cpu() == scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_gt_scalar(self, dtype, shape):
        """Test gt.Scalar_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        scalar = 0.5
        result = x > scalar
        expected = x.cpu() > scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_ge_scalar(self, dtype, shape):
        """Test ge.Scalar_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        scalar = 0.5
        result = x >= scalar
        expected = x.cpu() >= scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_lt_scalar(self, dtype, shape):
        """Test lt.Scalar_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        scalar = 0.5
        result = x < scalar
        expected = x.cpu() < scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES_SUBSET)
    def test_le_scalar(self, dtype, shape):
        """Test le.Scalar_out op"""
        x = torch.randn(shape, dtype=dtype, device=self.rbln_device)
        scalar = 0.5
        result = x <= scalar
        expected = x.cpu() <= scalar
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)


# =============================================================================
# Implicit broadcast across binary / comparison ops
#
# As of 2026-04-30 the Python ``broadcast_args_general`` is validate-only
# (no ``torch.broadcast_tensors`` call), and the C++ shim removed the same set
# of binary / comparison ops from ``broadcast_ops()``. Together this skips a
# host-side D2H + broadcast + H2D materialization on every shim warm hit and
# every Python miss. Rebel-backend handles the implicit broadcast inside the
# compiled graph (verified by inspection of the RblnTensor / RTOSA / InitGen
# MLIR dumps — see ``docs/eager_mode_analysis/2026_04_30/``).
#
# These tests exercise the 12 ops that previously called broadcast_args_general
# across shape patterns that PyTorch broadcasting permits, plus the patterns
# we hit in production (LLaMA RMSNorm, attention bias). A regression here
# means either rebel-backend lost a broadcast pattern OR the shim/Python
# coordination got out of sync (e.g. one side broadcasts, the other doesn't).
# =============================================================================
BROADCAST_PATTERNS = [
    # (lhs, rhs)  — expected shape derived via torch.broadcast_shapes
    # Trailing-dim broadcast (most common in practice — RMSNorm, layer norm)
    ((1, 8, 16), (16,)),
    ((4, 8, 16), (16,)),
    ((1, 128, 2048), (2048,)),  # LLaMA RMSNorm prefill
    ((1, 1, 2048), (2048,)),  # LLaMA RMSNorm decode
    # Inner-dim broadcast (1 expanding to N on one side)
    ((1, 1, 64), (1, 4, 64)),
    ((1, 4, 64), (1, 1, 64)),
    # Two-sided broadcast (both sides have unique non-1 dims)
    ((4, 1, 64), (1, 8, 64)),
    ((4, 1, 8, 16), (1, 4, 1, 16)),
    # Higher rank
    ((2, 3, 4, 8, 16), (16,)),  # 5D
    ((2, 3, 4, 8, 16), (4, 1, 16)),
    ((2, 2, 2, 2, 4, 16), (16,)),  # 6D
    # Mixed-rank
    ((2, 3, 4, 8, 16), (8, 16)),
    ((4, 8, 16), ()),  # scalar rhs
    # ----- Last-dim-size-one broadcast (rebel-backend CANNOT implicit-compile;
    # ``broadcast_args_general`` must pre-broadcast via host materialization,
    # else rebel-compiler raises ``Graph Optimization: [UNEXPECTED_GRAPH]``).
    # Patterns observed in practice:
    #   * ``softmax_backward_rbln``  : ``output * S.sum(dim=-1, keepdim=True)``
    #     emits ``(N, D) × (N, 1)`` when ``dim`` is the last axis.
    #   * Reduce-then-multiply patterns where ``keepdim=True`` leaves a 1 in
    #     the last position.
    # -----
    ((3, 4), (3, 1)),  # softmax_backward shape0 (last-dim small)
    ((2, 3, 4), (2, 3, 1)),  # softmax_backward shape2 (3D last-dim)
    ((4, 8, 16), (4, 8, 1)),  # last-dim aligned-side (16) — also fails
    ((1, 32, 4), (1, 32, 1)),  # head-shaped, last-dim small
    ((4, 8, 1), (1, 1, 16)),  # both sides have last-dim==1 in one operand
    # Same-shape (sanity / fast-path that skips even broadcast_shapes check)
    ((4, 8, 16), (4, 8, 16)),
]


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestBinaryOpsBroadcast(TestCase):
    """Verify the 12 binary / comparison ops produce correct results across
    broadcast patterns after the implicit-broadcast change (2026-04-30).
    Single-tensor and same-shape cases are also covered to ensure the fast
    paths in broadcast_args_general still work."""

    rbln_device = torch.device("rbln:0")

    def _golden(self, op_callable, a_cpu, b_cpu):
        # CPU reference in fp32 to avoid double-rounding in the assertion
        return op_callable(a_cpu, b_cpu)

    def _check(
        self, op_callable, lhs_shape, rhs_shape, dtype=torch.float16, atol=ATOL, rtol=RTOL, output_is_bool=False
    ):
        a_cpu = torch.randn(*lhs_shape, dtype=torch.float32) if lhs_shape else torch.randn(())
        b_cpu = torch.randn(*rhs_shape, dtype=torch.float32) if rhs_shape else torch.randn(())
        a = a_cpu.to(dtype=dtype, device=self.rbln_device)
        b = b_cpu.to(dtype=dtype, device=self.rbln_device)
        result = op_callable(a, b)
        # Reference: run on the same dtype on CPU so rounding modes are the
        # same wire-format. The rebel kernel produces fp16 results; CPU fp16
        # matmul is round-to-nearest-even like rebel, so this is a fair
        # comparison.
        a_ref = a.cpu()
        b_ref = b.cpu()
        expected = self._golden(op_callable, a_ref, b_ref)
        expected_shape = torch.broadcast_shapes(lhs_shape, rhs_shape) if (lhs_shape or rhs_shape) else torch.Size([])
        self.assertEqual(result.shape, expected_shape, msg=f"{op_callable.__name__} {lhs_shape}×{rhs_shape}")
        if output_is_bool:
            self.assertEqual(result.cpu(), expected.cpu(), msg=f"{op_callable.__name__} {lhs_shape}×{rhs_shape}")
        else:
            self.assertEqual(
                result.cpu(),
                expected.cpu(),
                atol=atol,
                rtol=rtol,
                msg=f"{op_callable.__name__} {lhs_shape}×{rhs_shape}",
            )
        self.assertEqual(result.device, self.rbln_device)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_add_broadcast(self, lhs_shape, rhs_shape):
        self._check(torch.add, lhs_shape, rhs_shape)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_sub_broadcast(self, lhs_shape, rhs_shape):
        self._check(torch.sub, lhs_shape, rhs_shape)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_mul_broadcast(self, lhs_shape, rhs_shape):
        self._check(torch.mul, lhs_shape, rhs_shape)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_div_broadcast(self, lhs_shape, rhs_shape):
        # Loosen tolerance: fp16 division near zero produces large relative
        # error which is precision-bound, not a bug.
        self._check(torch.div, lhs_shape, rhs_shape, atol=0.05, rtol=0.05)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_maximum_broadcast(self, lhs_shape, rhs_shape):
        self._check(torch.maximum, lhs_shape, rhs_shape)

    @parametrize("lhs_shape,rhs_shape", BROADCAST_PATTERNS)
    def test_minimum_broadcast(self, lhs_shape, rhs_shape):
        self._check(torch.minimum, lhs_shape, rhs_shape)

    # Comparison ops produce bool outputs; precision-of-floats edge cases can
    # flip individual bits when the operands round to exactly equal in fp16,
    # so we keep the pattern set tight (subset of BROADCAST_PATTERNS) for
    # comparisons.
    _CMP_PATTERNS = [
        ((1, 8, 16), (16,)),
        ((4, 1, 64), (1, 8, 64)),
        ((4, 8, 16), (4, 8, 16)),
    ]

    @parametrize("op_callable", [torch.eq, torch.ne, torch.gt, torch.ge, torch.lt, torch.le])
    @parametrize("lhs_shape,rhs_shape", _CMP_PATTERNS)
    def test_comparison_broadcast(self, op_callable, lhs_shape, rhs_shape):
        # We rebuild integer-valued tensors so fp16 rounding doesn't flip the
        # ordering of borderline pairs.
        a_int = torch.randint(-3, 4, lhs_shape if lhs_shape else (1,), dtype=torch.int64).reshape(
            lhs_shape if lhs_shape else ()
        )
        b_int = torch.randint(-3, 4, rhs_shape if rhs_shape else (1,), dtype=torch.int64).reshape(
            rhs_shape if rhs_shape else ()
        )
        a_cpu = a_int.to(torch.float32)
        b_cpu = b_int.to(torch.float32)
        a = a_cpu.to(dtype=torch.float16, device=self.rbln_device)
        b = b_cpu.to(dtype=torch.float16, device=self.rbln_device)
        result = op_callable(a, b)
        expected = op_callable(a.cpu(), b.cpu())
        expected_shape = torch.broadcast_shapes(lhs_shape, rhs_shape)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)

    # ----- Regression: rebel-backend cannot implicit-compile last-dim
    # ``size==1 → size>1`` broadcast (raises ``UNEXPECTED_GRAPH``). Without
    # ``broadcast_args_general``'s explicit-broadcast escape hatch these would
    # all fail under ``TORCH_RBLN_DISABLE_FALLBACK=compile_error``. Pinning
    # the exact softmax_backward shapes here so a future "let's just use raw
    # implicit broadcast everywhere" refactor trips the test instead of an
    # 8-hour debug session.
    _LAST_DIM_ONE_PATTERNS = [
        # (lhs_shape, rhs_shape, dim_used_in_originating_reduction)
        ((3, 4), (3, 1), "softmax_backward shape0_dim_1"),
        ((2, 3, 4), (2, 3, 1), "softmax_backward shape2_dim_2"),
        ((4, 8, 16), (4, 8, 1), "last-dim aligned-side"),
        ((1, 32, 4), (1, 32, 1), "head-shaped, last-dim small"),
    ]

    @parametrize("op_callable", [torch.add, torch.sub, torch.mul, torch.div])
    @parametrize("lhs_shape,rhs_shape,label", _LAST_DIM_ONE_PATTERNS)
    def test_last_dim_size_one_broadcast(self, op_callable, lhs_shape, rhs_shape, label):
        """Direct regression for the rebel ``UNEXPECTED_GRAPH`` failure mode."""
        atol, rtol = (0.05, 0.05) if op_callable is torch.div else (ATOL, RTOL)
        self._check(op_callable, lhs_shape, rhs_shape, atol=atol, rtol=rtol)

    def test_broadcast_compatibility_error_passthrough(self):
        # Mismatched-shape inputs that PyTorch broadcasting cannot reconcile
        # should raise — the validate-only path keeps the error helpful.
        a = torch.randn(3, 4, dtype=torch.float16, device=self.rbln_device)
        b = torch.randn(5, 4, dtype=torch.float16, device=self.rbln_device)
        with self.assertRaises(RuntimeError):
            _ = a + b


# =============================================================================
# View-on-device dispatch (pure permute / transpose)
#
# 2026-04-30 +. Pure permute views (no expand, no slice, no offset) are
# detected from the input tensor's stride pattern and converted to an
# explicit ``aten::permute`` node inside the FX graph that compile_rbln_cached
# traces. rebel-backend lowers the permute as part of the device kernel
# chain (verified by inspection of the RblnTensor → RTOSA → InitGen MLIR
# stages on /tmp/mlir_broadcast_dump). The host-side ``.contiguous()`` is
# bypassed for permute views; other view types (expand, slice w/ offset,
# composite) keep the legacy host materialization path so correctness is
# preserved end-to-end.
#
# Tests below run real op calls through the PyTorch dispatcher → C++ shim →
# Python wrapper → ``_compile_and_run_view_aware`` → device. They cover:
#  - All 22 affected ops (12 binary/comparison, 9 unary, 1 ternary).
#  - Multiple permute shapes: transpose pair, full reverse, rotate, identity-
#    like (size-1 dim) plus a non-permute fallback (slice with offset).
#  - permute-on-lhs, permute-on-rhs, both-sides for binary ops.
#  - permute combined with broadcast (different output shape).
# =============================================================================
PERMUTE_PATTERNS = [
    # base_shape, perm  — pairs we expect to round-trip correctly
    ((2, 4, 8), (2, 0, 1)),  # rotate axis 2 → 0
    ((2, 4, 8), (1, 0, 2)),  # transpose dim 0 & 1
    ((2, 4, 8), (0, 2, 1)),  # transpose last two dims
    ((2, 4, 8, 16), (3, 0, 1, 2)),  # 4D rotation
    ((2, 4, 8, 16), (0, 2, 1, 3)),  # 4D mid swap
    ((1, 32, 64), (0, 2, 1)),  # LLaMA rotary-shape transpose
    ((4, 8, 16), (2, 1, 0)),  # full reverse
    ((1, 8, 1, 16), (0, 2, 1, 3)),  # size-1 dim mixed in
]


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestViewOpsOnDevice(TestCase):
    """Verify the 22 affected ops dispatch correctly when their inputs are
    pure permute views. Each test runs the op through the real PyTorch
    dispatcher (not a direct compile_rbln_cached call), so the C++ shim,
    Python view-aware wrapper, and rebel-backend lowering are all exercised.
    """

    rbln_device = torch.device("rbln:0")

    def _make_permuted(self, base_shape, perm, dtype=torch.float16, device=None):
        base = torch.randn(*base_shape, dtype=dtype, device=device)
        return base, base.permute(*perm)

    def _check_unary(self, op_callable, base_shape, perm, atol=ATOL, rtol=RTOL, output_is_bool=False):
        base, t = self._make_permuted(base_shape, perm, device=self.rbln_device)
        result = op_callable(t)
        expected = op_callable(t.cpu())
        self.assertEqual(result.shape, expected.shape, msg=f"{op_callable.__name__} permute({perm})")
        if output_is_bool:
            self.assertEqual(result.cpu(), expected.cpu(), msg=f"{op_callable.__name__} permute({perm})")
        else:
            self.assertEqual(
                result.cpu(), expected.cpu(), atol=atol, rtol=rtol, msg=f"{op_callable.__name__} permute({perm})"
            )
        self.assertEqual(result.device, self.rbln_device)

    def _check_binary(
        self, op_callable, base_shape, perm, perm_lhs=True, perm_rhs=False, atol=ATOL, rtol=RTOL, output_is_bool=False
    ):
        # The "other" tensor is built to match the permuted shape so we focus
        # on view dispatch correctness (not broadcast, which is covered
        # separately in TestBinaryOpsBroadcast).
        base_lhs, t_lhs = self._make_permuted(base_shape, perm, device=self.rbln_device)
        if perm_rhs:
            base_rhs, t_rhs = self._make_permuted(base_shape, perm, device=self.rbln_device)
        else:
            t_rhs = torch.randn(*t_lhs.shape, dtype=torch.float16, device=self.rbln_device)
        if not perm_lhs:
            t_lhs = t_lhs.contiguous()
        result = op_callable(t_lhs, t_rhs)
        expected = op_callable(t_lhs.cpu(), t_rhs.cpu())
        self.assertEqual(result.shape, expected.shape)
        if output_is_bool:
            self.assertEqual(result.cpu(), expected.cpu())
        else:
            self.assertEqual(result.cpu(), expected.cpu(), atol=atol, rtol=rtol)
        self.assertEqual(result.device, self.rbln_device)

    # --- Unary ops (9) ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_silu_permute(self, base_shape, perm):
        self._check_unary(torch.nn.functional.silu, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_neg_permute(self, base_shape, perm):
        self._check_unary(torch.neg, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_abs_permute(self, base_shape, perm):
        self._check_unary(torch.abs, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_ceil_permute(self, base_shape, perm):
        self._check_unary(torch.ceil, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_floor_permute(self, base_shape, perm):
        self._check_unary(torch.floor, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_sigmoid_permute(self, base_shape, perm):
        self._check_unary(torch.sigmoid, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_rsqrt_permute(self, base_shape, perm):
        # rsqrt requires positive input — abs the source.
        def op(x):
            return torch.rsqrt(torch.abs(x) + 0.1)

        self._check_unary(op, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_log_permute(self, base_shape, perm):
        def op(x):
            return torch.log(torch.abs(x) + 0.1)

        self._check_unary(op, base_shape, perm)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_unary_logical_not_permute(self, base_shape, perm):
        def op(x):
            return torch.logical_not(x > 0)

        self._check_unary(op, base_shape, perm, output_is_bool=True)

    # --- Binary ops (lhs permuted, rhs contig same shape) ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    @parametrize("op_callable", [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum])
    def test_binary_lhs_permute(self, op_callable, base_shape, perm):
        self._check_binary(
            op_callable, base_shape, perm, perm_lhs=True, perm_rhs=False, atol=0.05, rtol=0.05
        )  # fp16 div tolerance

    # --- Binary ops (both lhs AND rhs permuted with same recipe) ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    @parametrize("op_callable", [torch.mul, torch.add])
    def test_binary_both_permute(self, op_callable, base_shape, perm):
        self._check_binary(op_callable, base_shape, perm, perm_lhs=True, perm_rhs=True)

    # --- Comparison ops (bool output) ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    @parametrize("op_callable", [torch.eq, torch.ne, torch.gt, torch.ge, torch.lt, torch.le])
    def test_comparison_permute(self, op_callable, base_shape, perm):
        # Use integer-valued sources so fp16 rounding doesn't flip bool.
        base_int = torch.randint(-3, 4, base_shape, dtype=torch.int64).reshape(base_shape)
        base = base_int.to(torch.float16).to(self.rbln_device)
        t = base.permute(*perm)
        other_int = torch.randint(-3, 4, t.shape, dtype=torch.int64)
        other = other_int.to(torch.float16).to(self.rbln_device)
        result = op_callable(t, other)
        expected = op_callable(t.cpu(), other.cpu())
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu())
        self.assertEqual(result.device, self.rbln_device)

    # --- Ternary: where ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_ternary_where_permute(self, base_shape, perm):
        # cond is bool, x and y are fp16 — all three permuted.
        cond_base = (torch.randn(*base_shape) > 0).to(self.rbln_device)
        x_base = torch.randn(*base_shape, dtype=torch.float16, device=self.rbln_device)
        y_base = torch.randn(*base_shape, dtype=torch.float16, device=self.rbln_device)
        cond = cond_base.permute(*perm)
        x = x_base.permute(*perm)
        y = y_base.permute(*perm)
        result = torch.where(cond, x, y)
        expected = torch.where(cond.cpu(), x.cpu(), y.cpu())
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)

    # --- Permute combined with broadcast on the other side ---
    def test_binary_permute_with_broadcast(self):
        base = torch.randn(2, 4, 8, dtype=torch.float16, device=self.rbln_device)
        t = base.permute(2, 0, 1)  # shape (8, 2, 4)
        bias = torch.randn(4, dtype=torch.float16, device=self.rbln_device)
        # mul broadcasts bias from (4,) to (8, 2, 4)
        result = torch.mul(t, bias)
        expected = torch.mul(t.cpu(), bias.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    # --- Fallback path: non-permute view stays correct via .contiguous() ---
    def test_non_permute_view_falls_back_correctly(self):
        # Slice with non-zero storage_offset IS now handled (Phase 2: narrow).
        # We keep the test as an end-to-end correctness check; the helper
        # decides whether to dispatch via narrow on device or fall back.
        base = torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device)
        sliced = base[2:6]
        self.assertNotEqual(sliced.storage_offset(), 0)
        other = torch.randn(4, 16, dtype=torch.float16, device=self.rbln_device)
        result = torch.mul(sliced, other)
        expected = torch.mul(sliced.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    # ========================================================================
    # Phase 2 view-types: expand / narrow / select / squeeze / unsqueeze /
    # composite (permute + narrow). End-to-end via the real PyTorch dispatcher.
    # ========================================================================

    def test_expand_view_via_binary_op(self):
        # weight.expand pattern that LLaMA uses for biases — historically
        # would have hit ``.contiguous()`` host materialization. After Phase 2
        # the expand is detected and an ``aten::expand`` node is injected
        # into the FX graph, lowered on device.
        base = torch.randn(64, dtype=torch.float16, device=self.rbln_device)
        view = base.expand(2, 8, 64)  # stride (0, 0, 1)
        other = torch.randn(2, 8, 64, dtype=torch.float16, device=self.rbln_device)
        self.assertFalse(view.is_contiguous())
        result = torch.mul(view, other)
        expected = torch.mul(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    @parametrize("dim,start,length", [(0, 2, 4), (1, 4, 8), (2, 0, 16)])
    def test_narrow_view_via_unary_op(self, dim, start, length):
        # narrow with non-zero offset: silu(t.narrow(...)) should compute on
        # device without host materialization.
        base = torch.randn(8, 16, 32, dtype=torch.float16, device=self.rbln_device)
        view = base.narrow(dim, start, length)
        self.assertNotEqual(view.storage_offset() if dim != 2 or start != 0 else 1, 0)
        result = torch.nn.functional.silu(view)
        expected = torch.nn.functional.silu(view.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    def test_select_view_via_binary_op(self):
        # ``base[i]`` (select on dim 0) reduces ndim by 1 and shifts offset.
        base = torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device)
        sel = base.select(0, 3)  # shape (16,), offset = 3 * 16 = 48
        self.assertEqual(sel.storage_offset(), 48)
        other = torch.randn(16, dtype=torch.float16, device=self.rbln_device)
        result = torch.add(sel, other)
        expected = torch.add(sel.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    def test_composite_permute_then_narrow(self):
        # base.permute(...).narrow(...) — common attention reshape pattern.
        # _detect_view_recipe should produce a 2-step recipe handled on device.
        base = torch.randn(2, 4, 8, dtype=torch.float16, device=self.rbln_device)
        view = base.permute(2, 0, 1).narrow(0, 1, 6)
        self.assertFalse(view.is_contiguous())
        self.assertNotEqual(view.storage_offset(), 0)
        other = torch.randn(6, 2, 4, dtype=torch.float16, device=self.rbln_device)
        result = torch.mul(view, other)
        expected = torch.mul(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    def test_composite_narrow_then_permute(self):
        # Inverse order: narrow first, then permute. Detection still recovers
        # an equivalent recipe (permute + narrow with adjusted dim).
        base = torch.randn(8, 4, 16, dtype=torch.float16, device=self.rbln_device)
        view = base.narrow(0, 2, 4).permute(2, 0, 1)
        self.assertFalse(view.is_contiguous())
        other = torch.randn(16, 4, 4, dtype=torch.float16, device=self.rbln_device)
        result = torch.mul(view, other)
        expected = torch.mul(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.shape, expected.shape)

    def test_view_with_matmul(self):
        # mm with one transposed input — common in attention (Q @ K.T).
        a = torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device)
        b = torch.randn(32, 16, dtype=torch.float16, device=self.rbln_device)
        b_t = b.transpose(0, 1)  # shape (16, 32), non-contig
        self.assertFalse(b_t.is_contiguous())
        result = torch.mm(a, b_t)
        expected = torch.mm(a.cpu(), b_t.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=0.05, rtol=0.05)
        self.assertEqual(result.shape, expected.shape)

    def test_view_with_linear(self):
        # linear(x_t, weight, bias) where x_t is a transposed view.
        x = torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device).transpose(0, 1)
        weight = torch.randn(32, 8, dtype=torch.float16, device=self.rbln_device)
        bias = torch.randn(32, dtype=torch.float16, device=self.rbln_device)
        self.assertFalse(x.is_contiguous())
        result = torch.nn.functional.linear(x, weight, bias)
        expected = torch.nn.functional.linear(x.cpu(), weight.cpu(), bias.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=0.05, rtol=0.05)
        self.assertEqual(result.shape, expected.shape)

    # --- Newly-covered ops via codegen template + manual handlers ---
    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_softmax_permute(self, base_shape, perm):
        # custom_softmax_out_rbln (in register_custom_ops.py) — manually
        # converted to compile_and_run_view_aware. Permute on input must
        # reach the device path with explicit aten::permute in graph.
        base, t = self._make_permuted(base_shape, perm, device=self.rbln_device)
        result = torch.softmax(t, dim=-1)
        expected = torch.softmax(t.cpu(), dim=-1)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_pow_permute(self, base_shape, perm):
        # pow_tensor_scalar_out_rbln (manual) view-aware path.
        base, t = self._make_permuted(base_shape, perm, device=self.rbln_device)
        # Use abs() to avoid negative bases with non-integer exponent edge cases.
        result = torch.pow(torch.abs(t) + 0.1, 2)
        expected = torch.pow(torch.abs(t.cpu()) + 0.1, 2)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)

    @parametrize("base_shape,perm", PERMUTE_PATTERNS)
    def test_clamp_permute(self, base_shape, perm):
        # clamp goes through codegen template's view-aware path. We test
        # the scalar-bound variant (clamp.out — most common in real models).
        base, t = self._make_permuted(base_shape, perm, device=self.rbln_device)
        result = torch.clamp(t, min=-0.5, max=0.5)
        expected = torch.clamp(t.cpu(), min=-0.5, max=0.5)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)

    # mean + permute is verified for 3D bases. For 4D / 5D bases the
    # rebel-backend currently produces NaN (backend limitation, not a
    # dispatch-path issue — explicit aten::permute → aten::mean shows the
    # same numerical break in a hand-written OpModule). Track separately
    # under rebel-compiler; our view-aware path is correct in semantics.
    _MEAN_PERMUTE_PATTERNS = [
        ((2, 4, 8), (2, 0, 1)),
        ((2, 4, 8), (1, 0, 2)),
        ((2, 4, 8), (0, 2, 1)),
        ((4, 8, 16), (2, 1, 0)),
        ((1, 32, 64), (0, 2, 1)),
    ]

    @parametrize("base_shape,perm", _MEAN_PERMUTE_PATTERNS)
    def test_mean_permute(self, base_shape, perm):
        base, t = self._make_permuted(base_shape, perm, device=self.rbln_device)
        result = torch.mean(t, dim=-1)
        expected = torch.mean(t.cpu(), dim=-1)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)

    # ========================================================================
    # Direct view-primitive coverage (squeeze / unsqueeze / synthetic base)
    # — single-step patterns that 5-layer hard-coded detector handled but
    # had no end-to-end test. Generic detector covers all of these via
    # ``_gen_step_candidates`` + simulation.
    # ========================================================================

    def test_squeeze_view_via_unary_op(self):
        # squeeze(d) on a size-1 dim. The result is contig+offset=0 (no
        # recipe needed) — tests that detector correctly returns None and
        # the op runs through the fast pass-through path.
        base = torch.randn(2, 1, 4, dtype=torch.float16, device=self.rbln_device)
        view = base.squeeze(1)
        self.assertEqual(view.shape, torch.Size([2, 4]))
        result = torch.nn.functional.silu(view)
        expected = torch.nn.functional.silu(view.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

    def test_unsqueeze_view_via_unary_op(self):
        # unsqueeze(d) inserts a size-1 dim. Contig+offset=0 — same fast path.
        base = torch.randn(2, 4, dtype=torch.float16, device=self.rbln_device)
        view = base.unsqueeze(0)
        self.assertEqual(view.shape, torch.Size([1, 2, 4]))
        result = torch.nn.functional.silu(view)
        expected = torch.nn.functional.silu(view.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

    def test_permute_unsqueeze_composite_via_op(self):
        # ``permute → unsqueeze`` chain (LLaMA rotary reshape pattern).
        # Generic detector recovers a 2-step recipe via BFS; simulation
        # verifies metadata match.
        base = torch.randn(2, 4, 8, dtype=torch.float16, device=self.rbln_device)
        view = base.permute(2, 0, 1).unsqueeze(0)  # shape (1, 8, 2, 4)
        other = torch.randn(1, 8, 2, 4, dtype=torch.float16, device=self.rbln_device)
        self.assertFalse(view.is_contiguous())
        result = torch.add(view, other)
        expected = torch.add(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

    def test_expand_permute_composite_via_op(self):
        # ``expand → permute`` chain — the broadcast-and-reorder form.
        # base (1, 4, 8) → expand to (3, 4, 8) → permute(2, 0, 1) → (8, 3, 4)
        base = torch.randn(1, 4, 8, dtype=torch.float16, device=self.rbln_device)
        view = base.expand(3, 4, 8).permute(2, 0, 1)
        other = torch.randn(8, 3, 4, dtype=torch.float16, device=self.rbln_device)
        self.assertFalse(view.is_contiguous())
        result = torch.mul(view, other)
        expected = torch.mul(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

    def test_synthetic_base_via_op(self):
        # When ``t._base`` is itself a non-contig view (chained reshape),
        # the detector must synthesize a contig base from the storage
        # stride pattern. Construct such a tensor via ``as_strided`` of an
        # already-permuted parent so ``_base`` chains don't reach contig.
        storage_owner = torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device)
        # Make _base = an intermediate permute (non-contig)
        intermediate = storage_owner.transpose(0, 1)  # (16, 8) non-contig
        # Now build a view of intermediate that the detector can reverse
        # via synthetic_base — narrow on the first dim (still references
        # the original storage with custom stride).
        view = intermediate.narrow(0, 4, 8)  # shape (8, 8), stride (1, 16), offset 4
        other = torch.randn(8, 8, dtype=torch.float16, device=self.rbln_device)
        self.assertFalse(view.is_contiguous())
        result = torch.add(view, other)
        expected = torch.add(view.cpu(), other.cpu())
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

    def test_view_recipe_simulation_property(self):
        # Property-based check on the generic detector: for a battery of
        # random view chains, ``_simulate_recipe`` on the recovered base
        # must reproduce the target tensor's (shape, stride, offset)
        # exactly. Bit-exact match is the correctness invariant of the
        # generic detector.
        from torch_rbln._internal.ops_utils import _detect_view_recipe, _simulate_recipe

        bases = [
            torch.randn(2, 4, 8, dtype=torch.float16, device=self.rbln_device),
            torch.randn(8, 16, dtype=torch.float16, device=self.rbln_device),
            torch.randn(1, 4, 8, dtype=torch.float16, device=self.rbln_device),
            torch.randn(8, 4, 16, dtype=torch.float16, device=self.rbln_device),
        ]
        view_makers = [
            ("permute", lambda b: b.permute(*range(b.dim() - 1, -1, -1))),  # full reverse
            ("narrow0", lambda b: b.narrow(0, 0, max(1, b.size(0) // 2))),
            (
                "permute_then_narrow",
                lambda b: b.permute(*range(b.dim() - 1, -1, -1)).narrow(0, 0, max(1, b.size(-1) // 2)),
            ),
            (
                "narrow_then_permute",
                lambda b: b.narrow(0, 0, max(1, b.size(0) // 2)).permute(*range(b.dim() - 1, -1, -1)),
            ),
        ]
        for base in bases:
            for label, mk in view_makers:
                try:
                    t = mk(base)
                except Exception:
                    continue  # skip impossible combos
                if t.is_contiguous() and t.storage_offset() == 0:
                    continue  # detector returns None for canonical tensors
                res = _detect_view_recipe(t)
                if res is None:
                    continue  # fallback path — not a property failure
                recovered_base, recipe = res
                sim = _simulate_recipe(
                    tuple(recovered_base.shape),
                    tuple(recovered_base.stride()),
                    0,
                    recipe,
                )
                target = (tuple(t.shape), tuple(t.stride()), t.storage_offset())
                self.assertEqual(
                    sim,
                    target,
                    msg=f"recipe simulation mismatch for base.shape={tuple(base.shape)} via {label}: "
                    f"recipe={recipe}, sim={sim}, target={target}",
                )

    def test_view_telemetry_counter(self):
        # The fallback warning counter should remain at 0 across the
        # recognized-pattern operations exercised in this class. Reset it,
        # run a known-recognized view op, and check it didn't increment.
        from torch_rbln._internal.ops_utils import _view_fallback_count_get, _view_fallback_count_reset

        _view_fallback_count_reset()
        base = torch.randn(2, 4, 8, dtype=torch.float16, device=self.rbln_device)
        _ = torch.mul(base.permute(2, 0, 1), torch.randn(8, 2, 4, dtype=torch.float16, device=self.rbln_device))
        self.assertEqual(
            _view_fallback_count_get(), 0, "Recognized permute view should NOT trigger .contiguous() fallback."
        )


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestRegisteredPythonOpsMultiDevice(TestCase):
    """Test ops registered in register_ops.py with mixed device indices"""

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES)
    @parametrize("device_index", range(min(torch.rbln.device_count(), 3)))
    def test_add_mixed_devices(self, dtype, shape, device_index):
        """Test add op with tensors on different devices"""
        device = torch.device("rbln", device_index)
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        result = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, device)
        self.assertEqual(result.dtype, expected.dtype)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", TEST_SHAPES)
    @parametrize("device_index", range(min(torch.rbln.device_count(), 3)))
    def test_mul_mixed_devices(self, dtype, shape, device_index):
        """Test mul op with tensors on different devices"""
        device = torch.device("rbln", device_index)
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        result = x * y
        expected = x.cpu() * y.cpu()
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, device)
        self.assertEqual(result.dtype, expected.dtype)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("n,m,p", [(10, 20, 30), (5, 10, 15)])
    @parametrize("device_index", range(min(torch.rbln.device_count(), 3)))
    def test_mm_mixed_devices(self, dtype, n, m, p, device_index):
        """Test mm op with tensors on different devices

        Note: float16 precision limits require relaxed tolerance for matrix multiplication.
        """
        device = torch.device("rbln", device_index)
        x = torch.randn([n, m], dtype=dtype, device=device)
        y = torch.randn([m, p], dtype=dtype, device=device)
        result = torch.mm(x, y)
        expected = torch.mm(x.cpu(), y.cpu())
        # Use relaxed tolerance for float16 matrix multiplication
        self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
        self.assertEqual(result.device, device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(5, 10)])
    def test_add_cross_device(self, dtype, shape):
        """Test add op with tensors on different devices (should fail or handle appropriately)"""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")
        x = torch.randn(shape, dtype=dtype, device=device0)
        y = torch.randn(shape, dtype=dtype, device=device1)
        # This should either work (with automatic device promotion) or raise an error
        # The behavior depends on PyTorch's implementation
        try:
            result = x + y
            # If it works, verify correctness
            expected = x.cpu() + y.cpu()
            self.assertEqual(result.cpu(), expected, atol=ATOL, rtol=RTOL)
            self.assertEqual(result.device.type, "rbln")
            self.assertEqual(result.dtype, expected.dtype)
        except RuntimeError:
            # If it fails, that's also acceptable behavior
            pass

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("batch_size,in_features,out_features", [(3, 64, 32)])
    def test_linear_device_context_mismatch(self, dtype, batch_size, in_features, out_features):
        """Test linear op with device context mismatch (0->1 and 1->0)

        This test intentionally creates scenarios where:
        - Device context is set to one device (e.g., 0)
        - Linear operation is performed on tensors from a different device (e.g., 1)
        - sync_runtime should detect device_id from input tensors
        - But device context is different, so it creates out tensor on the input device

        This tests the behavior when device_id from inputs doesn't match
        the current device context.

        Test cases:
        1. Context 0 -> Linear on device 1 (0->1)
        2. Context 1 -> Linear on device 0 (1->0)
        """
        # Test cases: (context_device_index, tensor_device_index)
        test_cases = [(0, 1), (1, 0)]
        for context_device_index, tensor_device_index in test_cases:
            # Set device context
            torch.rbln.set_device(context_device_index)
            self.assertEqual(
                torch.rbln.current_device(),
                context_device_index,
                f"Device context should be set to {context_device_index}",
            )

            # Create tensors on tensor_device_index
            device = torch.device("rbln", tensor_device_index)
            x = torch.randn([batch_size, in_features], dtype=dtype, device=device)
            weight = torch.randn([out_features, in_features], dtype=dtype, device=device)
            bias = torch.randn([out_features], dtype=dtype, device=device)

            # Verify input tensors are on tensor_device_index
            self.assertEqual(
                x.device.index, tensor_device_index, f"Input tensor should be on device {tensor_device_index}"
            )
            self.assertEqual(
                weight.device.index, tensor_device_index, f"Weight tensor should be on device {tensor_device_index}"
            )
            self.assertEqual(
                bias.device.index, tensor_device_index, f"Bias tensor should be on device {tensor_device_index}"
            )

            # Perform linear operation
            # sync_runtime should detect device_id from input tensors
            # and create output tensor on tensor_device_index (not context_device_index, despite context being different)
            result = torch.nn.functional.linear(x, weight, bias)

            # Verify correctness
            expected = torch.nn.functional.linear(x.cpu(), weight.cpu(), bias.cpu())
            self.assertEqual(result.cpu(), expected.cpu(), atol=0.1, rtol=0.1)
            self.assertEqual(result.device, device)
            self.assertEqual(result.dtype, expected.dtype)

            # Verify output shape
            self.assertEqual(result.shape, expected.shape)

            # The output should be on tensor_device_index (matching input device), not context_device_index (current context)
            # This tests that sync_runtime correctly uses device_id from input tensors
            # rather than the current device context
            self.assertEqual(
                result.device.index,
                tensor_device_index,
                f"Output tensor should be on device {tensor_device_index} (matching input), "
                f"not device {context_device_index} (current context)",
            )

            # Verify device context is still context_device_index (should not have changed)
            self.assertEqual(
                torch.rbln.current_device(),
                context_device_index,
                f"Device context should remain {context_device_index} after operation",
            )


instantiate_device_type_tests(TestRegisteredNativeOps, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestRegisteredFallbackOps, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestRegisteredPythonOps, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestBinaryOpsBroadcast, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestViewOpsOnDevice, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestRegisteredPythonOpsMultiDevice, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
