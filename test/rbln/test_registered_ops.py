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

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize(
        "shapes,dim,expected_shape",
        [
            ([(3, 4), (3, 4)], 0, (6, 4)),
            ([(3, 4), (3, 4)], 1, (3, 8)),
            ([(2, 3, 4), (2, 3, 4)], 0, (4, 3, 4)),
            ([(5, 10), (5, 10), (5, 10)], 1, (5, 30)),
        ],
    )
    def test_cat(self, dtype, shapes, dim, expected_shape):
        """Test cat op with various shapes"""
        tensors = [torch.randn(shape, dtype=dtype, device=self.rbln_device) for shape in shapes]
        result = torch.cat(tensors, dim=dim)
        expected = torch.cat([t.cpu() for t in tensors], dim=dim)
        self.assertEqual(result.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)
        self.assertEqual(result.device, self.rbln_device)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.shape, expected.shape)

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
instantiate_device_type_tests(TestRegisteredPythonOpsMultiDevice, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
