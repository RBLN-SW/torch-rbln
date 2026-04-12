# Owner(s): ["module: PrivateUse1"]

"""
Test suite for tensor copy operations.
"""

import math

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import DEFAULT_ATOL, DEFAULT_RTOL, SUPPORTED_DTYPES


ATOL = DEFAULT_ATOL
RTOL = DEFAULT_RTOL


TEST_DTYPES = SUPPORTED_DTYPES + [torch.float32]

COPY_TYPES = {
    "h2d": ("cpu", "rbln"),
    "d2h": ("rbln", "cpu"),
    "d2d": ("rbln", "rbln"),
}

TEST_SHAPES = [
    (2, 5),  # ≈10  (2-D)
    (4, 5, 5),  # ≈100 (3-D)
    (5, 10, 10, 2),  # ≈1000(4-D+)
]


def _make_safe_noncontig_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # 1) make row-major stride
    contig_stride = list(torch.empty(0, dtype=dtype, device=device).new_empty(shape).stride())

    # 2) Add 1 to the stride of the 'first axis with size greater than 1' for padding
    # → For (2,5), the 0th axis is selected so that it looks like (6,1)
    tgt_dim = next(i for i, sz in enumerate(shape) if sz > 1)
    contig_stride[tgt_dim] += 1
    stride = tuple(contig_stride)

    # 3) storage size actually needed = max(offset) + 1
    storage_size = 1 + sum((sz - 1) * st for sz, st in zip(shape, stride))
    buf = torch.empty(storage_size, dtype=dtype, device=device)

    # 4) make non-contiguous view
    view = torch.as_strided(buf, size=shape, stride=stride)

    # 5) fill values (it probably make 1d-storage with reshape_())
    view.copy_(torch.arange(view.numel(), dtype=dtype, device=device).view(shape))

    assert not view.is_contiguous()
    return view


def _make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    is_contiguous: bool,
) -> torch.Tensor:
    numel = math.prod(shape)
    base = torch.arange(numel, dtype=dtype).reshape(shape).to(device)

    if is_contiguous:
        t = base.contiguous()
    else:
        t = _make_safe_noncontig_tensor(base.shape, base.dtype, base.device)
        t.copy_(base)

    assert t.is_contiguous() == is_contiguous
    return t


@pytest.mark.test_set_ci
class TestTensorCopy(TestCase):
    rbln_device = torch.device("rbln:0")

    @parametrize("shape", TEST_SHAPES)
    @parametrize("src_device_type,dst_device_type", COPY_TYPES.values())
    @parametrize("src_dtype", TEST_DTYPES)
    @parametrize("dst_dtype", TEST_DTYPES)
    @parametrize("src_is_contiguous", [True, False])
    @parametrize("dst_is_contiguous", [True, False])
    def test_tensor_copy_kernel(
        self, shape, src_device_type, dst_device_type, src_dtype, dst_dtype, src_is_contiguous, dst_is_contiguous
    ):
        src_device = torch.device(src_device_type)
        src = _make_tensor(shape, src_dtype, src_device, src_is_contiguous)

        dst_device = torch.device(dst_device_type)
        dst = _make_tensor(shape, dst_dtype, dst_device, dst_is_contiguous).zero_()

        dst.copy_(src)

        self.assertEqual(dst.cpu(), src.cpu().to(dst_dtype), atol=ATOL, rtol=RTOL)
        self.assertEqual(dst.is_contiguous(), dst_is_contiguous)

    @dtypes(*TEST_DTYPES)
    def test_copy_from_tensor_with_nonzero_storage_offset(self, dtype):
        base_shape = (6, 12)
        view_slice = (slice(2, 4), slice(3, 8))

        # 1. Create a base tensor and fill with known pattern
        base_tensor = _make_tensor(base_shape, dtype, self.rbln_device, True)
        base_tensor.copy_(torch.arange(base_tensor.numel(), dtype=dtype).view(base_shape))

        # 2. Take a view (storage_offset > 0 expected)
        view_tensor = base_tensor[view_slice]
        self.assertFalse(view_tensor.is_contiguous(), "View should not be contiguous")
        self.assertGreater(view_tensor.storage_offset(), 0, "View must have non-zero storage offset")

        # 3. Create contiguous destination and copy
        dst_tensor = torch.empty_like(view_tensor).contiguous()
        dst_tensor.copy_(view_tensor)

        # 4. Verify correctness
        expected = base_tensor.cpu()[view_slice]
        actual = dst_tensor.cpu()

        self.assertEqual(
            actual, expected, atol=ATOL, rtol=RTOL, msg="Mismatch when copying from non-zero offset tensor"
        )

    @parametrize("src_dtype", TEST_DTYPES)
    @parametrize("dst_dtype", TEST_DTYPES)
    def test_partial_copy_between_rbln_sliced(self, src_dtype, dst_dtype):
        dst_full_shape = (4, 10)
        src_shape = (2, 5)
        dst_slice_indices = (slice(1, 1 + src_shape[0]), slice(1, 1 + src_shape[1]))

        initial_fill_val = torch.tensor(7.0, dtype=dst_dtype)

        # ==== CASE 1: Contiguous → Non-contiguous ====
        src_rbln = _make_tensor(src_shape, src_dtype, self.rbln_device, True)
        expected_src = torch.arange(math.prod(src_shape), dtype=src_dtype).view(src_shape)

        dst_rbln = _make_tensor(dst_full_shape, dst_dtype, self.rbln_device, True)
        if dst_rbln.numel() > 0:
            dst_rbln.fill_(initial_fill_val)

        expected_dst_before = torch.full(dst_full_shape, initial_fill_val, dtype=dst_dtype)

        # Slice view into dst_rbln
        offset = dst_rbln.storage_offset()
        offset += dst_slice_indices[0].start * dst_rbln.stride(0)
        offset += dst_slice_indices[1].start * dst_rbln.stride(1)
        dst_view_rbln = dst_rbln.as_strided(size=src_shape, stride=dst_rbln.stride(), storage_offset=offset)

        # Sanity check shapes of the view and source
        self.assertEqual(dst_view_rbln.shape, src_rbln.shape, "Shape mismatch between dst view and src")
        dst_view_rbln.copy_(src_rbln)

        expected_dst_after = expected_dst_before.clone()
        expected_dst_after[dst_slice_indices] = expected_src
        actual_dst_after = dst_rbln.cpu()
        self.assertEqual(
            actual_dst_after, expected_dst_after, atol=ATOL, rtol=RTOL, msg="Mismatch after contig → non-contig copy"
        )

        if dst_rbln.numel() > 1:
            self.assertFalse(dst_view_rbln.is_contiguous(), "RBLN destination tensor should remain non-contiguous")

        # ==== CASE 2-1: Non-contiguous → Contiguous ====
        dst_rbln = _make_tensor(dst_full_shape, dst_dtype, self.rbln_device, True)
        src_noncontig = dst_rbln[dst_slice_indices]  # non-contiguous source slice
        dst_contig = _make_tensor(src_shape, dst_dtype, self.rbln_device, True)
        dst_contig.zero_()

        # Sanity check shapes of the view and source
        self.assertEqual(src_noncontig.shape, dst_contig.shape, "Shape mismatch between dst view and src")

        dst_contig.copy_(src_noncontig)

        actual_dst_contig_cpu = dst_contig.cpu()
        expected_from_src = src_noncontig.cpu().contiguous()
        self.assertEqual(
            actual_dst_contig_cpu,
            expected_from_src,
            atol=ATOL,
            rtol=RTOL,
            msg="Mismatch after non-contig → contig copy",
        )

        if dst_rbln.numel() > 1:
            self.assertFalse(src_noncontig.is_contiguous(), "src_noncontig should remain non-contiguous")

        # ==== CASE 2-2: Non-contiguous → Contiguous ====
        dst_rbln = _make_tensor(dst_full_shape, dst_dtype, self.rbln_device, True)
        src_noncontig = dst_rbln[dst_slice_indices]  # non-contiguous source slice
        dst_contig = _make_tensor(src_shape, dst_dtype, self.rbln_device, True)
        dst_contig.zero_()

        # Sanity check shapes of the view and source
        self.assertEqual(src_noncontig.shape, dst_contig.shape, "Shape mismatch between dst view and src")

        dst_contig.copy_(src_noncontig)

        actual_dst_contig_cpu = dst_contig.cpu()
        expected_from_src = src_noncontig.cpu().contiguous()
        self.assertEqual(
            actual_dst_contig_cpu,
            expected_from_src,
            atol=ATOL,
            rtol=RTOL,
            msg="Mismatch after non-contig → contig copy",
        )

        if dst_rbln.numel() > 1:
            self.assertFalse(src_noncontig.is_contiguous(), "src_noncontig should remain non-contiguous")

        # ==== CASE 3: Non-contiguous → Non-contiguous ====
        src_nc_slice = (slice(1, 3), slice(1, 6))  # Same position as original
        dst_nc_slice = (slice(0, 2), slice(0, 5))  # Top-left corner

        src_rbln = _make_tensor(dst_full_shape, src_dtype, self.rbln_device, True)
        dst_rbln = _make_tensor(dst_full_shape, dst_dtype, self.rbln_device, True)
        if dst_rbln.numel() > 0:
            dst_rbln.fill_(initial_fill_val)

        src_nc = src_rbln[src_nc_slice]
        dst_nc = dst_rbln[dst_nc_slice]

        # Copy from one non-contig slice to another
        self.assertEqual(src_nc.shape, dst_nc.shape, "Shape mismatch for non-contig to non-contig copy")

        dst_nc.copy_(src_nc)

        expected_dst = torch.full(dst_full_shape, initial_fill_val, dtype=dst_dtype)
        expected_src = (src_rbln.cpu())[src_nc_slice]
        expected_dst[dst_nc_slice] = expected_src
        self.assertEqual(
            dst_rbln.cpu(),
            expected_dst,
            atol=ATOL,
            rtol=RTOL,
            msg="Mismatch in full tensor after non-contig → non-contig copy",
        )

        if dst_rbln.numel() > 1:
            self.assertFalse(src_nc.is_contiguous(), "src_nc should remain non-contiguous")
            self.assertFalse(dst_nc.is_contiguous(), "dst_nc should remain non-contiguous")

    def test_partial_copy_with_diff_dtype(self):
        # case 1: partial update float16 to custom_float16
        a = torch.tensor([1, 2, 3], dtype=torch.float16, device=self.rbln_device)
        b = torch.tensor([2, 3, 4], dtype=torch.float16, device=self.rbln_device)

        c = a + b  # converted custom_float16
        c[0] = 10  # partial update float16 to custom_float16

        result = torch.tensor([10, 5, 7], dtype=torch.float16, device=self.rbln_device)
        self.assertEqual(c, result, atol=ATOL, rtol=RTOL)

        # case 2: partial update float16 to custom_float16
        aa = torch.tensor([1], dtype=torch.float16, device=self.rbln_device)
        bb = torch.tensor([2], dtype=torch.float16, device=self.rbln_device)
        cc = torch.tensor([1, 2, 3], dtype=torch.float16, device=self.rbln_device)

        dd = aa + bb  # converted custom_float16
        cc[0] = dd  # partial update custom_float16 to float16

        result2 = torch.tensor([3, 2, 3], dtype=torch.float16, device=self.rbln_device)
        self.assertEqual(cc, result2, atol=ATOL, rtol=RTOL)

    def test_remain_cf16_to_cf16_in_d2d(self):
        """Test that the dtype of a tensor remains cf16 after copy to other device tensor."""
        a = torch.ones([2, 1, 64], dtype=torch.float16, device=self.rbln_device)
        a = a.reshape([1, 1, 128])

        b = torch.zeros([1, 1, 128], dtype=torch.float16, device=self.rbln_device)

        b.copy_(a)

        self.assertEqual(b.cpu(), a.cpu(), atol=ATOL, rtol=RTOL)

    @parametrize("src_dtype", TEST_DTYPES)
    @parametrize("dst_dtype", TEST_DTYPES)
    def test_copy_with_broadcast_and_dtype_conversion(self, src_dtype, dst_dtype):
        # (1,) → (1024,)
        src = torch.tensor([3.14], dtype=src_dtype, device=self.rbln_device)
        dst = torch.empty([1024], dtype=dst_dtype, device=self.rbln_device)
        dst.copy_(src.expand(1024))
        expected = torch.full((1024,), 3.14, dtype=dst_dtype)
        self.assertEqual(dst.cpu(), expected, atol=ATOL, rtol=RTOL)

        # (1, 8) → (4, 8)
        src_small = torch.randn([1, 8], dtype=src_dtype, device=self.rbln_device)
        dst_large = torch.empty([4, 8], dtype=dst_dtype, device=self.rbln_device)
        dst_large.copy_(src_small.expand(4, 8))
        expected = src_small.cpu().expand(4, 8).to(dst_dtype)
        self.assertEqual(dst_large.cpu(), expected, atol=ATOL, rtol=RTOL)

        # (1, 64) → (8, 64)
        src = torch.randn([1, 64], dtype=src_dtype, device=self.rbln_device)
        dst = torch.empty([8, 64], dtype=dst_dtype, device=self.rbln_device)
        dst.copy_(src.expand(8, 64))
        expected = src.cpu().expand(8, 64).to(dst_dtype)
        self.assertEqual(dst.cpu(), expected, atol=ATOL, rtol=RTOL)

        # (1, 1, 64) → (4, 8, 64)
        src = torch.randn([1, 1, 64], dtype=src_dtype, device=self.rbln_device)
        dst = torch.empty([4, 8, 64], dtype=dst_dtype, device=self.rbln_device)
        dst.copy_(src.expand(4, 8, 64))
        expected = src.cpu().expand(4, 8, 64).to(dst_dtype)
        self.assertEqual(dst.cpu(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.test_set_ci
class TestToOps(TestCase):
    @dtypes(*TEST_DTYPES)
    def test_to_cpu(self, dtype):
        x_cpu = torch.randn([2, 4], dtype=dtype, device="cpu")
        self.assertEqual(x_cpu.dtype, dtype)

        x = torch.tensor(x_cpu, dtype=x_cpu.dtype, device="rbln")
        self.assertEqual(x.dtype, dtype)

        y = x.to("cpu")
        self.assertEqual(y.dtype, dtype)

        z = x.to("cpu", dtype=dtype)
        self.assertEqual(z.dtype, dtype)

        self.assertEqual(z, x_cpu, atol=ATOL, rtol=RTOL)

    @dtypes(*TEST_DTYPES)
    def test_to_copy(self, dtype):
        x_cpu = torch.randn([2, 4], dtype=dtype, device="cpu")
        self.assertEqual(x_cpu.dtype, dtype)

        x = torch.tensor(x_cpu, dtype=x_cpu.dtype, device="rbln")
        self.assertEqual(x.dtype, dtype)

        y = x.to("cpu")
        self.assertEqual(y.dtype, dtype)

        z = torch.empty([2, 4], dtype=dtype, device="cpu")
        self.assertEqual(z.dtype, dtype)

        z.copy_(x)
        self.assertEqual(z.dtype, dtype)

        self.assertEqual(z, x_cpu, atol=ATOL, rtol=RTOL)

    def test_cf16_cast_fp16_order(self):
        """
        Tests that the order of operations for device transfer and dtype casting
        from a conceptual CF16 tensor produces numerically equivalent results.

        It verifies that casting from CF16 to float16 on the RBLN device and then
        moving to the CPU yields the same result as moving the CF16 tensor to
        the CPU first.
        """
        a = torch.randn([2, 1024], dtype=torch.float16, device="rbln")
        a = a @ a.t()  # To make custom_float16

        a_casted = a.to(torch.float16).to("cpu")
        a_offloaded = a.to("cpu").to(torch.float16)
        self.assertEqual(a_casted, a_offloaded)

    def test_cf16_cast_fp32_order(self):
        """
        Tests for numerical consistency when an operation sequence is performed on
        RBLN versus being fully offloaded to the CPU.

        It compares the result of two paths:
        1.  CPU Fallback Path: The CF16 tensor is cast to float32 on the RBLN
            device, and a matrix multiplication is performed on RBLN.
        2.  CPU Offloading Path: The CF16 tensor is first moved to the CPU,
            then cast to float32, and the matrix multiplication is performed on CPU.

        The final outputs from both paths should be numerically close.
        """
        a = torch.randn([2, 1024], dtype=torch.float16, device="rbln")
        a = a @ a.t()  # To make custom_float16

        out_cpu_fallback = a.to(torch.float32)
        out_cpu_fallback = out_cpu_fallback @ out_cpu_fallback

        out_cpu_offloading = a.to("cpu")
        out_cpu_offloading = out_cpu_offloading.to(torch.float32)
        out_cpu_offloading = out_cpu_offloading @ out_cpu_offloading
        self.assertEqual(out_cpu_offloading, out_cpu_fallback.to("cpu"))

    def test_cf16_to_cpu_remain_cf16(self):
        """
        Tests that the dtype of a tensor remains custom_float16 after casting to float16 and then back to rbln.
        """
        a = torch.randn([2, 2], dtype=torch.float16, device="rbln:0")
        self.assertEqual(a.dtype, torch.float16)

        a_casted = a.to(device="cpu", dtype=torch.float16)
        self.assertEqual(a_casted.dtype, torch.float16)

        a_cf16 = a @ a.T
        self.assertEqual(a_cf16.dtype, torch.float16)

        a_cpu = a_cf16.to("cpu")
        self.assertEqual(a_cpu.dtype, torch.float16)

        a_rbln = a_cpu.to("rbln:0")  # remain cf16
        self.assertEqual(a_rbln.dtype, torch.float16)

        out_rbln = a_rbln @ a_cf16
        self.assertEqual(out_rbln.dtype, torch.float16)

        out_ref = a_cpu.to(torch.float16) @ (a_casted @ a_casted.T)
        self.assertEqual(out_ref.dtype, torch.float16)

        self.assertEqual(out_rbln, out_ref, atol=ATOL, rtol=RTOL)

    def test_cf16_to_cpu_remain_cf16_with_to(self):
        """
        Tests that the dtype of a tensor remains custom_float16 after casting to
        float16 and then back to rbln with .to("cpu").to("cpu").
        """
        a = torch.randn([2, 2], dtype=torch.float16, device="rbln")
        a_cf16 = a @ a.T
        a_cpu = a_cf16.to("cpu")
        a_cpu2 = a_cpu.to("cpu")  # a_cpu2 is equal as a_cpu
        self.assertEqual(a_cpu2, a_cpu, atol=ATOL, rtol=RTOL)

        a_cpu_fp16 = a_cpu.to(torch.float16)  # a_cpu_fp16 casted to float16
        self.assertEqual(a_cpu_fp16.dtype, torch.float16)


instantiate_device_type_tests(TestTensorCopy, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestToOps, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
