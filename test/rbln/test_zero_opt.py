# Owner(s): ["module: PrivateUse1"]

"""
Comprehensive tests for the optimized zero_ op (EMPTY_INIT_WITH_ZERO marking).

The optimized zero_ marks VMemory as logically zero-initialized without allocating
host memory. These tests verify correctness across all access patterns:
- Host read (CPU readback), device read (NPU computation on zeros)
- Partial writes (slice assignment), in-place ops, arithmetic
- Views, reshapes, non-contiguous tensors, storage sharing
- Edge cases: empty tensors, scalars, large tensors, repeated zero_ calls
"""

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


ATOL = 0.01
RTOL = 0.01


@pytest.mark.test_set_ci
class TestZeroOptBasic(TestCase):
    """Basic correctness: zero_ produces all-zero tensors readable from CPU."""

    def test_zero_readback_1d(self):
        t = torch.ones(64, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t.cpu()
        self.assertTrue(torch.all(result == 0).item())

    def test_zero_readback_2d(self):
        t = torch.ones(8, 16, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t.cpu()
        self.assertTrue(torch.all(result == 0).item())

    def test_zero_readback_3d(self):
        t = torch.ones(4, 8, 16, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t.cpu()
        self.assertTrue(torch.all(result == 0).item())

    def test_zero_scalar(self):
        t = torch.tensor(42.0, dtype=torch.float16, device="rbln")
        t.zero_()
        self.assertEqual(t.cpu().item(), 0.0)

    def test_zero_empty_tensor(self):
        """zero_ on empty tensor should be a no-op and not crash."""
        t = torch.empty(0, dtype=torch.float16, device="rbln")
        t.zero_()  # should not raise
        self.assertEqual(t.numel(), 0)

    def test_zero_returns_self(self):
        """zero_ is in-place and should return the same tensor."""
        t = torch.ones(4, dtype=torch.float16, device="rbln")
        ret = t.zero_()
        self.assertIs(ret, t)

    def test_zero_overwrites_existing_data(self):
        """zero_ on a tensor with non-zero data should produce zeros."""
        t = torch.arange(16, dtype=torch.float16, device="rbln")
        self.assertFalse(torch.all(t.cpu() == 0).item())
        t.zero_()
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_zero_idempotent(self):
        """Calling zero_ multiple times should remain all zeros."""
        t = torch.ones(32, dtype=torch.float16, device="rbln")
        t.zero_()
        t.zero_()
        t.zero_()
        self.assertTrue(torch.all(t.cpu() == 0).item())


@pytest.mark.test_set_ci
class TestZeroOptArithmetic(TestCase):
    """Arithmetic on zero-initialized tensors (device read path)."""

    def test_zero_plus_one(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t + 1.0
        expected = torch.ones(16, dtype=torch.float16)
        self.assertEqual(result.cpu(), expected, atol=ATOL, rtol=RTOL)

    def test_zero_mul(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        other = torch.ones(16, dtype=torch.float16, device="rbln") * 5.0
        result = t * other
        expected = torch.zeros(16, dtype=torch.float16)
        self.assertEqual(result.cpu(), expected, atol=ATOL, rtol=RTOL)

    def test_zero_add_inplace(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        t.add_(3.0)
        expected = torch.full((16,), 3.0, dtype=torch.float16)
        self.assertEqual(t.cpu(), expected, atol=ATOL, rtol=RTOL)

    def test_zero_matmul(self):
        """Matrix multiply with zero-initialized tensor should produce zeros."""
        a = torch.ones(4, 8, dtype=torch.float16, device="rbln")
        a.zero_()
        b = torch.ones(8, 4, dtype=torch.float16, device="rbln")
        result = torch.mm(a, b)
        expected = torch.zeros(4, 4, dtype=torch.float16)
        self.assertEqual(result.cpu(), expected, atol=ATOL, rtol=RTOL)

    def test_zero_comparison(self):
        """Comparison: zero tensor == 0 should be all True."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        cmp = t == 0
        self.assertTrue(torch.all(cmp.cpu()).item())


@pytest.mark.test_set_ci
class TestZeroOptPartialWrite(TestCase):
    """Partial writes (slice assignment) after zero_ marking."""

    def test_slice_write_then_read(self):
        """Write to a slice of zero-init tensor, verify written and unwritten regions."""
        t = torch.zeros(16, dtype=torch.float16, device="rbln")
        # Write to first 4 elements
        t[:4] = torch.ones(4, dtype=torch.float16, device="rbln")
        result = t.cpu()
        # First 4 should be 1, rest should be 0
        self.assertEqual(result[:4], torch.ones(4, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[4:] == 0).item())

    def test_slice_write_middle(self):
        t = torch.zeros(16, dtype=torch.float16, device="rbln")
        t[4:8] = torch.full((4,), 2.0, dtype=torch.float16, device="rbln")
        result = t.cpu()
        self.assertTrue(torch.all(result[:4] == 0).item())
        self.assertEqual(result[4:8], torch.full((4,), 2.0, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[8:] == 0).item())

    def test_2d_row_write(self):
        t = torch.zeros(4, 8, dtype=torch.float16, device="rbln")
        t[0] = torch.ones(8, dtype=torch.float16, device="rbln")
        result = t.cpu()
        self.assertEqual(result[0], torch.ones(8, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[1:] == 0).item())

    def test_2d_column_write(self):
        t = torch.zeros(4, 8, dtype=torch.float16, device="rbln")
        t[:, 0] = torch.ones(4, dtype=torch.float16, device="rbln")
        result = t.cpu()
        self.assertEqual(result[:, 0], torch.ones(4, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[:, 1:] == 0).item())

    def test_advanced_indexing(self):
        t = torch.zeros(8, dtype=torch.float16, device="rbln")
        indices = torch.tensor([0, 3, 7])
        t[indices] = torch.ones(3, dtype=torch.float16, device="rbln")
        result = t.cpu()
        for i in range(8):
            if i in [0, 3, 7]:
                self.assertAlmostEqual(result[i].item(), 1.0, places=2)
            else:
                self.assertAlmostEqual(result[i].item(), 0.0, places=2)


@pytest.mark.test_set_ci
class TestZeroOptViews(TestCase):
    """Views and reshapes on zero-initialized tensors."""

    def test_reshape_after_zero(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        reshaped = t.reshape(4, 4)
        self.assertTrue(torch.all(reshaped.cpu() == 0).item())

    def test_view_after_zero(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        viewed = t.view(2, 8)
        self.assertTrue(torch.all(viewed.cpu() == 0).item())

    def test_transpose_after_zero(self):
        t = torch.ones(4, 8, dtype=torch.float16, device="rbln")
        t.zero_()
        transposed = t.t()
        self.assertEqual(transposed.shape, (8, 4))
        self.assertTrue(torch.all(transposed.cpu() == 0).item())

    def test_contiguous_after_zero_and_transpose(self):
        t = torch.ones(4, 8, dtype=torch.float16, device="rbln")
        t.zero_()
        transposed = t.t()
        contig = transposed.contiguous()
        self.assertTrue(torch.all(contig.cpu() == 0).item())

    def test_slice_view_after_zero(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        sliced = t[4:12]
        self.assertEqual(sliced.shape, (8,))
        self.assertTrue(torch.all(sliced.cpu() == 0).item())

    def test_write_through_view(self):
        """Write through a view of zero-init tensor, verify base tensor updates."""
        t = torch.zeros(16, dtype=torch.float16, device="rbln")
        view = t.view(4, 4)
        view[0] = torch.ones(4, dtype=torch.float16, device="rbln")
        result = t.cpu()
        # First 4 elements should be 1 (written through view), rest 0
        self.assertEqual(result[:4], torch.ones(4, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[4:] == 0).item())

    def test_expand_after_zero(self):
        t = torch.ones(1, 8, dtype=torch.float16, device="rbln")
        t.zero_()
        expanded = t.expand(4, 8)
        self.assertEqual(expanded.shape, (4, 8))
        self.assertTrue(torch.all(expanded.cpu() == 0).item())


@pytest.mark.test_set_ci
class TestZeroOptCopy(TestCase):
    """Copy operations involving zero-initialized tensors."""

    def test_copy_from_zero(self):
        """copy_ from a zero-init tensor to another tensor."""
        src = torch.ones(16, dtype=torch.float16, device="rbln")
        src.zero_()
        dst = torch.ones(16, dtype=torch.float16, device="rbln")
        dst.copy_(src)
        self.assertTrue(torch.all(dst.cpu() == 0).item())

    def test_copy_into_zero(self):
        """copy_ non-zero data into a zero-init tensor."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        src = torch.arange(16, dtype=torch.float16, device="rbln")
        t.copy_(src)
        self.assertEqual(t.cpu(), src.cpu(), atol=ATOL, rtol=RTOL)

    def test_clone_zero(self):
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        cloned = t.clone()
        self.assertTrue(torch.all(cloned.cpu() == 0).item())
        # Modifying clone should not affect original
        cloned.add_(1.0)
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_to_cpu_and_back(self):
        """Round-trip: zero_ → cpu → rbln should preserve zeros."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.zero_()
        cpu_t = t.cpu()
        self.assertTrue(torch.all(cpu_t == 0).item())
        back = cpu_t.to("rbln")
        self.assertTrue(torch.all(back.cpu() == 0).item())


@pytest.mark.test_set_ci
class TestZeroOptKVCachePattern(TestCase):
    """Simulate KV-cache usage pattern: zero-init then overwrite via NPU."""

    def test_kv_cache_write_first(self):
        """KV-cache pattern: zero_ then full NPU write (should skip zero transfer)."""
        cache = torch.empty(2, 4, 8, dtype=torch.float16, device="rbln")
        cache.zero_()
        # Simulate prefill: overwrite entire cache
        new_data = torch.randn(2, 4, 8, dtype=torch.float16, device="rbln")
        cache.copy_(new_data)
        self.assertEqual(cache.cpu(), new_data.cpu(), atol=ATOL, rtol=RTOL)

    def test_kv_cache_partial_fill(self):
        """Simulate: zero cache, fill seq_len=2 of 8, rest stays zero."""
        cache = torch.zeros(1, 8, 4, dtype=torch.float16, device="rbln")
        fill = torch.ones(1, 2, 4, dtype=torch.float16, device="rbln")
        cache[:, :2, :] = fill
        result = cache.cpu()
        self.assertEqual(result[:, :2, :], torch.ones(1, 2, 4, dtype=torch.float16), atol=ATOL, rtol=RTOL)
        self.assertTrue(torch.all(result[:, 2:, :] == 0).item())

    def test_kv_cache_incremental_decode(self):
        """Simulate decode: fill one position at a time after zero-init."""
        seq_len = 8
        cache = torch.zeros(1, seq_len, 4, dtype=torch.float16, device="rbln")
        for i in range(seq_len):
            val = torch.full((1, 1, 4), float(i + 1), dtype=torch.float16, device="rbln")
            cache[:, i : i + 1, :] = val
        result = cache.cpu()
        for i in range(seq_len):
            expected_val = float(i + 1)
            self.assertTrue(
                torch.allclose(
                    result[:, i, :],
                    torch.full((1, 4), expected_val, dtype=torch.float16),
                    atol=ATOL,
                    rtol=RTOL,
                )
            )


@pytest.mark.test_set_ci
class TestZeroOptLargeTensor(TestCase):
    """Large tensor tests — closer to real KV-cache sizes."""

    def test_large_zero_readback(self):
        """Large tensor: verify all zeros after mark."""
        t = torch.ones(1024, 1024, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t.cpu()
        self.assertTrue(torch.all(result == 0).item())

    def test_large_zero_then_compute(self):
        """Large tensor: zero_ then add should produce the addend."""
        t = torch.ones(512, 512, dtype=torch.float16, device="rbln")
        t.zero_()
        result = t + 1.0
        self.assertTrue(torch.all(result.cpu() == 1.0).item())


@pytest.mark.test_set_ci
class TestZeroOptReZero(TestCase):
    """Re-zeroing: zero_ after tensor already has data (state transitions)."""

    def test_zero_after_computation(self):
        """Tensor with computed data, zero_ again, should be all zeros."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.add_(5.0)  # now all 6.0
        t.zero_()
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_zero_after_host_read(self):
        """Read tensor to CPU (syncs host view), then zero_ again."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        _ = t.cpu()  # force host sync (USER_VIEW_IS_LATEST)
        t.zero_()  # should transition back to EMPTY_INIT_WITH_ZERO
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_zero_after_device_write_then_readback(self):
        """Write on device, zero_, readback — verifies state override."""
        t = torch.ones(16, dtype=torch.float16, device="rbln")
        t.add_(1.0)  # device write (PHYSICAL_VIEW_IS_LATEST)
        t.zero_()  # mark as zero again
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_alternating_zero_and_fill(self):
        """Alternate zero_ and fill_, verify each state is correct."""
        t = torch.empty(16, dtype=torch.float16, device="rbln")
        for val in [0.0, 3.0, 0.0, 7.0, 0.0]:
            if val == 0.0:
                t.zero_()
            else:
                t.fill_(val)
            expected = torch.full((16,), val, dtype=torch.float16)
            self.assertEqual(t.cpu(), expected, atol=ATOL, rtol=RTOL)


@pytest.mark.test_set_ci
class TestZeroOptTorchZeros(TestCase):
    """torch.zeros() factory — should also use the optimized path."""

    def test_torch_zeros(self):
        t = torch.zeros(16, dtype=torch.float16, device="rbln")
        self.assertTrue(torch.all(t.cpu() == 0).item())

    def test_torch_zeros_like(self):
        ref = torch.ones(4, 8, dtype=torch.float16, device="rbln")
        t = torch.zeros_like(ref)
        self.assertTrue(torch.all(t.cpu() == 0).item())
        self.assertEqual(t.shape, ref.shape)
        self.assertEqual(t.dtype, ref.dtype)

    def test_torch_zeros_then_operation(self):
        t = torch.zeros(16, dtype=torch.float16, device="rbln")
        result = t + torch.ones(16, dtype=torch.float16, device="rbln")
        self.assertEqual(result.cpu(), torch.ones(16, dtype=torch.float16), atol=ATOL, rtol=RTOL)


instantiate_device_type_tests(TestZeroOptBasic, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptArithmetic, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptPartialWrite, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptViews, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptCopy, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptKVCachePattern, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptLargeTensor, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptReZero, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestZeroOptTorchZeros, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
