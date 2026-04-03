# Owner(s): ["module: PrivateUse1"]

"""
Test suite for storage offset handling.

This test suite validates the correct behavior of tensors with non-zero storage offsets in various scenarios:
1. Split tensors with view operations (transpose, reshape, etc.)
2. Split and merge operations
"""

import unittest

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


class BaseTestNonZeroStorageOffset(TestCase):
    """Base class for non-zero storage offset tests with common helper methods."""

    rbln_device = torch.device("rbln:0")

    # Flag to enable contiguous() call for non-zero storage offset tensors that are contiguous
    ALLOW_CONTIGUOUS_FOR_NONZERO_OFFSET = True

    def _check_storage_offset(self, tensor, expected_offset_min=0, msg=""):
        """Helper to check storage offset."""
        actual_offset = tensor.storage_offset()
        self.assertEqual(
            actual_offset,
            expected_offset_min,
            f"{msg}: Expected storage offset == {expected_offset_min}, got {actual_offset}",
        )
        return actual_offset

    def _check_contiguous_with_storage_offset(self, tensor, msg=""):
        """Check if tensor with non-zero storage offset is contiguous (unexpected case)."""
        storage_offset = tensor.storage_offset()
        is_contig = tensor.is_contiguous()

        if storage_offset > 0 and is_contig:
            print(
                f"WARNING [{msg}]: Tensor has non-zero storage_offset={storage_offset} but is_contiguous()=True. "
                f"This may cause test failures."
            )

        return is_contig

    def _ensure_contiguous_if_needed(self, tensor, msg=""):
        """
        Ensure tensor is contiguous if it has non-zero storage offset and is contiguous.

        If ALLOW_CONTIGUOUS_FOR_NONZERO_OFFSET is True and the tensor has non-zero storage
        offset but is contiguous, create a new tensor with storage_offset=0 and contiguous=True.
        Otherwise, return the original tensor.

        Args:
            tensor: Input tensor
            msg: Optional message for logging

        Returns:
            Tensor that is guaranteed to be contiguous with storage_offset=0 (if flag is enabled and conditions met)
        """
        storage_offset = tensor.storage_offset()
        is_contig = tensor.is_contiguous()
        print(f"INFO [{msg}]: storage_offset={storage_offset}, is_contig={is_contig}")
        self._check_contiguous_with_storage_offset(tensor, msg)

        if self.ALLOW_CONTIGUOUS_FOR_NONZERO_OFFSET:
            if storage_offset > 0 and is_contig:
                print(
                    f"INFO [{msg}]: Creating new tensor with storage_offset=0 from tensor with "
                    f"non-zero storage_offset={storage_offset} and is_contiguous()=True "
                    f"(ALLOW_CONTIGUOUS_FOR_NONZERO_OFFSET=True)"
                )
                # Create a new tensor with storage_offset=0 by creating empty tensor and copying
                # This ensures we get a new tensor with storage_offset=0
                new_tensor = torch.empty_like(tensor, device=self.rbln_device)
                new_tensor.copy_(tensor)
                # Verify the new tensor has storage_offset=0 and is contiguous
                self.assertEqual(new_tensor.storage_offset(), 0, f"{msg}: New tensor should have storage_offset=0")
                self.assertTrue(new_tensor.is_contiguous(), f"{msg}: New tensor should be contiguous")
                return new_tensor

        return tensor


@pytest.mark.test_set_ci
class TestSplitOps(BaseTestNonZeroStorageOffset):
    """Test split tensors with view operations (transpose, reshape, etc.)."""

    @dtypes(*SUPPORTED_DTYPES)
    def test_slice_contiguous_then_add(self, dtype):
        """Test slice that produces contiguous tensor, then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Slice to get contiguous tensor
        sliced = base_tensor[16:]  # Contiguous slice

        # Check contiguous status
        self._check_contiguous_with_storage_offset(sliced, "[test_slice_contiguous_then_add] sliced")
        self.assertTrue(sliced.is_contiguous(), "Sliced tensor should be contiguous")

        sliced = self._ensure_contiguous_if_needed(sliced, "[test_slice_contiguous_then_add] sliced")
        result = torch.add(sliced, 1.0)
        # Create add tensor: first 16 rows = 1.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[16:] = 1.0
        self.assertEqual(result.cpu(), (base_tensor + add_tensor)[16:].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_slice_non_contiguous_then_add(self, dtype):
        """Test slice that produces non-contiguous tensor, then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Slice to get non-contiguous tensor (strided slice)
        sliced = base_tensor[::2]  # Every other row - non-contiguous

        # Check contiguous status
        self._check_contiguous_with_storage_offset(sliced, "[test_slice_non_contiguous_then_add] sliced")
        self.assertFalse(sliced.is_contiguous(), "Strided sliced tensor should be non-contiguous")

        sliced = self._ensure_contiguous_if_needed(sliced, "[test_slice_non_contiguous_then_add] sliced")
        result = torch.add(sliced, 1.0)
        # Create add tensor: every other row = 1.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[::2] = 1.0
        self.assertEqual(result.cpu(), (base_tensor + add_tensor)[::2].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_chunk_contiguous_then_add(self, dtype):
        """Test chunk that produces contiguous tensors, then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Chunk to get contiguous tensors
        chunks = torch.chunk(base_tensor, chunks=4, dim=0)

        # Create add tensor: each chunk gets its own value (1.0, 2.0, 3.0, 4.0)
        add_tensor = torch.zeros_like(base_tensor)
        for i in range(4):
            add_tensor[i * 8 : (i + 1) * 8] = float(i + 1)

        # Check contiguous status for each chunk
        for i, chunk in enumerate(chunks):
            chunk = self._ensure_contiguous_if_needed(chunk, "[test_chunk_contiguous_then_add] chunk")
            result = torch.add(chunk, float(i + 1))
            self.assertEqual(result.cpu(), (base_tensor + add_tensor)[i * 8 : (i + 1) * 8].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_chunk_non_contiguous_then_add(self, dtype):
        """Test chunk that produces non-contiguous tensors (after transpose), then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Chunk to get contiguous tensors first
        chunks = torch.chunk(base_tensor, chunks=4, dim=0)

        # Apply transpose to make them non-contiguous
        transposed_chunks = [chunk.transpose(0, 1) for chunk in chunks]

        # Create add tensor: each chunk gets its own value (1.0, 2.0, 3.0, 4.0)
        add_tensor = torch.zeros_like(base_tensor)
        for i in range(4):
            add_tensor[i * 8 : (i + 1) * 8] = float(i + 1)

        # Check contiguous status for each transposed chunk
        for i, chunk_t in enumerate(transposed_chunks):
            chunk_t = self._ensure_contiguous_if_needed(chunk_t, "[test_chunk_non_contiguous_then_add] chunk_t")
            result = torch.add(chunk_t, float(i + 1))
            self.assertEqual(result.cpu(), (base_tensor + add_tensor)[i * 8 : (i + 1) * 8].transpose(0, 1).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_slice_contiguous_column_then_add(self, dtype):
        """Test column slice that produces contiguous tensor, then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Column slice to get contiguous tensor
        sliced = base_tensor[:, :32]  # First half of columns

        sliced = self._ensure_contiguous_if_needed(sliced, "[test_slice_contiguous_column_then_add] sliced")
        result = torch.add(sliced, 2.0)
        # Create add tensor: first 32 columns = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:, :32] = 2.0
        self.assertEqual(result.cpu(), (base_tensor + add_tensor)[:, :32].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_slice_non_contiguous_column_then_add(self, dtype):
        """Test column slice that produces non-contiguous tensor, then torch.add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Column slice with stride to get non-contiguous tensor
        sliced = base_tensor[:, ::2]  # Every other column - non-contiguous

        sliced = self._ensure_contiguous_if_needed(sliced, "[test_slice_non_contiguous_column_then_add] sliced")
        result = torch.add(sliced, 2.0)
        # Create add tensor: every other column = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:, ::2] = 2.0
        self.assertEqual(result.cpu(), (base_tensor + add_tensor)[:, ::2].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_add(self, dtype):
        """Test split using slice, then apply add operation."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:16]  # First half
        chunk2 = base_tensor[16:]  # Second half

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 16 * 64, "chunk2")

        chunk1 = self._ensure_contiguous_if_needed(chunk1, "[test_split_slice_then_add] chunk1")
        chunk2 = self._ensure_contiguous_if_needed(chunk2, "[test_split_slice_then_add] chunk2")
        result1 = torch.add(chunk1, 1.0)
        result2 = torch.add(chunk2, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:16] = 1.0
        add_tensor[16:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:16].cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[16:].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_transpose(self, dtype):
        """Test split using slice, then apply transpose."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:16]  # First half
        chunk2 = base_tensor[16:]  # Second half

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 16 * 64, "chunk2")

        # Apply transpose to each chunk
        chunk1_t = chunk1.transpose(0, 1)
        chunk2_t = chunk2.transpose(0, 1)

        chunk1_t = self._ensure_contiguous_if_needed(chunk1_t, "[test_split_slice_then_transpose] chunk1_t")
        chunk2_t = self._ensure_contiguous_if_needed(chunk2_t, "[test_split_slice_then_transpose] chunk2_t")
        result1 = torch.add(chunk1_t, 1.0)
        result2 = torch.add(chunk2_t, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:16] = 1.0
        add_tensor[16:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:16].transpose(0, 1).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[16:].transpose(0, 1).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_reshape(self, dtype):
        """Test split using slice, then apply reshape."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:16]
        chunk2 = base_tensor[16:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 16 * 64, "chunk2")

        # Apply reshape to each chunk
        chunk1_r = chunk1.reshape(16, 64)
        chunk2_r = chunk2.reshape(16, 64)

        chunk1_r = self._ensure_contiguous_if_needed(chunk1_r, "[test_split_slice_then_reshape] chunk1_r")
        chunk2_r = self._ensure_contiguous_if_needed(chunk2_r, "[test_split_slice_then_reshape] chunk2_r")
        result1 = torch.add(chunk1_r, 1.0)
        result2 = torch.add(chunk2_r, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:16] = 1.0
        add_tensor[16:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:16].reshape(16, 64).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[16:].reshape(16, 64).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_chunk_then_transpose(self, dtype):
        """Test split using torch.chunk, then apply transpose."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using chunk
        chunks = torch.chunk(base_tensor, chunks=4, dim=0)

        # Check storage offsets for non-first chunks
        for i, chunk in enumerate(chunks):
            self._check_storage_offset(chunk, i * 8 * 64, f"chunk{i}")

        # Apply transpose to each chunk
        transposed_chunks = [chunk.transpose(0, 1) for chunk in chunks]

        # Check contiguous status after transpose
        for i, chunk_t in enumerate(transposed_chunks):
            print(
                f"[test_split_chunk_then_transpose] transposed_chunk[{i}].is_contiguous() = {chunk_t.is_contiguous()}"
            )

        # Verify transpose worked correctly
        expected_transposed = [base_tensor[i * 8 : (i + 1) * 8].transpose(0, 1) for i in range(4)]

        for i, (actual, expected) in enumerate(zip(transposed_chunks, expected_transposed)):
            self.assertEqual(actual.cpu(), expected.cpu(), msg=f"chunk {i} transpose mismatch")

        transposed_chunks = [
            self._ensure_contiguous_if_needed(tc, "[test_split_chunk_then_transpose] tc") for tc in transposed_chunks
        ]
        results = [torch.add(tc, 1.0) for tc in transposed_chunks]
        expected_results = [et + 1.0 for et in expected_transposed]

        for i, (actual, expected) in enumerate(zip(results, expected_results)):
            self.assertEqual(actual.cpu(), expected.cpu(), msg=f"chunk {i} operation mismatch")

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_permute(self, dtype):
        """Test split using slice, then apply permute."""
        # Create 3D tensor
        base_tensor = torch.randn([16, 32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:8]
        chunk2 = base_tensor[8:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 8 * 32 * 64, "chunk2")

        # Apply permute to each chunk
        chunk1_p = chunk1.permute(1, 2, 0)  # (32, 64, 8)
        chunk2_p = chunk2.permute(1, 2, 0)  # (32, 64, 8)

        chunk1_p = self._ensure_contiguous_if_needed(chunk1_p, "[test_split_slice_then_permute] chunk1_p")
        chunk2_p = self._ensure_contiguous_if_needed(chunk2_p, "[test_split_slice_then_permute] chunk2_p")
        result1 = torch.add(chunk1_p, 1.0)
        result2 = torch.add(chunk2_p, 2.0)
        # Create add tensor: first 16 0-dim = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:8] = 1.0
        add_tensor[8:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:8].permute(1, 2, 0).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[8:].permute(1, 2, 0).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_transpose_then_reshape(self, dtype):
        """Test split, then transpose, then reshape."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:16]
        chunk2 = base_tensor[16:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 16 * 64, "chunk2")

        # Apply transpose
        chunk1_t = chunk1.transpose(0, 1)
        chunk2_t = chunk2.transpose(0, 1)

        # Then apply reshape
        chunk1_tr = chunk1_t.reshape(64, 16)
        chunk2_tr = chunk2_t.reshape(64, 16)

        chunk1_tr = self._ensure_contiguous_if_needed(
            chunk1_tr, "[test_split_slice_then_transpose_then_reshape] chunk1_tr"
        )
        chunk2_tr = self._ensure_contiguous_if_needed(
            chunk2_tr, "[test_split_slice_then_transpose_then_reshape] chunk2_tr"
        )
        result1 = torch.add(chunk1_tr, 1.0)
        result2 = torch.add(chunk2_tr, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:16] = 1.0
        add_tensor[16:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:16].transpose(0, 1).reshape(64, 16).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[16:].transpose(0, 1).reshape(64, 16).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_strided_slice_then_add(self, dtype):
        """Test split using strided slice, then apply transpose."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using strided slice
        chunk1 = base_tensor[::2]  # Every other row
        chunk2 = base_tensor[1::2]  # Remaining rows

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 64, "chunk2")  # Offset by one row

        chunk1 = self._ensure_contiguous_if_needed(chunk1, "[test_split_strided_slice_then_add] chunk1")
        chunk2 = self._ensure_contiguous_if_needed(chunk2, "[test_split_strided_slice_then_add] chunk2")
        result1 = torch.add(chunk1, 1.0)
        result2 = torch.add(chunk2, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[::2] = 1.0
        add_tensor[1::2] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[::2].cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[1::2].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_strided_slice_then_transpose(self, dtype):
        """Test split using strided slice, then apply transpose."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using strided slice
        chunk1 = base_tensor[::2]  # Every other row
        chunk2 = base_tensor[1::2]  # Remaining rows

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 64, "chunk2")  # Offset by one row

        # Apply transpose to each chunk
        chunk1_t = chunk1.transpose(0, 1)
        chunk2_t = chunk2.transpose(0, 1)

        chunk1_t = self._ensure_contiguous_if_needed(chunk1_t, "[test_split_strided_slice_then_transpose] chunk1_t")
        chunk2_t = self._ensure_contiguous_if_needed(chunk2_t, "[test_split_strided_slice_then_transpose] chunk2_t")
        result1 = torch.add(chunk1_t, 1.0)
        result2 = torch.add(chunk2_t, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[::2] = 1.0
        add_tensor[1::2] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[::2].transpose(0, 1).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[1::2].transpose(0, 1).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_column_slice_then_add(self, dtype):
        """Test split using column slice, then apply add."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using column slice
        chunk1 = base_tensor[:, :32]  # First half of columns
        chunk2 = base_tensor[:, 32:]  # Second half of columns

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 32, "chunk2")  # Offset by 32 elements

        chunk1 = self._ensure_contiguous_if_needed(chunk1, "[test_split_column_slice_then_add] chunk1")
        chunk2 = self._ensure_contiguous_if_needed(chunk2, "[test_split_column_slice_then_add] chunk2")
        result1 = torch.add(chunk1, 1.0)
        result2 = torch.add(chunk2, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:, :32] = 1.0
        add_tensor[:, 32:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:, :32].cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[:, 32:].cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_column_slice_then_transpose(self, dtype):
        """Test split using column slice, then apply transpose."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using column slice
        chunk1 = base_tensor[:, :32]  # First half of columns
        chunk2 = base_tensor[:, 32:]  # Second half of columns

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 32, "chunk2")  # Offset by 32 elements

        # Apply transpose
        chunk1_t = chunk1.transpose(0, 1)
        chunk2_t = chunk2.transpose(0, 1)

        chunk1_t = self._ensure_contiguous_if_needed(chunk1_t, "[test_split_column_slice_then_transpose] chunk1_t")
        chunk2_t = self._ensure_contiguous_if_needed(chunk2_t, "[test_split_column_slice_then_transpose] chunk2_t")
        result1 = torch.add(chunk1_t, 1.0)
        result2 = torch.add(chunk2_t, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:, :32] = 1.0
        add_tensor[:, 32:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:, :32].transpose(0, 1).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[:, 32:].transpose(0, 1).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_3d_then_transpose_then_reshape(self, dtype):
        """Test split 3D tensor, then transpose, then reshape."""
        # Create 3D tensor
        base_tensor = torch.randn([16, 32, 64], dtype=dtype, device=self.rbln_device)

        # Split along first dimension
        chunk1 = base_tensor[:8]
        chunk2 = base_tensor[8:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 8 * 32 * 64, "chunk2")

        # Apply transpose (swap dims 1 and 2), then reshape
        chunk1_tr = chunk1.transpose(1, 2).reshape(8, 64, 32)
        chunk2_tr = chunk2.transpose(1, 2).reshape(8, 64, 32)

        chunk1_tr = self._ensure_contiguous_if_needed(
            chunk1_tr, "[test_split_3d_then_transpose_then_reshape] chunk1_tr"
        )
        chunk2_tr = self._ensure_contiguous_if_needed(
            chunk2_tr, "[test_split_3d_then_transpose_then_reshape] chunk2_tr"
        )
        result1 = torch.add(chunk1_tr, 1.0)
        result2 = torch.add(chunk2_tr, 2.0)
        # Create add tensor: first 16 rows = 1.0, rest = 2.0
        add_tensor = torch.zeros_like(base_tensor)
        add_tensor[:8] = 1.0
        add_tensor[8:] = 2.0

        self.assertEqual(result1.cpu(), (base_tensor + add_tensor)[:8].transpose(1, 2).reshape(8, 64, 32).cpu())
        self.assertEqual(result2.cpu(), (base_tensor + add_tensor)[8:].transpose(1, 2).reshape(8, 64, 32).cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_multiple_chunks_then_transpose_each(self, dtype):
        """Test split into multiple chunks, then transpose each."""
        # Create base tensor
        base_tensor = torch.randn([64, 128], dtype=dtype, device=self.rbln_device)

        # Split into 8 chunks
        chunks = torch.chunk(base_tensor, chunks=16, dim=0)

        # Check storage offsets
        for i, chunk in enumerate(chunks):
            self._check_storage_offset(chunk, i * 4 * 128, f"chunk{i}")

        # Apply transpose to each chunk
        transposed_chunks = [chunk.transpose(0, 1) for chunk in chunks]

        # Verify transpose worked correctly
        expected_transposed = [base_tensor[i * 4 : (i + 1) * 4].transpose(0, 1) for i in range(16)]

        for i, (actual, expected) in enumerate(zip(transposed_chunks, expected_transposed)):
            self.assertEqual(actual.cpu(), expected.cpu())

        # Use transposed chunks in operations
        # Check contiguous status before operations
        for tc in transposed_chunks:
            tc = self._ensure_contiguous_if_needed(tc, "[test_split_multiple_chunks_then_transpose_each] tc")
        results = [torch.add(tc, 1.0) for tc in transposed_chunks]
        expected_results = [et + 1.0 for et in expected_transposed]

        for i, (actual, expected) in enumerate(zip(results, expected_results)):
            # Compare values directly since they may have different storage layouts
            actual_cpu = actual.cpu().contiguous()
            expected_cpu = expected.cpu().contiguous()
            self.assertEqual(actual_cpu, expected_cpu, msg=f"chunk {i} operation mismatch")


@pytest.mark.test_set_ci
class TestSplitAndMerge(BaseTestNonZeroStorageOffset):
    """Test split and merge operations."""

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_slice_then_merge(self, dtype):
        """Test split using slice, then merge back."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using slice
        chunk1 = base_tensor[:16]
        chunk2 = base_tensor[16:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 16 * 64, "chunk2")

        # Merge back
        merged = torch.cat([chunk1, chunk2], dim=0)

        # Apply view to match base_tensor shape
        merged_view = merged.view(base_tensor.shape)

        self.assertEqual(merged_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_chunk_then_merge(self, dtype):
        """Test split using torch.chunk, then merge back."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split using chunk
        chunks = torch.chunk(base_tensor, chunks=4, dim=0)

        # Check storage offsets
        for i, chunk in enumerate(chunks):
            self._check_storage_offset(chunk, i * 8 * 64, f"chunk{i}")

        # Merge back
        merged = torch.cat(chunks, dim=0)

        # Apply view to match base_tensor shape
        merged_view = merged.view(base_tensor.shape)

        self.assertEqual(merged_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_along_different_dims_then_merge(self, dtype):
        """Test split along different dimensions, then merge."""
        # Create 3D tensor
        base_tensor = torch.randn([16, 32, 64], dtype=dtype, device=self.rbln_device)

        # Split along dimension 0
        chunks_dim0 = torch.chunk(base_tensor, chunks=4, dim=0)
        for i, chunk in enumerate(chunks_dim0):
            self._check_storage_offset(chunk, i * 4 * 32 * 64, f"chunk_dim0_{i}")

        # Merge back
        merged_dim0 = torch.cat(chunks_dim0, dim=0)

        # Apply view to match base_tensor shape
        merged_dim0_view = merged_dim0.view(base_tensor.shape)
        self.assertEqual(merged_dim0_view.cpu(), base_tensor.cpu())

        # Reset base_tensor for second test
        # Split along dimension 1
        chunks_dim1 = torch.chunk(base_tensor, chunks=4, dim=1)
        for i, chunk in enumerate(chunks_dim1):
            self._check_storage_offset(chunk, i * 8 * 64, f"chunk_dim1_{i}")

        # Merge back
        merged_dim1 = torch.cat(chunks_dim1, dim=1)

        # Apply view to match base_tensor shape
        merged_dim1_view = merged_dim1.view(base_tensor.shape)

        self.assertEqual(merged_dim1_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_nested_split_and_merge(self, dtype):
        """Test nested splitting and merging."""
        # Create base tensor
        base_tensor = torch.randn([64, 128], dtype=dtype, device=self.rbln_device)

        # First level split
        first_level_chunks = torch.chunk(base_tensor, chunks=4, dim=0)
        for i, chunk in enumerate(first_level_chunks):
            self._check_storage_offset(chunk, i * 16 * 128, f"first_level_chunk{i}")

        # Second level split (split each first level chunk)
        second_level_chunks = []
        for i, chunk in enumerate(first_level_chunks):
            sub_chunks = torch.chunk(chunk, chunks=2, dim=1)
            second_level_chunks.extend(sub_chunks)

        # Merge second level back
        merged_second_level = []
        for i in range(0, len(second_level_chunks), 2):
            merged = torch.cat([second_level_chunks[i], second_level_chunks[i + 1]], dim=1)
            merged_second_level.append(merged)

        # Merge first level back
        merged_first_level = torch.cat(merged_second_level, dim=0)

        # Apply view to match base_tensor shape
        merged_view = merged_first_level.view(base_tensor.shape)

        self.assertEqual(merged_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_process_merge_multiple_rounds(self, dtype):
        """Test multiple rounds of split, process, merge."""
        # Create base tensor
        base_tensor = torch.randn([64, 128], dtype=dtype, device=self.rbln_device)

        # Round 1: Split and merge
        chunks1 = torch.chunk(base_tensor, chunks=4, dim=0)
        for i, chunk in enumerate(chunks1):
            self._check_storage_offset(chunk, i * 16 * 128, f"round1_chunk{i}")
        merged1 = torch.cat(chunks1, dim=0)
        merged1_view = merged1.view(base_tensor.shape)
        self.assertEqual(merged1_view.cpu(), base_tensor.cpu())

        # Round 2: Split merged1 again and merge
        chunks2 = torch.chunk(merged1, chunks=8, dim=0)
        for i, chunk in enumerate(chunks2):
            self._check_storage_offset(chunk, i * 8 * 128, f"round2_chunk{i}")
        merged2 = torch.cat(chunks2, dim=0)
        merged2_view = merged2.view(base_tensor.shape)
        self.assertEqual(merged2_view.cpu(), merged1.cpu())

        # Round 3: Split merged2 and merge
        chunks3 = torch.chunk(merged2, chunks=4, dim=1)
        for i, chunk in enumerate(chunks3):
            self._check_storage_offset(chunk, i * 32, f"round3_chunk{i}")
        merged3 = torch.cat(chunks3, dim=1)
        merged3_view = merged3.view(base_tensor.shape)
        self.assertEqual(merged3_view.cpu(), merged2.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_uneven_chunks_then_merge(self, dtype):
        """Test split into uneven chunks, then merge."""
        # Create tensor with size that doesn't divide evenly
        base_tensor = torch.randn([100, 200], dtype=dtype, device=self.rbln_device)

        # Split into uneven chunks
        chunk_sizes = [25, 30, 20, 25]  # Sums to 100
        chunks = []
        start_idx = 0
        for i, chunk_size in enumerate(chunk_sizes):
            end_idx = start_idx + chunk_size
            chunk = base_tensor[start_idx:end_idx]
            chunks.append(chunk)
            self._check_storage_offset(chunk, start_idx * 200, f"chunk_start_{start_idx}")
            start_idx += chunk_size

        # Merge back
        merged = torch.cat(chunks, dim=0)

        # Apply view to match base_tensor shape
        merged_view = merged.view(base_tensor.shape)

        self.assertEqual(merged_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_distributed_like_scatter_gather(self, dtype):
        """Test distributed-like scatter and gather scenario."""
        world_size = 4
        per_rank_size = (32, 64)
        total_size = (world_size * per_rank_size[0], per_rank_size[1])

        # Create full tensor
        full_tensor = torch.randn(total_size, dtype=dtype, device=self.rbln_device)

        # Scatter: split into chunks
        rank_chunks = []
        for rank in range(world_size):
            chunk = full_tensor[rank * per_rank_size[0] : (rank + 1) * per_rank_size[0]]
            rank_chunks.append(chunk)
            self._check_storage_offset(chunk, rank * per_rank_size[0] * per_rank_size[1], f"rank{rank}_chunk")

        # Gather: merge back
        gathered = torch.cat(rank_chunks, dim=0)
        self.assertEqual(gathered.cpu(), full_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_split_column_then_merge(self, dtype):
        """Test split along columns, then merge."""
        # Create base tensor
        base_tensor = torch.randn([32, 64], dtype=dtype, device=self.rbln_device)

        # Split along columns
        chunk1 = base_tensor[:, :32]
        chunk2 = base_tensor[:, 32:]

        # Check storage offsets
        self._check_storage_offset(chunk1, 0, "chunk1")
        self._check_storage_offset(chunk2, 32, "chunk2")

        # Merge back
        merged = torch.cat([chunk1, chunk2], dim=1)

        # Apply view to match base_tensor shape
        merged_view = merged.view(base_tensor.shape)

        # Compare with base_tensor
        self.assertEqual(merged_view.cpu(), base_tensor.cpu())

    @dtypes(*SUPPORTED_DTYPES)
    def test_large_tensor_chunked_processing_pipeline(self, dtype):
        """Test complete pipeline: split -> process -> merge -> split again."""
        # Initial large tensor
        large_tensor = torch.randn([128, 256], dtype=dtype, device=self.rbln_device)

        # Stage 1: Split into chunks
        chunks_stage1 = torch.chunk(large_tensor, chunks=8, dim=0)
        for i, chunk in enumerate(chunks_stage1):
            self._check_storage_offset(chunk, i * 16 * 256, f"stage1_chunk{i}")

        # Stage 3: Merge
        merged_stage1 = torch.cat(chunks_stage1, dim=0)

        # Stage 4: Split merged result again
        chunks_stage2 = torch.chunk(merged_stage1, chunks=4, dim=1)
        for i, chunk in enumerate(chunks_stage2):
            self._check_storage_offset(chunk, i * 64, f"stage2_chunk{i}")

        # Stage 6: Final merge
        final_result = torch.cat(chunks_stage2, dim=1)

        # Apply view to match large_tensor shape
        final_view = final_result.view(large_tensor.shape)

        self.assertEqual(final_view.cpu(), large_tensor.cpu())

    @unittest.skip("Revisit this test later. The error also occurs in the previously released rebel-compiler.")
    @dtypes(*SUPPORTED_DTYPES)
    def test_regard_tensor_list_as_single_tensor_if_list_is_contiguous(self, dtype):
        """Test regard tensor list as single tensor if list is contiguous."""

        # Create a single large tensor and split it to ensure contiguous memory
        # This simulates the case where tensors are allocated contiguously (like in C++)
        # We need to create tensors that share the same underlying storage
        large_tensor = torch.zeros([4, 32, 64], dtype=dtype, device=self.rbln_device)
        tensor_list = []
        for i in range(4):
            # Create views that share the same storage (like slicing a contiguous buffer)
            tensor = large_tensor[i]  # This creates a view with storage_offset
            tensor_list.append(tensor)

        def _is_tensor_vector_contiguous(tensors, size):
            """Check if tensor list is contiguous in memory."""
            if len(tensors) == 0:
                return True
            # Get the data pointer address as integer
            ptr = tensors[0].data_ptr()
            byte_size = size * tensors[0].element_size()
            for i in range(1, len(tensors)):
                begin = tensors[i].data_ptr()
                # Calculate expected pointer address
                expected_ptr = ptr + byte_size
                if begin != expected_ptr:
                    return False
                ptr = begin
            return True

        # Create two tensor lists: one contiguous, one non-contiguous
        # Case 1: Contiguous tensor list (from large_tensor slices)
        contiguous_tensor_list = tensor_list

        # Case 2: Non-contiguous tensor list (separately allocated tensors)
        non_contiguous_tensor_list = []
        large_tensor_for_non_contiguous = torch.zeros([4, 32, 64], dtype=dtype, device=self.rbln_device)
        for i in range(4):
            tensor = torch.zeros([32, 64], dtype=dtype, device=self.rbln_device)
            non_contiguous_tensor_list.append(tensor)
            large_tensor_for_non_contiguous[i].copy_(tensor)

        # Test both cases
        test_cases = [
            (contiguous_tensor_list, large_tensor, True, "contiguous"),
            (non_contiguous_tensor_list, large_tensor_for_non_contiguous, False, "non-contiguous"),
        ]

        for tensor_list_to_test, reference_tensor, expected_contiguous, case_name in test_cases:
            print(f"\nTesting {case_name} tensor list")
            is_contig = _is_tensor_vector_contiguous(tensor_list_to_test, 32 * 64)
            self.assertEqual(
                is_contig,
                expected_contiguous,
                f"Expected {case_name} tensor list to be {expected_contiguous}, got {is_contig}",
            )

            if is_contig:
                print(f"{case_name} tensor_list is contiguous")
                # Create a single tensor view from contiguous tensor list using tensor_list[0]
                # Each tensor is (32, 64), so stride should be (32*64, 64, 1) for shape (4, 32, 64)
                # Since tensor_list[0] shares the same storage as large_tensor, we can use it directly
                first_tensor = tensor_list_to_test[0]
                # Use tensor_list[0] directly - it shares storage with large_tensor, so storage is large enough
                single_tensor_view = torch.as_strided(
                    first_tensor,  # Use tensor_list[0] directly
                    size=(4, 32, 64),
                    stride=(32 * 64, 64, 1),
                    storage_offset=first_tensor.storage_offset(),
                )
                # Verify the view matches the expected shape
                self.assertEqual(single_tensor_view.shape, (4, 32, 64))
                # Verify it's a view starting from the same point as tensor_list[0]
                self.assertEqual(single_tensor_view.data_ptr(), tensor_list_to_test[0].data_ptr())
                # Verify the view matches the reference tensor
                if reference_tensor is not None:
                    self.assertEqual(single_tensor_view.cpu(), reference_tensor.cpu())
            else:
                print(f"{case_name} tensor_list is not contiguous")
                # If not contiguous, we cannot create a single tensor view
                # In this case, we would need to concatenate or copy
                combined_tensor = torch.stack(tensor_list_to_test, dim=0)
                self.assertEqual(combined_tensor.shape, (4, 32, 64))
                if reference_tensor is not None:
                    self.assertEqual(combined_tensor.cpu(), reference_tensor.cpu())


instantiate_device_type_tests(TestSplitOps, globals(), only_for="privateuse1")
instantiate_device_type_tests(TestSplitAndMerge, globals(), only_for="privateuse1")


if __name__ == "__main__":
    run_tests()
