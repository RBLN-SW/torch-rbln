#pragma once

#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <vector>

namespace at::native::rbln {

/**
 * @brief Return the smallest dim `j` such that the tensor's suffix [j, rank) is
 * contiguous in memory.
 *
 * Size-1 dims are treated as free passes that do not break contiguity (their
 * stride is irrelevant since they never iterate). A return value of 0 means
 * the whole tensor is contiguous. Used by v2v kernels to find the inner
 * contiguous slab that can be copied as one memcpy.
 *
 * NOTE: This helper does NOT special-case stride==0 broadcast dims. Callers
 * that want broadcast dims forced into the outer iteration (so each broadcast
 * position emits its own copy) should use `common_inner_start` below, which
 * additionally rejects stride==0 on non-size-1 dims.
 */
inline int64_t contig_suffix_start(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  const int64_t rank = static_cast<int64_t>(sizes.size());
  if (rank == 0)
    return 0;
  int64_t expected_stride = 1;
  int64_t j = rank;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (sizes[i] == 1) {
      j = i;
      continue;
    }
    if (strides[i] == expected_stride) {
      expected_stride *= sizes[i];
      j = i;
    } else {
      break;
    }
  }
  return j;
}

/**
 * @brief Return the smallest dim `j` such that the suffix [j, rank) is
 * jointly contiguous in BOTH src and dst, treating stride==0 broadcast dims
 * (on non-size-1 axes) as non-contiguous.
 *
 * This is the boundary that v2v engine code uses to split a copy into:
 *   - an inner block of `prod(sizes[j..rank])` elements, which is one v2v
 *   - outer iteration over `sizes[0..j]`, which is K = prod of those dims
 *
 * Forcing stride-0 dims to the outer loop is deliberate: an expand()-broadcast
 * tensor reads the same source memory N times, and we want each replication
 * to be its own v2v write so the destination is faithfully filled. Absorbing
 * a stride-0 dim into the inner block would silently copy only the first slab.
 */
inline int64_t common_inner_start(
    c10::IntArrayRef sizes,
    c10::IntArrayRef src_strides,
    c10::IntArrayRef dst_strides) {
  const int64_t rank = static_cast<int64_t>(sizes.size());
  if (rank == 0)
    return 0;
  int64_t expected = 1;
  int64_t j = rank;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (sizes[i] == 1) {
      // size-1 dim: stride is irrelevant (never iterated). Free pass.
      j = i;
      continue;
    }
    // Reject stride-0 broadcast on non-size-1 dims.
    if (src_strides[i] == 0 || dst_strides[i] == 0) {
      break;
    }
    if (src_strides[i] == expected && dst_strides[i] == expected) {
      expected *= sizes[i];
      j = i;
    } else {
      break;
    }
  }
  return j;
}

/**
 * @brief Increment a flat outer index across an outer-shape, in row-major
 * order.
 *
 * Returns false when iteration is exhausted (idx wraps back to all zeros).
 */
inline bool advance_multi_index(std::vector<int64_t>& idx, c10::IntArrayRef outer_sizes) {
  for (int64_t d = static_cast<int64_t>(outer_sizes.size()) - 1; d >= 0; --d) {
    if (++idx[d] < outer_sizes[d])
      return true;
    idx[d] = 0;
  }
  return false;
}

} // namespace at::native::rbln
