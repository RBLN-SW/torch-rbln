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
 * @brief Increment a flat outer index across an outer-shape, in row-major
 * order.
 *
 * Returns false when iteration is exhausted (idx wraps back to all zeros).
 * Used by v2v kernels to walk the non-contiguous outer dims while the inner
 * contiguous block is copied per step.
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
