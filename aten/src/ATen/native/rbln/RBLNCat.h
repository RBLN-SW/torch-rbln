#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>

namespace at::native::rbln {

/**
 * @brief RBLN-native cat.out implementation using device-to-device memcpy (v2v).
 *
 * Bypasses rebel-compiler entirely; cat is reduced to a sequence of contiguous
 * v2v copies derived from each input's stride structure. Non-contiguous inputs
 * are not materialised — instead we iterate the outer (non-contiguous) dims
 * and v2v the innermost contiguous run of each input slab into the output.
 *
 * @param tensors  Input tensors (all on the same RBLN device, same dtype).
 * @param dim      Concatenation axis (may be negative).
 * @param out      Pre-allocated contiguous output tensor on the same device.
 *                 Resized to the resulting shape if needed.
 * @return reference to `out`.
 */
at::Tensor& cat_out_rbln(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out);

} // namespace at::native::rbln
