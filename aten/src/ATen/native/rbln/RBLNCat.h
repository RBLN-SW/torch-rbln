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
 * Restrictions vs PyTorch upstream cat:
 *   - All inputs and `out` must share the same dtype. Upstream cat performs
 *     type promotion on mixed-dtype inputs; this kernel errors out instead.
 *     Callers needing promotion should `.to(common_dtype)` before cat.
 *   - All inputs and `out` must live on the same RBLN device.
 *   - `out` must be contiguous (resized to the cat result shape if needed).
 *   - No input may overlap `out` (checked via `at::assert_no_overlap`).
 *
 * @param tensors  Input tensors (all on the same RBLN device, same dtype).
 * @param dim      Concatenation axis (may be negative).
 * @param out      Pre-allocated contiguous output tensor on the same device.
 *                 Resized to the resulting shape if needed.
 * @return reference to `out`.
 */
at::Tensor& cat_out_rbln(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out);

} // namespace at::native::rbln
