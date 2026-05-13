#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::rbln {

/**
 * @brief RBLN-native index_select.out using device-to-device memcpy (v2v).
 *
 * Decomposes the gather into a sequence of contiguous slice copies. The
 * `index` tensor is read on host (if it lives on the device, it is fetched
 * via v2h before the loop). Consecutive index runs are coalesced into a
 * single v2v call to reduce per-call overhead.
 *
 * @param self  Source tensor on RBLN device.
 * @param dim   Gather axis.
 * @param index 1-D index tensor (any int dtype; CPU or RBLN).
 * @param out   Pre-allocated contiguous output tensor.
 * @return reference to `out`.
 */
at::Tensor& index_select_out_rbln(
    const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out);

/// Non-out overload: allocates a fresh contiguous output of the right shape
/// then delegates to `index_select_out_rbln`.
at::Tensor index_select_rbln(const at::Tensor& self, int64_t dim, const at::Tensor& index);

} // namespace at::native::rbln
