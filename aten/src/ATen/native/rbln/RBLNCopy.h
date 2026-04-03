#pragma once

#include <ATen/native/Copy.h>

namespace at::native::rbln {

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * @param src The source tensor.
 * @param dst The destination tensor.
 * @param non_blocking If true, the copy is performed asynchronously.
 * @return The destination tensor with copied data.
 *
 * @note The source tensor data is converted to match the dtype of the destination tensor during the copy.
 */
at::Tensor _copy_from_rbln(const at::Tensor& src, const at::Tensor& dst, bool non_blocking);

/**
 * @brief Resizes the destination tensor to match the source tensor and copies data from the source tensor to the destination tensor.
 *
 * @param src The source tensor.
 * @param dst The destination tensor.
 * @return The destination tensor with copied data.
 */
at::Tensor _copy_from_and_resize_rbln(const at::Tensor& src, const at::Tensor& dst);

} // namespace at::native::rbln
