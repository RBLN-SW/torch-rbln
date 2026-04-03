#pragma once

#include <ATen/native/Resize.h>

namespace at::native::rbln {

/**
 * @brief Resize a tensor in-place, preserving its data when possible.
 *
 * @param self The tensor to be resized.
 * @param sizes The new shape of the tensor.
 * @param memory_format_opt Optional memory format for the resized tensor.
 * @return The resized tensor.
 */
const at::Tensor& resize_rbln_(const at::Tensor& self, c10::IntArrayRef sizes, std::optional<c10::MemoryFormat> memory_format_opt);

/**
 * @brief Set the storage of a tensor in-place, preserving its data when possible.
 *
 * @param result The tensor whose storage is to be set.
 * @param storage The new storage to be assigned to the tensor.
 * @param storage_offset The offset within the storage.
 * @param sizes The new shape of the tensor.
 * @param strides The new strides of the tensor.
 * @return The tensor with updated storage.
 */
at::Tensor& set_storage_rbln_(
    at::Tensor& result,
    c10::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides);

} // namespace at::native::rbln
