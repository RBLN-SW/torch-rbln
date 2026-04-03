#pragma once

#include <ATen/native/TensorFactories.h>

namespace at::native::rbln {

/**
 * @brief Returns a tensor filled with uninitialized data.
 *
 * @param sizes The shape of the returned tensor.
 * @param dtype_opt The desired data type of the returned tensor.
 * @param layout_opt The desired layout of the returned Tensor.
 * @param device_opt The desired device of the returned tensor.
 * @param pin_memory_opt If set, the returned tensor would be allocated in the pinned memory.
 * @param memory_format_opt The desired memory format of the returned tensor.
 * @return An uninitialized tensor with the specified properties.
 */
at::Tensor empty_rbln(
  c10::IntArrayRef sizes,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt);

/**
 * @brief Returns a tensor filled with uninitialized data.
 *
 * @param sizes The shape of the returned tensor.
 * @param strides The strides of the returned tensor.
 * @param dtype_opt The desired data type of the returned tensor.
 * @param layout_opt The desired layout of the returned Tensor.
 * @param device_opt The desired device of the returned tensor.
 * @param pin_memory_opt If set, the returned tensor would be allocated in the pinned memory.
 * @return An uninitialized tensor with the specified properties.
 */
at::Tensor empty_strided_rbln(
  c10::IntArrayRef sizes,
  c10::IntArrayRef strides,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt);

} // namespace at::native::rbln
