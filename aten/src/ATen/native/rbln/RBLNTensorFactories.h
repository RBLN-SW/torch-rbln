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

/**
 * @brief RBLN-native impl of `aten::_efficientzerotensor`.
 *
 * Returns an RBLN tensor with the requested shape/dtype that reads as all
 * zeros. The CPU fallback path crashes when redispatching this op (no tensor
 * inputs but a Device IValue, see RBLNCPUFallback redispatchBoxed) — handling
 * it directly here lets `sgn_backward`-style autograd paths return zero
 * gradients without going through cpu_fallback_rbln.
 */
at::Tensor _efficientzerotensor_rbln(
  c10::SymIntArrayRef sizes,
  std::optional<c10::ScalarType> dtype_opt,
  std::optional<c10::Layout> layout_opt,
  std::optional<c10::Device> device_opt,
  std::optional<bool> pin_memory_opt);

/**
 * @brief In-place zero of an RBLN tensor.
 *
 * Marks the v-memory backing as EMPTY_INIT_WITH_ZERO via `mark_zeros`. No
 * host buffer is allocated, no host-to-device transfer is issued, no actual
 * write happens — zeros materialise lazily on the first NPU read, or are
 * skipped entirely when the first access is a write (KV-cache output
 * pattern). Replaces the prior Python `custom_zero__rbln` shim.
 */
at::Tensor& zero_rbln_(at::Tensor& self);

} // namespace at::native::rbln
