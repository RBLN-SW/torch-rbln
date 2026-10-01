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
 * A dense `self` (one byte range from its data pointer, whatever its offset or dim order)
 * is zero-filled on the device with `fill_zeros`; any other view routes through
 * `fill_scalar_rbln_(self, 0)`.
 */
at::Tensor& zero_rbln_(at::Tensor& self);

/**
 * @brief Native impl of aten::fill_.Scalar, without cpu_fallback_rbln's redispatchBoxed +
 * TensorIterator path.
 *
 * A dense self is zero-filled on the device, or written from a host pattern for any other
 * value; a strided view is written run by run from one host pattern; broadcast-overlap
 * (stride-0) views collapse to a non-overlapping view first. Unsupported dtypes and views
 * the device paths decline go through a CPU tensor.
 */
at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value);

/**
 * @brief Native impl of aten::arange.start_out: computes out[i] = start + i*step on the
 * host and writes it with one host-to-device copy. `out` is resized to the arange length.
 */
at::Tensor& arange_start_out_rbln(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out);

/**
 * @brief Native impl of aten::_local_scalar_dense (`.item()`): one device-to-host copy of
 * the element, without cpu_fallback_rbln's schema cache + redispatch overhead.
 */
at::Scalar _local_scalar_dense_rbln(const at::Tensor& self);

} // namespace at::native::rbln
