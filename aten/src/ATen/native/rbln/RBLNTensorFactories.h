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
 * @brief In-place fill of an RBLN tensor with a scalar value.
 *
 * Acquires the host-side mirror of the v-memory backing via
 * `acquire_host_ptr_for_overwrite` (no D2H sync — the entire region is about
 * to be overwritten), runs `std::fill_n` to materialise the constant on the
 * host buffer, then returns the borrow with `updated=true` so the next device
 * consumer triggers a lazy host→device sync.
 *
 * Replaces the cpu_fallback path for `aten::fill_.Scalar` (3k+ calls per
 * LLaMA-1B decode run before this op was native), removing the per-call
 * cpu_fallback_rbln setup (~80 us) and the schema-cache lookup.
 */
at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value);

/**
 * @brief In-place fill of an RBLN tensor with a 0-dim tensor value.
 *
 * Same as `fill_scalar_rbln_` but the fill value is provided as a 0-dim
 * tensor (the `aten::fill_.Tensor` overload). Extracts the scalar via
 * `value.item<scalar_t>()` then dispatches to the scalar path.
 *
 * Note: `value.item()` on an RBLN 0-dim tensor still triggers
 * `_local_scalar_dense` cpu_fallback. Callers who can avoid building a 0-dim
 * tensor in the first place (and call `fill_scalar_rbln_` directly) save the
 * extraction round-trip; on the fallback path the round-trip is unavoidable.
 */
at::Tensor& fill_tensor_rbln_(at::Tensor& self, const at::Tensor& value);

} // namespace at::native::rbln
