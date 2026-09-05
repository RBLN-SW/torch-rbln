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
 * When `self` spans its whole backing allocation, marks the v-memory as
 * EMPTY_INIT_WITH_ZERO via `mark_zeros`: no host buffer, no transfer — zeros
 * materialise lazily on the first NPU read, or are skipped when the first
 * access is a write (KV-cache pattern). Partial/offset views (e.g. `base[2:4]`)
 * route through `fill_scalar_rbln_(self, 0)`, since `mark_zeros` has no
 * offset/size and would zero the enclosing allocation.
 */
at::Tensor& zero_rbln_(at::Tensor& self);

/**
 * @brief ``torch_rbln::bind_device_memory_at``: materialise ``self``'s device allocation on
 *        ``chiplet``. ``self`` must cover its whole storage (contiguous, storage offset 0).
 *
 * The dispatcher-reachable form of c10::rbln::bind_device_memory_at(), for callers that link
 * only ATen. See that function for the placement semantics.
 */
void bind_device_memory_at_rbln(at::Tensor& self, int64_t chiplet);

/**
 * @brief ``torch_rbln::chiplet_count``: chiplets per NPU of ``self``'s device.
 */
int64_t chiplet_count_rbln(const at::Tensor& self);

// Native impl of aten::fill_.Scalar — borrow host pointer + typed std::fill_n
// + return_borrowed(updated=true). Bypasses cpu_fallback_rbln's redispatchBoxed
// + TensorIterator path. The fast path is taken for a contiguous self with a
// supported dtype; broadcast-overlap (stride-0) views collapse to a non-overlapping
// view, and non-contiguous or unsupported-dtype self routes through the CPU
// fallback (no hard assert).
at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value);

// Native impl of aten::arange.start_out — acquire_host_ptr_for_overwrite
// (D2H skipped, write-only), host-fill out[i] = start + i*step, then
// return_borrowed(updated=true). `out` is assumed already sized.
at::Tensor& arange_start_out_rbln(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out);

// Native impl of aten::_local_scalar_dense — 1-element tensor → Python scalar
// (= `.item()`). D2H is unavoidable (we need the value on host) but we skip
// cpu_fallback_rbln's schema cache + redispatch overhead. Read-only borrow.
at::Scalar _local_scalar_dense_rbln(const at::Tensor& self);

} // namespace at::native::rbln
