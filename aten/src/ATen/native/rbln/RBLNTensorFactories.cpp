#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

namespace at::native::rbln {

at::Tensor empty_rbln(
    c10::IntArrayRef sizes,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  RBLN_SCOPE_GUARD();
  const auto dtype = c10::dtype_or_default(dtype_opt);
  const auto layout = c10::layout_or_default(layout_opt);
  const auto device = c10::device_or_default(device_opt);
  const auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  const auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  RBLN_LOG_DEBUG(
      "sizes={}, dtype={}, layout={}, device={}, pin_memory={}, memory_format={}",
      c10::str(sizes),
      c10::str(dtype),
      c10::str(layout),
      c10::str(device),
      pin_memory,
      c10::str(memory_format));
  RBLN_CHECK(layout == c10::kStrided, "Only Strided layout is supported, but got {}", c10::str(layout));
  RBLN_CHECK(device.is_privateuseone(), "Only privateuseone device is supported, but got {}", c10::str(device));
  RBLN_CHECK(!pin_memory, "Pinned memory is not supported");

  const auto device_guard = c10::DeviceGuard(device);
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  constexpr auto dispatch_key_set = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  const at::Tensor out = at::detail::empty_generic(sizes, allocator, dispatch_key_set, dtype, memory_format);
  RBLN_LOG_DEBUG("out_data={}", fmt::ptr(out.data_ptr()));
  return out;
}

at::Tensor empty_strided_rbln(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  RBLN_SCOPE_GUARD();
  const auto dtype = c10::dtype_or_default(dtype_opt);
  const auto layout = c10::layout_or_default(layout_opt);
  const auto device = c10::device_or_default(device_opt);
  const auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);
  RBLN_LOG_DEBUG(
      "sizes={}, strides={}, dtype={}, layout={}, device={}, pin_memory={}",
      c10::str(sizes),
      c10::str(strides),
      c10::str(dtype),
      c10::str(layout),
      c10::str(device),
      pin_memory);
  RBLN_CHECK(layout == c10::kStrided, "Only Strided layout is supported, but got {}", c10::str(layout));
  RBLN_CHECK(device.is_privateuseone(), "Only privateuseone device is supported, but got {}", c10::str(device));
  RBLN_CHECK(!pin_memory, "Pinned memory is not supported");

  const auto device_guard = c10::DeviceGuard(device);
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  constexpr auto dispatch_key_set = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  const at::Tensor out = at::detail::empty_strided_generic(sizes, strides, allocator, dispatch_key_set, dtype);
  RBLN_LOG_DEBUG("out_data={}", fmt::ptr(out.data_ptr()));
  return out;
}

at::Tensor _efficientzerotensor_rbln(
    c10::SymIntArrayRef sizes_sym,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  RBLN_SCOPE_GUARD();
  // Materialize SymInts to int64. Eager-mode RBLN doesn't generate symbolic
  // sizes, so this is always concrete — fall back to TORCH_CHECK if a real
  // SymInt sneaks in.
  std::vector<int64_t> sizes;
  sizes.reserve(sizes_sym.size());
  for (const auto& s : sizes_sym) {
    sizes.push_back(s.guard_int(__FILE__, __LINE__));
  }
  // Allocate fresh privateuse1 storage and mark its v-memory as zero-init.
  // `mark_zeros` flips the EMPTY_INIT_WITH_ZERO flag on the v-memory entry —
  // no host allocation, no D→H copy, no actual write. Zeros materialise lazily
  // on the first NPU read (or are skipped entirely when the first access is a
  // write, e.g. KV-cache output). This mirrors what `aten::zero_` already
  // does on RBLN via `custom_zero__rbln`.
  auto rbln_out = empty_rbln(sizes, dtype_opt, layout_opt, device_opt,
                             pin_memory_opt,
                             /*memory_format_opt=*/std::nullopt);
  if (rbln_out.numel() == 0) {
    return rbln_out;
  }
  c10::rbln::mark_zeros(rbln_out.data_ptr());
  return rbln_out;
}

at::Tensor& zero_rbln_(at::Tensor& self) {
  RBLN_SCOPE_GUARD();
  if (self.numel() == 0) {
    return self;
  }
  c10::rbln::mark_zeros(self.data_ptr());
  return self;
}

} // namespace at::native::rbln
