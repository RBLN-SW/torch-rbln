#include <ATen/Dispatch_v2.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

#include <algorithm>

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

at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value) {
  RBLN_SCOPE_GUARD();
  RBLN_CHECK(self.device().is_privateuseone(),
             "fill_scalar_rbln_ expected an RBLN tensor, got device={}",
             c10::str(self.device()));
  const auto numel = self.numel();
  if (numel == 0) {
    return self;
  }
  // Non-contiguous self with a stride that exceeds the contiguous size would
  // require writing through gaps; not worth special-casing here since the
  // observed callers (vllm-rbln decode-step block_tables / slot_mapping init,
  // sampler scratch) all hand contiguous tensors. Loud assert so any future
  // non-contig caller is caught.
  RBLN_CHECK(self.is_contiguous(),
             "fill_scalar_rbln_ requires contiguous self; got strides={}, sizes={}",
             c10::str(self.strides()), c10::str(self.sizes()));
  const auto nbytes = self.nbytes();
  auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(self.data_ptr(), nbytes);
  AT_DISPATCH_V2(
      self.scalar_type(),
      "fill_scalar_rbln_",
      AT_WRAP([&] {
        std::fill_n(reinterpret_cast<scalar_t*>(borrowed.host_ptr),
                    static_cast<size_t>(numel),
                    value.to<scalar_t>());
      }),
      AT_EXPAND(AT_ALL_TYPES),
      AT_EXPAND(AT_FLOAT8_TYPES),
      kHalf,
      kBFloat16,
      kBool);
  // updated=true: the next device consumer (or a v->h sync) sees the new
  // host bytes as authoritative.
  c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
  return self;
}

at::Tensor& fill_tensor_rbln_(at::Tensor& self, const at::Tensor& value) {
  RBLN_SCOPE_GUARD();
  RBLN_CHECK(value.dim() == 0,
             "fill_.Tensor expects a 0-dim value tensor, got dim={}", value.dim());
  // Extract the scalar (may trigger _local_scalar_dense cpu_fallback if value
  // lives on rbln; that round-trip is intrinsic to the fill_.Tensor overload
  // and not something this native path can avoid).
  return fill_scalar_rbln_(self, value.item());
}

} // namespace at::native::rbln
