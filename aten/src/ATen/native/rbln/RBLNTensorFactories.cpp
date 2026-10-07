#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <ATen/ops/zeros.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHostBatch.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNTyped.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

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
  auto rbln_out = empty_rbln(
      sizes,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      /*memory_format_opt=*/std::nullopt);
  zero_rbln_(rbln_out);
  return rbln_out;
}

at::Tensor& zero_rbln_(at::Tensor& self) {
  RBLN_SCOPE_GUARD();
  if (self.numel() == 0) {
    return self;
  }
  // A tensor of a type zeroes its bytes when a zero element is zero bytes, and is otherwise written
  // zeros encoded on the host (see c10/rbln/RBLNTyped.h).
  std::optional<c10::rbln::typed::AsTyped> as_typed;
  if (c10::rbln::caching::any_typed() && !c10::rbln::caching::locating_as_typed()) {
    auto found = c10::rbln::caching::try_locate_typed(self.const_data_ptr());
    if (found && found->type) {
      if (!found->type->zero_is_zero_bytes) {
        self.copy_(at::zeros(self.sizes(), self.options().device(at::kCPU)));
        return self;
      }
      as_typed.emplace();
    }
  }
  // A dense view covers one byte range from data_ptr(), whatever its offset or dim order;
  // anything else is filled run by run.
  if (!self.is_non_overlapping_and_dense()) {
    return fill_scalar_rbln_(self, 0);
  }
  c10::rbln::fill_zeros(self.data_ptr(), self.nbytes());
  return self;
}

namespace {
// Dispatch a scalar value into the appropriate typed std::fill_n over a host
// buffer of given element count. We support the dtypes vllm-rbln actually hits
// on Llama-class workloads (slot_mapping=int64, masks=bool, positions=int64,
// fp16/bf16/fp32 activations). Adding more is a one-line case extension.
bool fill_host_typed(void* host_ptr, int64_t numel, c10::ScalarType st, const at::Scalar& value) {
  switch (st) {
    case at::kLong:
      std::fill_n(static_cast<int64_t*>(host_ptr), numel, value.to<int64_t>());
      return true;
    case at::kInt:
      std::fill_n(static_cast<int32_t*>(host_ptr), numel, value.to<int32_t>());
      return true;
    case at::kShort:
      std::fill_n(static_cast<int16_t*>(host_ptr), numel, value.to<int16_t>());
      return true;
    case at::kChar:
      std::fill_n(static_cast<int8_t*>(host_ptr), numel, value.to<int8_t>());
      return true;
    case at::kByte: {
      const auto v = value.to<uint8_t>();
      std::memset(host_ptr, v, static_cast<size_t>(numel));
      return true;
    }
    case at::kBool: {
      const auto v = value.to<bool>();
      std::memset(host_ptr, v ? 1 : 0, static_cast<size_t>(numel));
      return true;
    }
    case at::kFloat:
      std::fill_n(static_cast<float*>(host_ptr), numel, value.to<float>());
      return true;
    case at::kDouble:
      std::fill_n(static_cast<double*>(host_ptr), numel, value.to<double>());
      return true;
    case at::kHalf:
      std::fill_n(static_cast<at::Half*>(host_ptr), numel, value.to<at::Half>());
      return true;
    case at::kBFloat16:
      std::fill_n(static_cast<at::BFloat16*>(host_ptr), numel, value.to<at::BFloat16>());
      return true;
    default:
      return false; // unsupported dtype — caller falls back to cpu_fallback
  }
}

// Mirrors fill_host_typed's case set, so fill_ picks its path before building any host
// buffer. Other dtypes (e.g. ComplexHalf, ComplexFloat) go through fill_scalar_via_cpu.
bool fill_host_typed_supports(c10::ScalarType st) {
  switch (st) {
    case at::kLong:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
    case at::kFloat:
    case at::kDouble:
    case at::kHalf:
    case at::kBFloat16:
      return true;
    default:
      return false;
  }
}

// fill_.Scalar through a CPU tensor filled on the host and copied into `self`, for what
// the device paths below decline (complex dtypes, views whose runs could alias, too many runs).
at::Tensor& fill_scalar_via_cpu(at::Tensor& self, const at::Scalar& value) {
  auto self_cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
  self_cpu.fill_(value);
  self.copy_(self_cpu);
  return self;
}

constexpr int64_t kMaxFillRuns = 4096;
constexpr size_t kMaxPatternBytes = size_t{16} << 20; // one pattern buffer per fill

// Zeros are filled on the device; any other value is laid out once in a host pattern of at
// most kMaxPatternBytes, written over the dense view's bytes chunk by chunk.
void fill_dense(at::Tensor& self, const at::Scalar& value) {
  const auto st = self.scalar_type();
  alignas(sizeof(int64_t)) std::array<uint8_t, sizeof(int64_t)> element{};
  fill_host_typed(element.data(), 1, st, value);
  const size_t element_size = self.element_size();
  const size_t nbytes = self.nbytes();
  if (std::all_of(element.begin(), element.begin() + element_size, [](uint8_t b) { return b == 0; })) {
    c10::rbln::fill_zeros(self.data_ptr(), nbytes);
    return;
  }
  const size_t pattern_bytes = std::min(nbytes, kMaxPatternBytes);
  std::vector<uint8_t> pattern(pattern_bytes);
  fill_host_typed(pattern.data(), static_cast<int64_t>(pattern_bytes / element_size), st, value);
  auto* dst = static_cast<uint8_t*>(self.data_ptr());
  for (size_t done = 0; done < nbytes; done += pattern_bytes) {
    c10::rbln::memcpy_h2v(dst + done, pattern.data(), std::min(pattern_bytes, nbytes - done));
  }
}

/**
 * Region fill of a strided view through the host->device batch copy. The contiguous suffix
 * of the view's dims is one run; the dims above it are the outer iteration, enqueued with a
 * source stride of 0 so one pattern buffer feeds every run. Declined: a view whose runs
 * could alias (bulk destinations must be disjoint), more runs than kMaxFillRuns, or a run
 * larger than the pattern buffer cap.
 */
bool fill_region_device(at::Tensor& self, const at::Scalar& value) {
  const auto sizes = self.sizes();
  const auto strides = self.strides();
  if (at::has_internal_overlap(self) != at::MemOverlap::No && view_may_self_overlap(sizes, strides)) {
    return false;
  }
  const int64_t rank = self.dim();
  const int64_t elm = self.element_size();
  const int64_t inner_start = contig_suffix_start(sizes, strides);
  int64_t inner_elems = 1;
  for (int64_t i = inner_start; i < rank; ++i) {
    inner_elems *= sizes[i];
  }
  int64_t outer_count = 1;
  for (int64_t i = 0; i < inner_start; ++i) {
    outer_count *= sizes[i];
  }
  if (outer_count > kMaxFillRuns) {
    return false;
  }
  const size_t run_bytes = static_cast<size_t>(inner_elems) * static_cast<size_t>(elm);
  if (run_bytes == 0 || run_bytes > kMaxPatternBytes) {
    return false;
  }
  std::vector<uint8_t> pattern(run_bytes);
  if (!fill_host_typed(pattern.data(), inner_elems, self.scalar_type(), value)) {
    return false;
  }
  c10::SmallVector<int64_t, 8> outer_sizes(sizes.begin(), sizes.begin() + inner_start);
  c10::SmallVector<int64_t, 8> src_byte_strides(static_cast<size_t>(inner_start), 0);
  c10::SmallVector<int64_t, 8> dst_byte_strides;
  for (int64_t i = 0; i < inner_start; ++i) {
    dst_byte_strides.push_back(strides[i] * elm);
  }
  c10::rbln::H2VBatch batch;
  batch.enqueue_strided(self.data_ptr(), pattern.data(), run_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
  batch.submit();
  RBLN_LOG_DEBUG("fill_: region fill through h2v, runs={} run_bytes={}", outer_count, run_bytes);
  return true;
}
} // namespace

at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value) {
  RBLN_SCOPE_GUARD();
  if (self.numel() == 0) {
    return self;
  }
  // A tensor of a type is written the value encoded on the host (see c10/rbln/RBLNTyped.h).
  if (c10::rbln::caching::any_typed() && !c10::rbln::caching::locating_as_typed()) {
    auto found = c10::rbln::caching::try_locate_typed(self.const_data_ptr());
    if (found && found->type) {
      return fill_scalar_via_cpu(self, value);
    }
  }
  // Broadcast/expand view: a size>1 dim with stride 0 aliases many logical elements onto a
  // single storage element (internal overlap). A scalar fill is still well-defined — every
  // aliased element takes the same value, matching CPU's overlap-tolerant fill_ — but the
  // copy_-based fallback below would throw ("more than one element ... refers to a single
  // memory location"). Collapse each stride-0 dim to size 1 so each distinct storage element
  // is written exactly once, then fill that non-overlapping view.
  bool has_broadcast_overlap = false;
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (self.size(d) > 1 && self.stride(d) == 0) {
      has_broadcast_overlap = true;
      break;
    }
  }
  if (has_broadcast_overlap) {
    at::Tensor collapsed = self;
    for (int64_t d = 0; d < self.dim(); ++d) {
      if (self.size(d) > 1 && self.stride(d) == 0) {
        collapsed = collapsed.narrow(d, 0, 1);
      }
    }
    fill_scalar_rbln_(collapsed, value);
    return self;
  }

  if (!fill_host_typed_supports(self.scalar_type())) {
    return fill_scalar_via_cpu(self, value);
  }
  if (self.is_non_overlapping_and_dense()) {
    fill_dense(self, value);
    return self;
  }
  if (fill_region_device(self, value)) {
    return self;
  }
  return fill_scalar_via_cpu(self, value);
}

namespace {
template <typename scalar_t>
void arange_fill_host(scalar_t* host_ptr, int64_t n, scalar_t start, scalar_t step) {
  for (int64_t i = 0; i < n; ++i) {
    host_ptr[i] = static_cast<scalar_t>(start + static_cast<scalar_t>(i) * step);
  }
}

// out[i] = start + i*step for the int family and float/double; false for any other dtype.
bool arange_fill_host_typed(
    void* host_ptr,
    int64_t n,
    c10::ScalarType st,
    const at::Scalar& start,
    const at::Scalar& step) {
  switch (st) {
    case at::kLong:
      arange_fill_host<int64_t>(static_cast<int64_t*>(host_ptr), n, start.to<int64_t>(), step.to<int64_t>());
      return true;
    case at::kInt:
      arange_fill_host<int32_t>(static_cast<int32_t*>(host_ptr), n, start.to<int32_t>(), step.to<int32_t>());
      return true;
    case at::kShort:
      arange_fill_host<int16_t>(static_cast<int16_t*>(host_ptr), n, start.to<int16_t>(), step.to<int16_t>());
      return true;
    case at::kChar:
      arange_fill_host<int8_t>(static_cast<int8_t*>(host_ptr), n, start.to<int8_t>(), step.to<int8_t>());
      return true;
    case at::kByte:
      arange_fill_host<uint8_t>(static_cast<uint8_t*>(host_ptr), n, start.to<uint8_t>(), step.to<uint8_t>());
      return true;
    case at::kFloat:
      arange_fill_host<float>(static_cast<float*>(host_ptr), n, start.to<float>(), step.to<float>());
      return true;
    case at::kDouble:
      arange_fill_host<double>(static_cast<double*>(host_ptr), n, start.to<double>(), step.to<double>());
      return true;
    default:
      return false;
  }
}
} // namespace

/**
 * Native impl of aten::arange.start_out for RBLN.
 *   schema: arange.start_out(Scalar start, Scalar end, Scalar step,
 *                            *, Tensor(a!) out) -> Tensor(a!)
 * Upstream sizes and validates `out` in a structured meta function that a
 * PrivateUse1 impl bypasses, so this kernel does both itself with the helper
 * the CPU kernel uses. The values are built on the host and written to `out`
 * with one host-to-device copy.
 */
at::Tensor& arange_start_out_rbln(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  RBLN_SCOPE_GUARD();
  int64_t n = 0;
  AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "arange_start_out_rbln", [&] {
    n = at::native::compute_arange_size<scalar_t>(start, end, step);
  });
  if (out.numel() != n) {
    out.resize_({n});
  }
  if (n == 0) {
    return out;
  }
  auto out_cpu = at::empty({n}, out.options().device(at::kCPU));
  if (!arange_fill_host_typed(out_cpu.data_ptr(), n, out.scalar_type(), start, step)) {
    at::arange_out(out_cpu, start, end, step);
  }
  if (out.is_contiguous()) {
    c10::rbln::memcpy_h2v(out.data_ptr(), out_cpu.data_ptr(), out.nbytes());
  } else {
    out.copy_(out_cpu);
  }
  return out;
}

at::Scalar _local_scalar_dense_rbln(const at::Tensor& self) {
  RBLN_SCOPE_GUARD();
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense_rbln: expected 1-element tensor, got numel=", self.numel());

  // Large enough for the widest element, complex<double>.
  alignas(16) std::array<uint8_t, 16> host{};
  c10::rbln::memcpy_v2h(host.data(), self.data_ptr(), self.element_size());

  at::Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::kComplexHalf, at::kHalf, at::kBool, at::kBFloat16, self.scalar_type(), "_local_scalar_dense_rbln", [&] {
        scalar_t value;
        std::memcpy(&value, host.data(), sizeof(scalar_t));
        r = at::Scalar(value);
      });
  return r;
}

} // namespace at::native::rbln
