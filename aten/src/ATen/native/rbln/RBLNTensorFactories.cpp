#include <ATen/MemoryOverlap.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHostBatch.h>
#include <c10/rbln/RBLNLogging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  // Allocate fresh privateuse1 storage and mark its v-memory as zero-init.
  // `mark_zeros` flips the EMPTY_INIT_WITH_ZERO flag on the v-memory entry —
  // no host allocation, no D→H copy, no actual write. Zeros materialise lazily
  // on the first NPU read (or are skipped entirely when the first access is a
  // write, e.g. KV-cache output). This mirrors what `aten::zero_` already
  // does on RBLN via `custom_zero__rbln`.
  auto rbln_out = empty_rbln(
      sizes,
      dtype_opt,
      layout_opt,
      device_opt,
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
  // `mark_zeros` takes only a vaddr (no offset/size) and flips
  // EMPTY_INIT_WITH_ZERO on the whole enclosing allocation, so it is correct
  // only when `self` spans that allocation. A partial/offset view (e.g.
  // base[2:4]) would otherwise zero the entire tensor; route those through
  // fill_(0), which writes just the view's byte range.
  const bool covers_whole_allocation = self.is_contiguous() && self.storage_offset() == 0 &&
      static_cast<size_t>(self.numel()) * self.element_size() == self.storage().nbytes();
  if (!covers_whole_allocation) {
    return fill_scalar_rbln_(self, 0);
  }
  c10::rbln::mark_zeros(self.data_ptr());
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
} // namespace

// Native impl of aten::fill_.Scalar for RBLN. Bypasses cpu_fallback_rbln's
// redispatchBoxed(CPU) overhead: borrows a host pointer into the rbln vmemory
// backing ``self``, writes the value with a typed std::fill_n (compiler
// auto-vectorises), then commits via return_borrowed(updated=true). The
// device→host transfer is implicit in borrow_host_ptr (we use the read variant
// rather than acquire_host_ptr_for_overwrite because partial-view writes need
// the rest of the storage to remain intact). A contiguous offset view (e.g.
// base[2:4], where zero_ routes here) borrows correctly: data_ptr() is an
// interior vaddr and the runtime returns host_ptr = allocation base + offset
// with the range bounds-checked, so we write only the view's bytes.
//
// Only contiguous tensors take this fast path; a non-contiguous self (or an
// unsupported dtype) routes through the standard CPU fallback instead (see the
// is_contiguous() guard below). Stride iteration here is a possible future
// extension.
// Cheap dtype check that mirrors ``fill_host_typed``'s case set. Used so we
// can decide whether the borrow-and-fill fast path is viable *before*
// acquiring the host-pointer (which has its own cost). When this returns
// false (e.g. ComplexHalf, ComplexFloat) we route through the CPU fallback
// path below so the OpInfo tests that exercise those dtypes don't trip a
// hard assert in the native impl.
namespace {
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

// CPU fallback for fill_.Scalar: copy device tensor to CPU, fill, copy back.
// Used for shapes/dtypes the native borrow path can't handle (non-contig
// self, complex / quantized dtypes, etc.). Keeps the native impl
// transparent to callers: they observe the same in-place mutation.
at::Tensor& fill_scalar_via_cpu(at::Tensor& self, const at::Scalar& value) {
  auto self_cpu = self.cpu();
  self_cpu.fill_(value);
  self.copy_(self_cpu);
  return self;
}

// Region fill through the runtime's host->virtual copy.
//
// The borrow path below materialises the whole storage on the host: a device-latest
// tensor is read back in full, the view is filled, and the storage is left host-latest,
// so the next device consumer uploads all of it again. Writing the view's runs through
// the h2v copy instead lets the runtime route by the entry's own sync state, as it does
// for every cpu->rbln copy_: a device-resident entry is written in place and stays
// device-latest; a host-latest or empty entry is written in its user view with no
// transfer.
//
// The contiguous suffix of the view's dims is one run; the dims above it are the outer
// iteration, enqueued with a source stride of 0 so one pattern buffer feeds every run,
// and H2VBatch splits the submit to the runtime's bulk caps. Declined, and left to the
// paths below: a view whose runs could alias (bulk destinations must be disjoint), more
// runs than kMaxFillRuns, or a run larger than the pattern buffer cap.
namespace {
constexpr int64_t kMaxFillRuns = 4096;
constexpr size_t kMaxPatternBytes = size_t{16} << 20; // one pattern buffer per fill

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

// RAII for a host borrow: if the scope exits via a throw (e.g. Scalar::to<T>()'s checked
// cast overflowing), return the borrow (updated=false) so it isn't leaked. The happy path
// returns explicitly then disarms.
struct BorrowGuard {
  uint64_t id;
  bool armed = true;
  ~BorrowGuard() {
    if (armed) {
      try {
        c10::rbln::return_borrowed(id, /*updated=*/false);
      } catch (...) {
      }
    }
  }
};
} // namespace

at::Tensor& fill_scalar_rbln_(at::Tensor& self, const at::Scalar& value) {
  RBLN_SCOPE_GUARD();
  if (self.numel() == 0) {
    return self;
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

  if (fill_host_typed_supports(self.scalar_type()) && fill_region_device(self, value)) {
    return self;
  }

  // Fast path is restricted to contiguous tensors with a dtype the typed
  // ``std::fill_n`` cascade in ``fill_host_typed`` covers. Anything else
  // (non-contig views — strided OpInfo samples — and complex / quantized
  // dtypes) routes through the CPU fallback so we don't hard-assert.
  if (!self.is_contiguous() || !fill_host_typed_supports(self.scalar_type())) {
    return fill_scalar_via_cpu(self, value);
  }

  const auto nbytes = static_cast<size_t>(self.numel()) * self.element_size();
  const auto borrow = c10::rbln::borrow_host_ptr(self.data_ptr(), nbytes);

  // fill_host_typed's checked cast can throw on overflow (e.g. fill_(128) into int8);
  // guard the borrow so the throw can't skip return_borrowed and leak it.
  BorrowGuard guard{borrow.borrow_id};

  void* host_ptr = reinterpret_cast<void*>(borrow.host_ptr);
  const bool handled = fill_host_typed(host_ptr, self.numel(), self.scalar_type(), value);
  // Return while still armed, then disarm: if this explicit return throws, the
  // guard's dtor retries (updated=false) rather than leaking the borrow.
  c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/handled);
  guard.armed = false;
  // ``fill_host_typed_supports`` already vetted the dtype above, so the
  // call above should always succeed. Keep a defensive fallback for the
  // (hypothetical) case where the two predicates diverge.
  if (!handled) {
    return fill_scalar_via_cpu(self, value);
  }
  return self;
}

namespace {
template <typename scalar_t>
void arange_fill_host(scalar_t* host_ptr, int64_t n, scalar_t start, scalar_t step) {
  for (int64_t i = 0; i < n; ++i) {
    host_ptr[i] = static_cast<scalar_t>(start + static_cast<scalar_t>(i) * step);
  }
}
} // namespace

// Native impl of aten::arange.start_out for RBLN.
//   schema: arange.start_out(Scalar start, Scalar end, Scalar step,
//                            *, Tensor(a!) out) -> Tensor(a!)
// Write-only on `out`, so we use acquire_host_ptr_for_overwrite (D2H sync
// skipped) and commit via return_borrowed(updated=true). `out` is assumed
// already sized correctly by the caller (vllm-rbln preallocates the tensor;
// torch.arange's meta function sizes it before dispatch).
at::Tensor& arange_start_out_rbln(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  RBLN_SCOPE_GUARD();
  // Compute expected length = ceil((end - start) / step). PyTorch's native
  // CPU/CUDA path runs through TensorIterator + a structured meta function
  // that resizes `out`. On PrivateUse1 we own the impl, so we have to
  // mirror that behaviour explicitly. Use double for the range arithmetic
  // and clamp to >=0.
  const double s_d = start.to<double>();
  const double e_d = end.to<double>();
  const double st_d = step.to<double>();
  TORCH_CHECK(st_d != 0.0, "arange.start_out: step must be non-zero");
  int64_t n = 0;
  if ((st_d > 0.0 && e_d > s_d) || (st_d < 0.0 && e_d < s_d)) {
    n = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
  }
  if (out.numel() != n) {
    out.resize_({n});
  }
  if (n == 0) {
    return out;
  }
  // Fast path needs both contiguous out AND a dtype the typed arange-fill
  // cascade covers (int family + float/double). Half / BFloat16 / complex
  // dtypes that OpInfo tests for arange route through the CPU fallback
  // below so we don't hard-assert and crash the suite.
  auto cpu_fallback_for_arange = [&]() -> at::Tensor& {
    auto out_cpu = at::empty({n}, out.options().device(c10::Device(c10::DeviceType::CPU)));
    at::arange_out(out_cpu, start, end, step);
    out.copy_(out_cpu);
    return out;
  };
  if (!out.is_contiguous()) {
    return cpu_fallback_for_arange();
  }
  const auto st = out.scalar_type();
  const bool dtype_supported = st == at::kLong || st == at::kInt || st == at::kShort || st == at::kChar ||
      st == at::kByte || st == at::kFloat || st == at::kDouble;
  if (!dtype_supported) {
    return cpu_fallback_for_arange();
  }

  const auto nbytes = static_cast<size_t>(out.numel()) * out.element_size();
  const auto borrow = c10::rbln::acquire_host_ptr_for_overwrite(out.data_ptr(), nbytes);
  void* host_ptr = reinterpret_cast<void*>(borrow.host_ptr);
  // start.to<T>()/step.to<T>() below are checked casts that can throw on overflow; guard the
  // borrow so a throw can't skip return_borrowed and leak it.
  BorrowGuard guard{borrow.borrow_id};

  switch (st) {
    case at::kLong:
      arange_fill_host<int64_t>(static_cast<int64_t*>(host_ptr), n, start.to<int64_t>(), step.to<int64_t>());
      break;
    case at::kInt:
      arange_fill_host<int32_t>(static_cast<int32_t*>(host_ptr), n, start.to<int32_t>(), step.to<int32_t>());
      break;
    case at::kShort:
      arange_fill_host<int16_t>(static_cast<int16_t*>(host_ptr), n, start.to<int16_t>(), step.to<int16_t>());
      break;
    case at::kChar:
      arange_fill_host<int8_t>(static_cast<int8_t*>(host_ptr), n, start.to<int8_t>(), step.to<int8_t>());
      break;
    case at::kByte:
      arange_fill_host<uint8_t>(static_cast<uint8_t*>(host_ptr), n, start.to<uint8_t>(), step.to<uint8_t>());
      break;
    case at::kFloat:
      arange_fill_host<float>(static_cast<float*>(host_ptr), n, start.to<float>(), step.to<float>());
      break;
    case at::kDouble:
      arange_fill_host<double>(static_cast<double*>(host_ptr), n, start.to<double>(), step.to<double>());
      break;
    default:
      // dtype_supported gate above rules this branch out.
      c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/false);
      guard.armed = false;
      return cpu_fallback_for_arange();
  }
  c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/true);
  guard.armed = false;
  return out;
}

at::Scalar _local_scalar_dense_rbln(const at::Tensor& self) {
  RBLN_SCOPE_GUARD();
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense_rbln: expected 1-element tensor, got numel=", self.numel());

  const size_t nbytes = static_cast<size_t>(self.element_size());
  const auto borrow = c10::rbln::borrow_host_ptr(self.data_ptr(), nbytes);
  void* p = reinterpret_cast<void*>(borrow.host_ptr);

  at::Scalar r;
  bool handled = true;
  switch (self.scalar_type()) {
    case at::kLong:
      r = at::Scalar(*static_cast<int64_t*>(p));
      break;
    case at::kInt:
      r = at::Scalar(*static_cast<int32_t*>(p));
      break;
    case at::kShort:
      r = at::Scalar(*static_cast<int16_t*>(p));
      break;
    case at::kChar:
      r = at::Scalar(*static_cast<int8_t*>(p));
      break;
    case at::kByte:
      r = at::Scalar(*static_cast<uint8_t*>(p));
      break;
    case at::kBool:
      r = at::Scalar(*static_cast<bool*>(p));
      break;
    case at::kFloat:
      r = at::Scalar(*static_cast<float*>(p));
      break;
    case at::kDouble:
      r = at::Scalar(*static_cast<double*>(p));
      break;
    case at::kHalf:
      r = at::Scalar(*static_cast<at::Half*>(p));
      break;
    case at::kBFloat16:
      r = at::Scalar(*static_cast<at::BFloat16*>(p));
      break;
    default:
      handled = false;
  }
  c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/false);
  TORCH_INTERNAL_ASSERT(handled, "_local_scalar_dense_rbln: unsupported dtype ", c10::toString(self.scalar_type()));
  return r;
}

} // namespace at::native::rbln
