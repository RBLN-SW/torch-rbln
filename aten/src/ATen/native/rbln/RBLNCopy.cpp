// TODO: The previous copy optimizations based on physical shape and physical dtype were removed during
// v-memory integration. Revisit these optimizations in a future pass.
#include <algorithm>

#include <ATen/MemoryOverlap.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNHostCopy.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNPinnedAllocator.h>
#include <c10/rbln/RBLNProfiler.h>
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace at::native::rbln {

namespace {

// Recursion guard: a non-direct rbln->rbln copy_ services itself by a v2h
// (get_cpu_copy_of_rbln_tensor) followed by a cpu->rbln copy. We count that
// round-trip ONCE (kRbln2RblnIndirect); this flag suppresses the inner cpu->rbln
// staging / noncontig counters so a single copy_ is not double-counted.
thread_local int g_indirect_d2d_depth = 0;
struct IndirectD2DGuard {
  IndirectD2DGuard() {
    ++g_indirect_d2d_depth;
  }
  ~IndirectD2DGuard() {
    --g_indirect_d2d_depth;
  }
};

bool is_direct_copy(const at::Tensor& src, const at::Tensor& dst) {
  const bool same_sizes = (src.sizes() == dst.sizes());
  const bool same_dtype = (src.scalar_type() == dst.scalar_type());
  const bool both_contiguous = (src.is_contiguous() && dst.is_contiguous());
  const bool direct_copy = (same_sizes && same_dtype && both_contiguous);
  return direct_copy;
}

void tensor_copy_from_cpu_to_rbln(const at::Tensor& cpu_src, const at::Tensor& rbln_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(cpu_src.data_ptr()), fmt::ptr(rbln_dst.data_ptr()));
  RBLN_CHECK(cpu_src.device().is_cpu() && rbln_dst.device().is_privateuseone());

  const auto direct_copy = is_direct_copy(cpu_src, rbln_dst);
  if (direct_copy) {
    RBLN_LOG_DEBUG("Directly copying CPU src to RBLN dst");
    auto* dst_data = rbln_dst.data_ptr();
    const auto* src_data = cpu_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(cpu_src.sizes(), cpu_src.strides(), cpu_src.element_size());
    c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
  } else {
    if (rbln_dst.is_contiguous()) {
      const auto dst_sizes = rbln_dst.sizes();
      const auto dst_dtype = rbln_dst.scalar_type();
      const bool same_sizes = (cpu_src.sizes() == dst_sizes);
      const bool same_dtype = (cpu_src.scalar_type() == dst_dtype);

      RBLN_LOG_DEBUG("Preparing contiguous CPU src matching dst sizes/dtype");
      auto prepared_cpu_src = cpu_src;
      if (!same_sizes || !same_dtype || !cpu_src.is_contiguous()) {
        // PROFILER (cold branch): hidden staging alloc + CPU copy before h2v,
        // forced by a broadcast / dtype / contiguity mismatch on the cpu src.
        // Suppressed when nested inside an rbln->rbln indirect copy (counted once there).
        if (g_indirect_d2d_depth == 0) {
          c10::rbln::prof::record_bounce(
              c10::rbln::prof::BounceSite::kCpu2RblnStaging,
              static_cast<uint64_t>(rbln_dst.numel()) * rbln_dst.element_size());
        }
        prepared_cpu_src =
            at::empty(dst_sizes, dst_dtype, std::nullopt, c10::Device(c10::kCPU), false, c10::MemoryFormat::Contiguous);

        // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
        prepared_cpu_src.copy_(cpu_src);
      }
      RBLN_CHECK(prepared_cpu_src.sizes() == dst_sizes);
      RBLN_CHECK(prepared_cpu_src.scalar_type() == dst_dtype);
      RBLN_CHECK(prepared_cpu_src.is_contiguous());

      RBLN_LOG_DEBUG("Copying prepared CPU src to RBLN dst");
      auto* dst_data = rbln_dst.data_ptr();
      const auto* src_data = prepared_cpu_src.data_ptr();
      const auto nbytes = at::detail::computeStorageNbytes(
          prepared_cpu_src.sizes(), prepared_cpu_src.strides(), prepared_cpu_src.element_size());
      c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
    } else {
      // PROFILER (cold branch): a non-contiguous rbln dst is pulled to host
      // (hidden v2h), written on CPU, then copied back (h2v). Suppressed when
      // nested inside an rbln->rbln indirect copy (counted once there).
      if (g_indirect_d2d_depth == 0) {
        c10::rbln::prof::record_bounce(
            c10::rbln::prof::BounceSite::kCpu2RblnNoncontigDst,
            static_cast<uint64_t>(rbln_dst.numel()) * rbln_dst.element_size());
      }
      RBLN_LOG_DEBUG("Creating CPU copy of non-contiguous RBLN dst");
      auto cpu_dst = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_dst);

      RBLN_LOG_DEBUG("Copying CPU src to CPU copy");
      // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
      cpu_dst.copy_(cpu_src);

      RBLN_LOG_DEBUG("Copying CPU copy back to RBLN dst");
      auto* dst_data = rbln_dst.data_ptr();
      const auto* src_data = cpu_dst.data_ptr();
      const auto nbytes = at::detail::computeStorageNbytes(cpu_dst.sizes(), cpu_dst.strides(), cpu_dst.element_size());
      c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
    }
  }
}

void tensor_copy_from_rbln_to_cpu(const at::Tensor& rbln_src, const at::Tensor& cpu_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(rbln_src.data_ptr()), fmt::ptr(cpu_dst.data_ptr()));
  RBLN_CHECK(rbln_src.device().is_privateuseone() && cpu_dst.device().is_cpu());

  const auto direct_copy = is_direct_copy(rbln_src, cpu_dst);
  if (direct_copy) {
    RBLN_LOG_DEBUG("Directly copying RBLN src to CPU dst");

    auto* dst_data = cpu_dst.data_ptr();
    const auto* src_data = rbln_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(rbln_src.sizes(), rbln_src.strides(), rbln_src.element_size());
    c10::rbln::memcpy_v2h(dst_data, src_data, nbytes);
  } else {
    RBLN_LOG_DEBUG("Creating CPU copy of RBLN src");
    const auto cpu_src = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_src);

    RBLN_LOG_DEBUG("Copying CPU copy to CPU dst");
    // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
    cpu_dst.copy_(cpu_src);
  }
}

void tensor_copy_from_rbln_to_rbln(const at::Tensor& rbln_src, const at::Tensor& rbln_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(rbln_src.data_ptr()), fmt::ptr(rbln_dst.data_ptr()));
  RBLN_CHECK(rbln_src.device().is_privateuseone() && rbln_dst.device().is_privateuseone());

  if (is_direct_copy(rbln_src, rbln_dst)) {
    RBLN_LOG_DEBUG("Directly copying RBLN src to RBLN dst");

    auto* dst_data = rbln_dst.data_ptr();
    const auto* src_data = rbln_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(rbln_src.sizes(), rbln_src.strides(), rbln_src.element_size());
    c10::rbln::memcpy_v2v(dst_data, src_data, nbytes);
    return;
  }

  // Strided copy: route to the on-device v2v engine while the outer iteration
  // count stays within the runtime per-dst v2v cap (::rbln::kMaxV2VMultiCopies,
  // the single source shared with the runtime); above it the engine fans out to
  // a host fallback anyway, so bounce via host here.
  if (rbln_src.sizes() == rbln_dst.sizes() && rbln_src.scalar_type() == rbln_dst.scalar_type() &&
      rbln_src.device() == rbln_dst.device()) {
    const auto inner_start = common_inner_start(rbln_src.sizes(), rbln_src.strides(), rbln_dst.strides());
    int64_t outer_count = 1;
    for (int64_t i = 0; i < inner_start; ++i) {
      outer_count *= rbln_src.size(i);
    }
    if (outer_count <= static_cast<int64_t>(::rbln::kMaxV2VMultiCopies)) {
      strided_v2v_copy(rbln_dst, rbln_src);
      return;
    }
  }

  // PROFILER (cold branch): reaching here is a non-direct device->device copy_
  // that round-trips host — v2h (get_cpu_copy_of_rbln_tensor) then h2v (the
  // headline "hidden host bounce", e.g. #94 KV partial-block copy). The direct
  // and on-device strided-v2v paths above both return first, so this fall-through
  // always means a real host bounce. Count incident + bytes; IndirectD2DGuard
  // suppresses the inner cpu->rbln staging counters so it is not double-counted.
  c10::rbln::prof::record_bounce(
      c10::rbln::prof::BounceSite::kRbln2RblnIndirect,
      static_cast<uint64_t>(rbln_src.numel()) * rbln_src.element_size());
  RBLN_LOG_DEBUG("Creating CPU copy of RBLN src");
  const auto cpu_src = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_src);

  RBLN_LOG_DEBUG("Copying CPU copy to RBLN dst");
  const IndirectD2DGuard guard;
  tensor_copy_from_cpu_to_rbln(cpu_src, rbln_dst);
}

void copy_impl_rbln(const at::Tensor& src, const at::Tensor& dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_metadata={}", at::native::rbln::get_tensor_metadata_string(src));
  RBLN_LOG_DEBUG("dst_metadata={}", at::native::rbln::get_tensor_metadata_string(dst));

  RBLN_LOG_DEBUG("Validating tensors with TensorIterator");
  const auto iter = at::TensorIteratorConfig()
                        .add_output(dst)
                        .add_const_input(src)
                        .resize_outputs(false)
                        .check_all_same_dtype(false)
                        .check_all_same_device(false)
                        .build();

  if (iter.numel() == 0) {
    RBLN_LOG_DEBUG("No elements to copy");
    return;
  }
  const auto src_numel = src.numel();
  const auto dst_numel = dst.numel();
  RBLN_CHECK(src_numel > 0, "Source tensor must have positive numel, got {}", src_numel);
  RBLN_CHECK(dst_numel > 0, "Destination tensor must have positive numel, got {}", dst_numel);

  const auto src_device = src.device();
  const auto dst_device = dst.device();
  if (src_device.is_cpu() && dst_device.is_privateuseone()) {
    tensor_copy_from_cpu_to_rbln(src, dst);
  } else if (src_device.is_privateuseone() && dst_device.is_cpu()) {
    tensor_copy_from_rbln_to_cpu(src, dst);
  } else if (src_device.is_privateuseone() && dst_device.is_privateuseone()) {
    tensor_copy_from_rbln_to_rbln(src, dst);
  } else {
    RBLN_CHECK(
        false, "Tensor copy from {} device to {} device is not supported", c10::str(src_device), c10::str(dst_device));
  }
}

} // namespace

void copy_impl_rbln_async(const at::Tensor& src, const at::Tensor& dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("Attempting async copy");

  const auto src_device = src.device();
  const auto dst_device = dst.device();
  const auto direct_copy = is_direct_copy(src, dst);

  // Async only supported for direct copies (same size, dtype, both contiguous).
  // Non-direct copies require CPU-side staging which needs synchronous data access.
  if (!direct_copy) {
    // The sync fallback drains in-flight async transfers in the runtime (rbln_vmem_api).
    RBLN_LOG_DEBUG("Non-direct copy, falling back to sync");
    copy_impl_rbln(src, dst);
    return;
  }

  const auto nbytes = at::detail::computeStorageNbytes(src.sizes(), src.strides(), src.element_size());
  if (nbytes == 0) {
    RBLN_LOG_DEBUG("No bytes to copy");
    return;
  }

  // CUDA semantics: async only when the host side is pinned. Host reads never
  // pass through RBLN, so nothing can drain a pending transfer first — sync
  // for pageable makes that safe; pinned opts in (caller synchronizes).
  const at::Tensor& host = src_device.is_cpu() ? src : dst;
  if (host.device().is_cpu() && !c10::rbln::is_pinned_ptr(host.data_ptr())) {
    RBLN_LOG_DEBUG("Pageable host tensor, downgrading non_blocking copy to sync");
    copy_impl_rbln(src, dst);
    return;
  }

  if (src_device.is_cpu() && dst_device.is_privateuseone()) {
    RBLN_LOG_DEBUG("Async CPU -> RBLN");
    c10::rbln::memcpy_h2v_async(dst.data_ptr(), src.data_ptr(), nbytes);
  } else if (src_device.is_privateuseone() && dst_device.is_cpu()) {
    RBLN_LOG_DEBUG("Async RBLN -> CPU");
    c10::rbln::memcpy_v2h_async(dst.data_ptr(), src.data_ptr(), nbytes);
  } else if (src_device.is_privateuseone() && dst_device.is_privateuseone()) {
    RBLN_LOG_DEBUG("Async RBLN -> RBLN");
    c10::rbln::memcpy_v2v_async(dst.data_ptr(), src.data_ptr(), nbytes);
  } else {
    RBLN_CHECK(
        false, "Tensor copy from {} device to {} device is not supported", c10::str(src_device), c10::str(dst_device));
  }
}

at::Tensor _copy_from_rbln(const at::Tensor& src, const at::Tensor& dst, bool non_blocking) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(src.data_ptr()), fmt::ptr(dst.data_ptr()));

  if (non_blocking) {
    copy_impl_rbln_async(src, dst);
  } else {
    // Sync copy; the runtime (rbln_vmem_api) drains in-flight async transfers first.
    copy_impl_rbln(src, dst);
  }

  return dst;
}

at::Tensor _copy_from_and_resize_rbln(const at::Tensor& src, const at::Tensor& dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(src.data_ptr()), fmt::ptr(dst.data_ptr()));

  const auto src_sizes = src.sizes();
  const auto dst_sizes = dst.sizes();
  RBLN_LOG_DEBUG("src_sizes={}, dst_sizes={}", c10::str(src_sizes), c10::str(dst_sizes));
  if (dst_sizes != src_sizes) {
    RBLN_LOG_DEBUG("Resizing dst to match src");
    dst.resize_(src_sizes);
  }

  copy_impl_rbln(src, dst);

  return dst;
}

at::Tensor clone_rbln(const at::Tensor& self, std::optional<c10::MemoryFormat> memory_format) {
  RBLN_SCOPE_GUARD();
  // aten::clone's default is MemoryFormat::Preserve, which is a *directive*
  // (allocate the output with whatever layout best preserves the source's
  // observable strides), not a concrete storage layout you can hand to
  // ``at::empty``. Mirror PyTorch's reference resolution before allocating:
  //   * Preserve + dense contiguous source → Contiguous
  //   * Preserve + channels_last source → ChannelsLast (2D/3D)
  //   * Preserve + arbitrary strided view → empty_strided(self.sizes(),
  //     self.strides()) so the output observes identical strides
  // For anything else the user explicitly named (Contiguous, ChannelsLast,
  // ChannelsLast3d) we pass it through to ``at::empty`` as before.
  const auto mf_req = memory_format.value_or(c10::MemoryFormat::Preserve);
  const size_t nbytes = static_cast<size_t>(self.numel()) * self.element_size();

  // Direct-d2d eligibility: contiguous storage layout, zero storage offset,
  // and the user view spans the entire storage. When all three hold the
  // source elements are contiguous in storage, so a single
  // c10::rbln::memcpy_v2v moves them to a freshly allocated output without
  // going through aten::copy_'s dispatch + TensorIterator path.
  const bool direct_d2d_eligible = self.is_contiguous() && self.storage_offset() == 0 &&
      static_cast<int64_t>(self.storage().nbytes()) == self.numel() * self.element_size();

  at::Tensor out;
  if (mf_req == c10::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Strided dense source: replay the strides on a fresh buffer so the
      // clone observes identical layout.
      out = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::ChannelsLast));
    } else if (self.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    } else {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::Contiguous));
    }
  } else {
    out = at::empty(self.sizes(), self.options().memory_format(mf_req));
  }

  if (direct_d2d_eligible && nbytes > 0 && out.is_contiguous() && out.strides() == self.strides()) {
    // Bypass aten::copy_ dispatch — write directly into the fresh output
    // buffer. Both self and out live on the same RBLN device (Tensor::options
    // preserves device), so this is always a same-device v2v. Require
    // matching strides so the byte order is identical; if the requested
    // memory format reshuffles strides (e.g. Preserve on a strided view
    // gives empty_strided with non-contig strides), fall through to the
    // copy_ path which honours strides.
    c10::rbln::memcpy_v2v(out.data_ptr(), self.data_ptr(), nbytes);
  } else {
    // Non-contig / partial-storage view: keep the default composite path
    // so aten::copy_'s TensorIterator handles the strided gather.
    out.copy_(self);
  }
  return out;
}

namespace {

// Byte extent [lo, hi) that `t` actually addresses, from sizes/strides — so
// disjoint slices of one storage (e.g. kv_cache[0] vs kv_cache[1]) report
// non-overlapping ranges. Negative strides extend `lo`; +1 makes it half-open.
struct ByteRange {
  const char* lo;
  const char* hi;
};

ByteRange tensor_byte_range(const at::Tensor& t) {
  const char* base = static_cast<const char*>(t.data_ptr());
  const int64_t elsize = t.element_size();
  int64_t lo_elems = 0;
  int64_t hi_elems = 0;
  const auto sizes = t.sizes();
  const auto strides = t.strides();
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (sizes[d] <= 1) {
      continue;
    }
    const int64_t step = (sizes[d] - 1) * strides[d];
    if (step >= 0) {
      hi_elems += step;
    } else {
      lo_elems += step;
    }
  }
  return {base + lo_elems * elsize, base + (hi_elems + 1) * elsize};
}

bool ranges_overlap(const ByteRange& a, const ByteRange& b) {
  return a.lo < b.hi && b.lo < a.hi;
}

// The batched path submits eligible copies together (unordered) and after the
// inline fallback copies, so it matches list-order copy_ only when copies don't
// alias across pairs. Returns true on any cross-pair overlap — a destination
// hitting another pair's source (RAW/WAR) or destination (WAW). Within-pair
// (self[i]/src[i]) is left to copy_.
//
// Host tensors are tracked alongside device ones. They have to be: the h2v/v2h
// runtime entrypoints require destination ranges to be disjoint and do not
// validate it, so for a device->host batch this check is the ONLY thing standing
// between an aliased destination list and a silently wrong result. Ranges on
// different devices can never alias, and CPU is its own address space — hence
// keying on the full c10::Device rather than just the index.
//
// Quadratic in the pair count, which holds at the sizes this op sees: each
// comparison is a handful of integer tests, so even a thousand pairs stays in
// the low milliseconds next to transfers that move real payload. Worth a sorted
// sweep only if a caller turns up with a fan-out far beyond that.
bool foreach_copy_reorder_unsafe(at::TensorList self, at::TensorList src) {
  // One entry per tracked tensor, sorted so overlaps fall next to each other.
  // Comparing every pair with every other is quadratic, and this runs before a
  // single byte moves: at 2048 pairs it costs more than the transfer itself
  // (measured 6.5ms of bookkeeping for 2048 one-element pairs). Sorting answers
  // the same question in n log n.
  struct Entry {
    int64_t device_key;
    const char* lo;
    const char* hi;
    size_t idx;
  };
  const auto device_key = [](const c10::Device& d) {
    return (static_cast<int64_t>(d.type()) << 16) | static_cast<int64_t>(d.index());
  };
  const auto by_position = [](const Entry& a, const Entry& b) {
    return a.device_key != b.device_key ? a.device_key < b.device_key : a.lo < b.lo;
  };
  const auto trackable = [](const at::Tensor& t) {
    return (t.device().is_privateuseone() || t.device().is_cpu()) && t.numel() > 0;
  };

  const size_t n = self.size();
  std::vector<Entry> dsts;
  std::vector<Entry> srcs;
  dsts.reserve(n);
  srcs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (trackable(self[i])) {
      const auto r = tensor_byte_range(self[i]);
      dsts.push_back({device_key(self[i].device()), r.lo, r.hi, i});
    }
    if (trackable(src[i])) {
      const auto r = tensor_byte_range(src[i]);
      srcs.push_back({device_key(src[i].device()), r.lo, r.hi, i});
    }
  }
  std::sort(dsts.begin(), dsts.end(), by_position);
  std::sort(srcs.begin(), srcs.end(), by_position);

  // WAW: two destinations sharing bytes. Neighbours in sort order suffice --
  // carry the furthest end seen so far so a long range still catches the ones
  // nested inside it.
  const char* reach = nullptr; // furthest end seen in the current device group
  for (size_t k = 0; k < dsts.size(); ++k) {
    if (k == 0 || dsts[k].device_key != dsts[k - 1].device_key) {
      reach = dsts[k].hi;
      continue;
    }
    if (dsts[k].lo < reach) {
      return true;
    }
    reach = std::max(reach, dsts[k].hi);
  }

  // RAW / WAR: a source sharing bytes with another pair's destination. The
  // destinations are known disjoint by the loop above, so a source can only
  // touch several of them if at least one belongs to a different pair; if it
  // touches exactly one, it is safe only when that is its own pair's.
  for (const auto& s : srcs) {
    const Entry probe{s.device_key, s.hi, s.hi, 0};
    auto it = std::lower_bound(dsts.begin(), dsts.end(), probe, by_position);
    // Walk back over destinations that start before this source ends.
    size_t touching = 0;
    size_t only_idx = 0;
    while (it != dsts.begin()) {
      --it;
      if (it->device_key != s.device_key || it->hi <= s.lo) {
        break;
      }
      if (it->lo < s.hi) {
        only_idx = it->idx;
        if (++touching > 1) {
          return true;
        }
      }
    }
    if (touching == 1 && only_idx != s.idx) {
      return true;
    }
  }
  return false;
}

} // namespace

void _foreach_copy__rbln(at::TensorList self, at::TensorList src, bool non_blocking) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      self.size() == src.size(),
      "_foreach_copy_: self and src lists must have equal length ({} vs {})",
      self.size(),
      src.size());

  // Cross-pair aliasing would make the batch's reordering observable; fall back
  // to an ordered copy_ loop. The common disjoint scatter keeps the fast path.
  if (foreach_copy_reorder_unsafe(self, src)) {
    for (size_t i = 0; i < self.size(); ++i) {
      self[i].copy_(src[i], non_blocking);
    }
    return;
  }

  // Batch every conversion-free, same-shape pair so N copies flush through one
  // runtime submit instead of N. Three directions get their own batch:
  //
  //   rbln -> rbln   V2VBatch   the write-side mirror of cat's gather; a scatter
  //                             into N per-layer tensors collapses to 1 submit
  //   cpu  -> rbln   H2VBatch   weight load / load_state_dict fan-out
  //   rbln -> cpu    V2HBatch   the gather direction, which had no batched path
  //
  // Pairs needing broadcast or a dtype cast still fall back to a per-pair copy_:
  // the multi entrypoints move bytes and do not convert.
  //
  // Cross-device is not a disqualifier for the host directions: H2VBatch /
  // V2HBatch split per device instead of host-bouncing. Only rbln->rbln with
  // mismatched devices stays ineligible.
  c10::rbln::V2VBatch v2v_batch;
  c10::rbln::H2VBatch h2v_batch;
  c10::rbln::V2HBatch v2h_batch;
  std::vector<std::pair<at::Tensor, at::Tensor>> v2v_batched;
  std::vector<std::pair<at::Tensor, at::Tensor>> h2v_batched;
  std::vector<std::pair<at::Tensor, at::Tensor>> v2h_batched;
  v2v_batched.reserve(self.size());

  // Flush everything queued so far. Called before an inline copy_ so the
  // observable order matches the list: without it a pair deferred to submit
  // could be skipped entirely when a later inline copy_ throws, leaving its
  // destination untouched where the sequential path would have written it.
  const auto flush_pending = [&] {
    submit_or_fallback(v2v_batch, "_foreach_copy_", [&] {
      for (const auto& pair : v2v_batched) {
        pair.first.copy_(pair.second.cpu());
      }
    });
    submit_or_fallback(h2v_batch, "_foreach_copy_", [&] {
      for (const auto& pair : h2v_batched) {
        pair.first.copy_(pair.second);
      }
    });
    submit_or_fallback(v2h_batch, "_foreach_copy_", [&] {
      for (const auto& pair : v2h_batched) {
        pair.first.copy_(pair.second);
      }
    });
    v2v_batched.clear();
    h2v_batched.clear();
    v2h_batched.clear();
  };

  for (size_t i = 0; i < self.size(); ++i) {
    const at::Tensor& dst = self[i];
    const at::Tensor& s = src[i];
    const bool convertible = dst.scalar_type() == s.scalar_type() && dst.sizes() == s.sizes() && dst.numel() > 0;
    if (!convertible) {
      flush_pending();
      dst.copy_(s, non_blocking);
      continue;
    }
    // copy_ returns early when destination and source describe the same view,
    // before it looks at overlap at all, so an expand()ed identity pair is a
    // no-op there rather than an error. Match that: the batch must not reject
    // what copy_ accepts.
    if (is_same_view(dst, s)) {
      continue;
    }
    // A destination that maps several logical elements onto one address cannot
    // be written by a batch: entries are unordered, so which write survives is
    // arbitrary. Flush first so the pairs before this one still land.
    //
    //   Yes      an expand()ed view. copy_ refuses it, so raise its error.
    //   TooHard  has_internal_overlap gives up on gappy strides. Decide what
    //            can be decided cheaply: a destination that may alias itself
    //            goes down the per-pair path, which writes in a defined order,
    //            while one that merely has gaps stays batchable.
    const auto overlap = at::has_internal_overlap(dst);
    if (overlap == at::MemOverlap::Yes) {
      flush_pending();
      at::assert_no_internal_overlap(dst);
    }
    if (overlap == at::MemOverlap::TooHard && view_may_self_overlap(dst.sizes(), dst.strides())) {
      flush_pending();
      dst.copy_(s, non_blocking);
      continue;
    }
    const bool dst_dev = dst.device().is_privateuseone();
    const bool src_dev = s.device().is_privateuseone();
    const bool dst_cpu = dst.device().is_cpu();
    const bool src_cpu = s.device().is_cpu();

    // A pinned host buffer with non_blocking=True is the one case where the
    // per-pair path is genuinely asynchronous (RBLNCopy's async copy downgrades
    // to sync for pageable host memory). The multi entrypoints are sync-only, so
    // batching such a pair would trade a real overlap for a submit count.
    // Caveat: in a mixed list a later sync submit drains the in-flight transfer
    // anyway, so this only preserves the overlap when every pair takes it.
    const bool host_side_pinned_async = non_blocking &&
        ((dst_dev && src_cpu && c10::rbln::is_pinned_ptr(s.data_ptr())) ||
         (src_dev && dst_cpu && c10::rbln::is_pinned_ptr(dst.data_ptr())));

    // Host-direction batching is contiguous-only. A contiguous pair contributes
    // exactly one descriptor, so batching N of them replaces N submits with one —
    // always a win (measured ~3x at 1 KiB tensors, which is the weight-load shape).
    // A pair that fans out competes instead against copy_'s single bulk DMA over a
    // staged buffer, where the per-descriptor cost decides and the answer moves
    // with descriptor size. No caller in tree produces such a host pair today, so
    // they take the per-pair path; batching them belongs with the change that
    // routes plain copy_ through these primitives, where the threshold can be
    // justified against the same measurement.
    //
    // v2v keeps its strided path. There the per-pair path issues the same
    // on-device strided copy and, above the runtime's per-destination cap,
    // round-trips the host, so refusing a strided v2v pair would trade one submit
    // for N and can turn a device-side copy into a host bounce.
    const bool contig_pair = dst.is_contiguous() && s.is_contiguous();

    if (dst_dev && src_dev && dst.device() == s.device()) {
      strided_v2v_copy(dst, s, v2v_batch);
      v2v_batched.emplace_back(dst, s);
    } else if (dst_dev && src_cpu && contig_pair && !host_side_pinned_async) {
      h2v_copy(dst, s, h2v_batch);
      h2v_batched.emplace_back(dst, s);
    } else if (src_dev && dst_cpu && contig_pair && !host_side_pinned_async) {
      v2h_copy(dst, s, v2h_batch);
      v2h_batched.emplace_back(dst, s);
    } else {
      flush_pending();
      dst.copy_(s, non_blocking);
    }
  }

  // The host tensors enqueued above are all caller-owned list entries, alive for
  // the duration of this call, so no keep_alive registration is needed here.
  flush_pending();
}

} // namespace at::native::rbln
