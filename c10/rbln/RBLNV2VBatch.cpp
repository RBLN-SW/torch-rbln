#include <c10/core/Device.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNV2VBatch.h>
#include <c10/rbln/detail/RBLNCopyBatchImpl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace c10::rbln {

namespace {
// Both ends of a v2v copy are device memory, so both must agree on the device.
constexpr auto kAnchor = detail::DeviceAnchor::kBothEnds;
constexpr const char* kWho = "V2VBatch";
} // namespace

struct V2VBatch::Impl {
  detail::BatchState<V2VCopyOp> st;
  std::vector<V2VStridedCopyOp> strided;
  size_t strided_rows = 0;

  void reset() noexcept {
    st.reset();
    strided.clear();
    strided_rows = 0;
  }
};

V2VBatch::V2VBatch() : impl_(std::make_unique<Impl>()) {}

V2VBatch::~V2VBatch() {
  if (impl_) {
    detail::warn_if_unsubmitted(impl_->st, kWho);
    if (impl_->st.pending.empty() && !impl_->strided.empty() && std::uncaught_exceptions() == 0) {
      RBLN_LOG_WARN("{} destroyed with {} pending strided entries — missing submit()", kWho, impl_->strided.size());
    }
  }
}

void V2VBatch::enqueue(void* dst, const void* src, size_t nbytes) {
  detail::enqueue_one<V2VCopyOp, kAnchor>(impl_->st, kWho, dst, src, nbytes);
}

void V2VBatch::enqueue_strided(
    void* dst,
    const void* src,
    size_t inner_block_bytes,
    c10::IntArrayRef outer_sizes,
    c10::IntArrayRef src_byte_strides,
    c10::IntArrayRef dst_byte_strides) {
  bool negative_stride = false;
  for (int64_t s : src_byte_strides) {
    negative_stride |= s < 0;
  }
  for (int64_t s : dst_byte_strides) {
    negative_stride |= s < 0;
  }
  if (outer_sizes.empty() || negative_stride || inner_block_bytes == 0 || src == nullptr ||
      !memcpy_v2v_strided_supported(get_torch_device_id(src)) ||
      outer_sizes.size() != src_byte_strides.size() || outer_sizes.size() != dst_byte_strides.size()) {
    detail::enqueue_strided_impl<V2VCopyOp, kAnchor>(
        impl_->st, kWho, dst, src, inner_block_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
    return;
  }
  RBLN_CHECK(dst != nullptr, "{}::enqueue_strided: dst is nullptr", kWho);
  RBLN_CHECK(src != nullptr, "{}::enqueue_strided: src is nullptr", kWho);
  for (int64_t s : outer_sizes) {
    RBLN_CHECK(s > 0, "{}::enqueue_strided: outer_sizes must be all positive, got {}", kWho, c10::str(outer_sizes));
  }
  detail::update_homogeneity<kAnchor>(impl_->st.homogeneous, impl_->st.anchor, src, dst);

  // The largest outer dim (innermost on a tie) becomes the runtime's chunk count; the others are
  // walked here, so a head-major -> token-major copy is one descriptor per head, not per token.
  size_t k = outer_sizes.size() - 1;
  for (size_t d = 0; d < outer_sizes.size(); ++d) {
    if (outer_sizes[d] > outer_sizes[k]) {
      k = d;
    }
  }
  const auto count = static_cast<size_t>(outer_sizes[k]);
  std::vector<int64_t> plane_sizes, plane_src_bs, plane_dst_bs;
  for (size_t d = 0; d < outer_sizes.size(); ++d) {
    if (d != k) {
      plane_sizes.push_back(outer_sizes[d]);
      plane_src_bs.push_back(src_byte_strides[d]);
      plane_dst_bs.push_back(dst_byte_strides[d]);
    }
  }
  int64_t plane_count = 1;
  for (int64_t s : plane_sizes) {
    plane_count *= s;
  }

  auto* dst_base = static_cast<uint8_t*>(dst);
  const auto* src_base = static_cast<const uint8_t*>(src);
  std::vector<int64_t> idx(plane_sizes.size(), 0);
  impl_->strided.reserve(impl_->strided.size() + static_cast<size_t>(plane_count));
  for (int64_t p = 0; p < plane_count; ++p) {
    int64_t src_off = 0;
    int64_t dst_off = 0;
    for (size_t d = 0; d < plane_sizes.size(); ++d) {
      src_off += idx[d] * plane_src_bs[d];
      dst_off += idx[d] * plane_dst_bs[d];
    }
    impl_->strided.push_back(
        {dst_base + dst_off, src_base + src_off, inner_block_bytes, count, src_byte_strides[k], dst_byte_strides[k]});
    detail::advance_index(idx, plane_sizes);
  }
  impl_->strided_rows += static_cast<size_t>(plane_count) * count;
}

void V2VBatch::submit() {
  if (!impl_ || (impl_->st.pending.empty() && impl_->strided.empty())) {
    return;
  }
  // RAII guard: reset batch state on any exit so the destructor's
  // "missing submit()" warning fires only when submit() was genuinely skipped.
  struct ResetGuard {
    Impl* impl;
    ~ResetGuard() noexcept {
      impl->reset();
    }
  } guard{impl_.get()};

  const auto& all = impl_->st.pending;
  const auto& strided = impl_->strided;

  const auto per_entry_strided = [](const V2VStridedCopyOp& s) {
    for (size_t i = 0; i < s.count; ++i) {
      memcpy_v2v(
          static_cast<uint8_t*>(s.dst) + static_cast<int64_t>(i) * s.dst_stride,
          static_cast<const uint8_t*>(s.src) + static_cast<int64_t>(i) * s.src_stride,
          s.chunk_bytes);
    }
  };

  if (!impl_->st.homogeneous) {
    // Heterogeneous — per-entry memcpy_v2v handles host-bounce internally and
    // tolerates any ordering between entries.
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (per-entry, cross-device)", pending_count());
    for (const auto& e : all) {
      memcpy_v2v(e.dst, e.src, e.nbytes);
    }
    for (const auto& s : strided) {
      per_entry_strided(s);
    }
    return;
  }

  // Strided entries go out in one call of their own; the runtime keeps what it
  // cannot run as strided descriptors on its existing paths. Per-entry on
  // rejection, for the same reason as the flat calls below.
  if (!strided.empty()) {
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} strided entries (batched)", strided.size());
    try {
      memcpy_v2v_strided_multi(strided);
    } catch (const c10::Error& e) {
      c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kV2VBatchToPerEntry, 0);
      RBLN_LOG_WARN(
          "V2VBatch::submit strided path rejected ({} entries) — falling back to per-entry: {}",
          strided.size(),
          e.what_without_backtrace());
      for (const auto& s : strided) {
        per_entry_strided(s);
      }
    }
  }
  if (all.empty()) {
    return;
  }

  // Submit `all[begin, end)` as one bulk call, per-entry on rejection.
  //
  // The runtime's ``CopyVirtualToVirtualMulti`` enforces stricter no-overlap
  // invariants between concurrent sub-copies than the per-entry path needs
  // (writes can be observed out-of-order across the batch), and some strided
  // patterns we emit from ``strided_v2v_copy`` — e.g. multi-slab ``cat`` into a
  // non-contig output where each slab fans out to thousands of narrow
  // sub-copies — hit that check even though no two sub-copies actually alias
  // each other. Catch the rejection and fall back to the per-entry path, which
  // has no such inter-copy ordering constraint. Replay is sound under an API
  // documenting no rollback because a copy is idempotent: a plain write, never
  // a read-modify-write, so a re-applied entry writes the same bytes. Retrying
  // one call's range rather than the whole batch keeps a rejection local to the
  // group that caused it.
  const auto submit_one_call = [&all](size_t begin, size_t end) {
    try {
      if (begin == 0 && end == all.size()) {
        memcpy_v2v_multi(all); // whole batch fits one call — submit it as is, no copy
      } else {
        memcpy_v2v_multi(std::vector<V2VCopyOp>(
            all.begin() + static_cast<std::ptrdiff_t>(begin), all.begin() + static_cast<std::ptrdiff_t>(end)));
      }
      return;
    } catch (const c10::Error& e) {
      // PROFILER (cold branch): batched v2v rejected by the runtime's no-overlap
      // check; the per-entry loop below has no inter-copy constraint.
      c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kV2VBatchToPerEntry, 0);
      RBLN_LOG_WARN(
          "V2VBatch::submit batched path rejected ({} entries) — falling back to per-entry: {}",
          end - begin,
          e.what_without_backtrace());
    }
    for (size_t i = begin; i < end; ++i) {
      memcpy_v2v(all[i].dst, all[i].src, all[i].nbytes);
    }
  };

  // Work one bulk call may carry. The runtime dispatches at most
  // ::rbln::kMaxV2VMultiCopies sub-copies per call through the device command
  // buffer; past that it completes the call by syncing the entries to the host
  // instead, which leaves the destination storage host-latest and makes the
  // next device consumer re-upload all of it. Splitting keeps every call on the
  // device: entries of one batch are disjoint by contract, so the split result
  // is the same. Unlike RBLNHostBatch's caps this one is the runtime's own
  // published constant, not a measured one.
  constexpr size_t kMaxBulkEntries = ::rbln::kMaxV2VMultiCopies;
  if (all.size() <= kMaxBulkEntries) {
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (batched)", all.size());
    submit_one_call(0, all.size());
    return;
  }
  RBLN_LOG_DEBUG(
      "V2VBatch::submit splitting {} entries to stay under the cap ({} entries)", all.size(), kMaxBulkEntries);
  for (size_t begin = 0; begin < all.size(); begin += kMaxBulkEntries) {
    submit_one_call(begin, std::min(begin + kMaxBulkEntries, all.size()));
  }
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->st.pending.size() + impl_->strided_rows : 0;
}

} // namespace c10::rbln
