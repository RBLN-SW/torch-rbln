#include <c10/core/Device.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNV2VBatch.h>
#include <c10/rbln/detail/RBLNCopyBatchImpl.h>

#include <cstdint>
#include <vector>

namespace c10::rbln {

namespace {
// Both ends of a v2v copy are device memory, so both must agree on the device.
constexpr auto kAnchor = detail::DeviceAnchor::kBothEnds;
constexpr const char* kWho = "V2VBatch";

// Walk a multi-index in row-major order. Returns false when wrapping back to
// all zeros (iteration exhausted).
inline bool advance_index(std::vector<int64_t>& idx, c10::IntArrayRef sizes) {
  for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
    if (++idx[d] < sizes[d]) {
      return true;
    }
    idx[d] = 0;
  }
  return false;
}

/**
 * @brief Expand a strided description into `prod(outer_sizes)` flat entries.
 *
 * For outer iteration index `idx[]` the k-th entry has:
 *   dst_off = sum_d idx[d] * dst_byte_strides[d]
 *   src_off = sum_d idx[d] * src_byte_strides[d]
 *
 * A stride of 0 is preserved verbatim, which is what makes a broadcast source
 * replicate correctly instead of collapsing to its first slab.
 *
 * The IntArrayRefs are read during this call only.
 */
inline void enqueue_strided_v2v(
    detail::BatchState<V2VCopyOp>& st,
    const char* who,
    void* dst,
    const void* src,
    size_t inner_block_bytes,
    c10::IntArrayRef outer_sizes,
    c10::IntArrayRef src_byte_strides,
    c10::IntArrayRef dst_byte_strides) {
  RBLN_CHECK(dst != nullptr, "{}::enqueue_strided: dst is nullptr", who);
  RBLN_CHECK(src != nullptr, "{}::enqueue_strided: src is nullptr", who);
  RBLN_CHECK(
      outer_sizes.size() == src_byte_strides.size() && outer_sizes.size() == dst_byte_strides.size(),
      "{}::enqueue_strided: size/stride length mismatch (outer={}, src={}, dst={})",
      who,
      outer_sizes.size(),
      src_byte_strides.size(),
      dst_byte_strides.size());

  if (inner_block_bytes == 0) {
    return;
  }

  // Degenerate: no outer dims → single slab.
  if (outer_sizes.empty()) {
    detail::enqueue_one<V2VCopyOp, kAnchor>(st, who, dst, src, inner_block_bytes);
    return;
  }

  // Reject zero-extent outer dims (would yield zero work but a 0 in sizes
  // means the caller produced a 0-numel tensor — they should have early-returned).
  int64_t outer_count = 1;
  for (int64_t s : outer_sizes) {
    RBLN_CHECK(s > 0, "{}::enqueue_strided: outer_sizes must be all positive, got {}", who, c10::str(outer_sizes));
    outer_count *= s;
  }

  st.pending.reserve(st.pending.size() + static_cast<size_t>(outer_count));

  // All N expanded entries share the same base pointers — one lookup suffices.
  detail::update_homogeneity<kAnchor>(st.homogeneous, st.anchor, src, dst);

  auto* dst_base = static_cast<uint8_t*>(dst);
  const auto* src_base = static_cast<const uint8_t*>(src);

  std::vector<int64_t> idx(outer_sizes.size(), 0);
  for (int64_t o = 0; o < outer_count; ++o) {
    int64_t src_off = 0;
    int64_t dst_off = 0;
    for (size_t d = 0; d < outer_sizes.size(); ++d) {
      src_off += idx[d] * src_byte_strides[d];
      dst_off += idx[d] * dst_byte_strides[d];
    }
    st.pending.push_back({dst_base + dst_off, src_base + src_off, inner_block_bytes});
    advance_index(idx, outer_sizes);
  }
}
} // namespace

struct V2VBatch::Impl {
  detail::BatchState<V2VCopyOp> st;
};

V2VBatch::V2VBatch() : impl_(std::make_unique<Impl>()) {}

V2VBatch::~V2VBatch() {
  if (impl_) {
    detail::warn_if_unsubmitted(impl_->st, kWho);
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
  enqueue_strided_v2v(impl_->st, kWho, dst, src, inner_block_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
}

void V2VBatch::submit() {
  if (!impl_ || impl_->st.pending.empty()) {
    return;
  }
  // RAII guard: reset batch state on any exit so the destructor's
  // "missing submit()" warning fires only when submit() was genuinely skipped.
  struct ResetGuard {
    detail::BatchState<V2VCopyOp>* st;
    ~ResetGuard() noexcept {
      st->reset();
    }
  } guard{&impl_->st};

  bool drained = false;
  if (impl_->st.homogeneous) {
    // Try the batched fast path first. The runtime's
    // ``CopyVirtualToVirtualMulti`` enforces stricter no-overlap invariants
    // between concurrent sub-copies than the per-entry path needs (writes
    // can be observed out-of-order across the batch), and some strided
    // patterns we emit from ``strided_v2v_copy`` — e.g. multi-slab ``cat``
    // into a non-contig output where each slab fans out to thousands of
    // narrow sub-copies — hit that check even though no two sub-copies
    // actually alias each other. Catch the runtime rejection and fall back
    // to the per-entry path, which has no such inter-copy ordering
    // constraint. Correctness-equivalent to the dev path; loses the batch
    // throughput win only for the offending submit().
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (batched)", impl_->st.pending.size());
    try {
      memcpy_v2v_multi(impl_->st.pending);
      drained = true;
    } catch (const c10::Error& e) {
      // PROFILER (cold branch): batched v2v rejected by the runtime's no-overlap
      // check; we fall to the per-entry loop (host-bounces cross-device entries).
      c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kV2VBatchToPerEntry, 0);
      RBLN_LOG_WARN(
          "V2VBatch::submit batched path rejected ({} entries) — falling back to per-entry: {}",
          impl_->st.pending.size(),
          e.what_without_backtrace());
    }
  }
  if (!drained) {
    // Heterogeneous (or batched path was rejected) — per-entry memcpy_v2v
    // handles host-bounce internally and tolerates any ordering between
    // entries.
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (per-entry fallback)", impl_->st.pending.size());
    for (const auto& e : impl_->st.pending) {
      memcpy_v2v(e.dst, e.src, e.nbytes);
    }
  }
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->st.pending.size() : 0;
}

} // namespace c10::rbln
