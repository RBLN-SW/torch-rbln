#include <c10/core/Device.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNV2VBatch.h>

#include <cstdint>
#include <exception>
#include <vector>

namespace c10::rbln {

namespace {

// Walk a multi-index in row-major order. Returns false when wrapping back to
// all zeros (iteration exhausted).
inline bool advance(std::vector<int64_t>& idx, c10::IntArrayRef sizes) {
  for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; --d) {
    if (++idx[d] < sizes[d]) {
      return true;
    }
    idx[d] = 0;
  }
  return false;
}

// Flip `homogeneous` to false if this (src, dst) pair would break the
// "all entries on one device" invariant required by the bulk dispatch.
// Once flipped, all subsequent calls short-circuit.
//
// src is looked up first; if it already mismatches the anchor we can skip
// the dst lookup entirely. The dst lookup only runs on the happy path
// (src matches anchor) to catch cross-device entries.
inline void update_homogeneity(bool& homogeneous, c10::DeviceIndex& anchor, const void* src, const void* dst) {
  if (!homogeneous) {
    return;
  }
  const auto s = get_torch_device_id(src);
  if (anchor < 0) {
    anchor = s;
  } else if (anchor != s) {
    homogeneous = false;
    return;
  }
  const auto d = get_torch_device_id(dst);
  if (s != d) {
    homogeneous = false;
  }
}

} // namespace

struct V2VBatch::Impl {
  std::vector<V2VCopyOp> pending;
  // True while every enqueued (src, dst) pair shares one device == `anchor`.
  // submit() takes the bulk path when set, per-entry fallback otherwise.
  bool homogeneous = true;
  c10::DeviceIndex anchor = -1;
};

V2VBatch::V2VBatch() : impl_(std::make_unique<Impl>()) {}

V2VBatch::~V2VBatch() {
  // Safety net — never issue backend calls (a rejection during stack unwind
  // would terminate the process). On a normal path with pending entries,
  // warn so the missing submit() gets caught; during unwind, stay silent.
  if (impl_ && !impl_->pending.empty() && std::uncaught_exceptions() == 0) {
    RBLN_LOG_WARN("V2VBatch destroyed with {} pending entries — missing submit()", impl_->pending.size());
  }
}

void V2VBatch::enqueue(void* dst, const void* src, size_t nbytes) {
  RBLN_CHECK(dst != nullptr, "V2VBatch::enqueue: dst is nullptr");
  RBLN_CHECK(src != nullptr, "V2VBatch::enqueue: src is nullptr");
  if (nbytes == 0) {
    return;
  }
  update_homogeneity(impl_->homogeneous, impl_->anchor, src, dst);
  impl_->pending.push_back({dst, src, nbytes});
}

void V2VBatch::enqueue_strided(
    void* dst,
    const void* src,
    size_t inner_block_bytes,
    c10::IntArrayRef outer_sizes,
    c10::IntArrayRef src_byte_strides,
    c10::IntArrayRef dst_byte_strides) {
  RBLN_CHECK(dst != nullptr, "V2VBatch::enqueue_strided: dst is nullptr");
  RBLN_CHECK(src != nullptr, "V2VBatch::enqueue_strided: src is nullptr");
  RBLN_CHECK(
      outer_sizes.size() == src_byte_strides.size() && outer_sizes.size() == dst_byte_strides.size(),
      "V2VBatch::enqueue_strided: size/stride length mismatch (outer={}, src={}, dst={})",
      outer_sizes.size(),
      src_byte_strides.size(),
      dst_byte_strides.size());

  if (inner_block_bytes == 0) {
    return;
  }

  // Degenerate: no outer dims → single slab.
  if (outer_sizes.empty()) {
    enqueue(dst, src, inner_block_bytes);
    return;
  }

  // Reject zero-extent outer dims (would yield zero work but a 0 in sizes
  // means the caller produced a 0-numel tensor — they should have early-returned).
  int64_t outer_count = 1;
  for (int64_t s : outer_sizes) {
    RBLN_CHECK(s > 0, "V2VBatch::enqueue_strided: outer_sizes must be all positive, got {}", c10::str(outer_sizes));
    outer_count *= s;
  }

  impl_->pending.reserve(impl_->pending.size() + static_cast<size_t>(outer_count));

  // All N expanded entries share the same base devices — one lookup suffices.
  update_homogeneity(impl_->homogeneous, impl_->anchor, src, dst);

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
    impl_->pending.push_back({dst_base + dst_off, src_base + src_off, inner_block_bytes});
    advance(idx, outer_sizes);
  }
}

void V2VBatch::submit() {
  if (!impl_ || impl_->pending.empty()) {
    return;
  }
  // RAII guard: reset batch state on any exit so the destructor's
  // "missing submit()" warning fires only when submit() was genuinely skipped.
  struct ResetGuard {
    Impl* impl;
    ~ResetGuard() noexcept {
      impl->pending.clear();
      impl->homogeneous = true;
      impl->anchor = -1;
    }
  } guard{impl_.get()};

  bool drained = false;
  if (impl_->homogeneous) {
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
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (batched)", impl_->pending.size());
    try {
      memcpy_v2v_multi(impl_->pending);
      drained = true;
    } catch (const c10::Error& e) {
      // PROFILER (cold branch): batched v2v rejected by the runtime's no-overlap
      // check; we fall to the per-entry loop (host-bounces cross-device entries).
      c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kV2VBatchToPerEntry, 0);
      RBLN_LOG_WARN(
          "V2VBatch::submit batched path rejected ({} entries) — falling back to per-entry: {}",
          impl_->pending.size(),
          e.what_without_backtrace());
    }
  }
  if (!drained) {
    // Heterogeneous (or batched path was rejected) — per-entry memcpy_v2v
    // handles host-bounce internally and tolerates any ordering between
    // entries.
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (per-entry fallback)", impl_->pending.size());
    for (const auto& e : impl_->pending) {
      memcpy_v2v(e.dst, e.src, e.nbytes);
    }
  }
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->pending.size() : 0;
}

} // namespace c10::rbln
