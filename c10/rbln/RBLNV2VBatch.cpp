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
  detail::enqueue_strided_impl<V2VCopyOp, kAnchor>(
      impl_->st, kWho, dst, src, inner_block_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
}

void V2VBatch::submit(bool non_blocking) {
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
    RBLN_LOG_DEBUG(
        "V2VBatch::submit draining {} entries (batched{})", impl_->st.pending.size(), non_blocking ? ", async" : "");
    try {
      if (non_blocking) {
        memcpy_v2v_multi_async(impl_->st.pending);
      } else {
        memcpy_v2v_multi(impl_->st.pending);
      }
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
