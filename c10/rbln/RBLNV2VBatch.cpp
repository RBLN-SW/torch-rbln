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

  const auto& all = impl_->st.pending;

  if (!impl_->st.homogeneous) {
    // Heterogeneous — per-entry memcpy_v2v handles host-bounce internally and
    // tolerates any ordering between entries.
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (per-entry, cross-device)", all.size());
    for (const auto& e : all) {
      memcpy_v2v(e.dst, e.src, e.nbytes);
    }
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
  return impl_ ? impl_->st.pending.size() : 0;
}

} // namespace c10::rbln
