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

  RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (batched)", all.size());
  memcpy_v2v_multi(all);
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->st.pending.size() : 0;
}

} // namespace c10::rbln
