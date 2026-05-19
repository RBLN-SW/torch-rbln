#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
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

} // namespace

struct V2VBatch::Impl {
  // Flat list of (dst, src, nbytes) entries.
  //
  // Today: each entry will be drained by one rbln_memcpy_v2v call in submit().
  // When the runtime adds a batched API this becomes the input to one bulk
  // call. The struct is intentionally trivial (no smart pointers / lifetimes)
  // so that future batching primitives can hand it to C-style APIs directly.
  struct Entry {
    void* dst;
    const void* src;
    size_t nbytes;
  };
  std::vector<Entry> pending;
};

V2VBatch::V2VBatch() : impl_(std::make_unique<Impl>()) {}

V2VBatch::~V2VBatch() {
  // Safety net only — never issue backend calls here. A rebel rejection
  // during stack unwinding would propagate out of the destructor and
  // terminate the process. Reaching this point with pending entries on a
  // normal path means the caller forgot submit(); log loudly so it gets
  // caught in dev. During exception unwind stay silent: the real error is
  // the in-flight throw, not the unsubmitted batch.
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
  // Drain in order. When a batched rebel API arrives this loop becomes a
  // single bulk call over impl_->pending.
  RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries", impl_->pending.size());
  for (const auto& e : impl_->pending) {
    memcpy_v2v(e.dst, e.src, e.nbytes);
  }
  impl_->pending.clear();
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->pending.size() : 0;
}

} // namespace c10::rbln
