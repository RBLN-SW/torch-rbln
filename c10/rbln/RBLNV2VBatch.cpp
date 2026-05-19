#include <c10/core/Device.h>
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

// Flip `homogeneous` to false if this (src, dst) pair would break the
// "all entries on one device" invariant required by the bulk dispatch.
// Once flipped, the lookup is skipped — no way to recover within a batch.
inline void update_homogeneity(bool& homogeneous, c10::DeviceIndex& anchor, const void* src, const void* dst) {
  if (!homogeneous) {
    return;
  }
  const auto s = static_cast<c10::DeviceIndex>(get_memory_info(src).torch_device_id);
  const auto d = static_cast<c10::DeviceIndex>(get_memory_info(dst).torch_device_id);
  if (s != d) {
    homogeneous = false;
    return;
  }
  if (anchor < 0) {
    anchor = s;
  } else if (anchor != s) {
    homogeneous = false;
  }
}

} // namespace

struct V2VBatch::Impl {
  // Flat list of pending slab descriptors. Storing V2VCopyOp directly lets
  // submit() hand the vector to memcpy_v2v_multi without an intermediate copy.
  std::vector<V2VCopyOp> pending;
  // submit() routes through the bulk memcpy_v2v_multi only while this stays
  // true (all entries on the same device == `anchor`). The first cross-device
  // or anchor-mismatching entry flips it; submit() then falls back to
  // per-entry memcpy_v2v which handles host-bounce on its own.
  bool homogeneous = true;
  c10::DeviceIndex anchor = -1;
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
  if (impl_->homogeneous) {
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (batched)", impl_->pending.size());
    memcpy_v2v_multi(impl_->pending);
  } else {
    // Heterogeneous batch — drain in enqueue order via per-entry memcpy_v2v,
    // which routes cross-device entries through a host bounce buffer.
    RBLN_LOG_DEBUG("V2VBatch::submit draining {} entries (per-entry fallback)", impl_->pending.size());
    for (const auto& e : impl_->pending) {
      memcpy_v2v(e.dst, e.src, e.nbytes);
    }
  }
  impl_->pending.clear();
  impl_->homogeneous = true;
  impl_->anchor = -1;
}

size_t V2VBatch::pending_count() const {
  return impl_ ? impl_->pending.size() : 0;
}

} // namespace c10::rbln
