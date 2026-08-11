#pragma once

// Internal to V2VBatch / H2VBatch / V2HBatch. Not installed and not for use
// outside c10/rbln/*.cpp: it exposes the pending storage the public headers hide
// behind a pimpl.
//
// Only the mechanical part of a batch is shared — pending storage, the row-major
// expansion of a strided description, device homogeneity, reset and destructor
// behaviour. submit() is not: v2v drops to
// per-entry (host-bouncing) copies once entries span devices and carries a
// runtime-rejection retry, while h2v/v2h split into one bulk submit per device.

#include <c10/core/Device.h>
#include <c10/rbln/RBLNFunctions.h> // get_torch_device_id
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace c10::rbln::detail {

/**
 * @brief Which side of a copy pins the batch to one device.
 *
 * A bulk entrypoint targets one device per call. Host pointers never constrain
 * it, so which end answers the question differs by direction.
 */
enum class DeviceAnchor : uint8_t {
  kBothEnds, // v2v: src and dst are both device memory
  kDstOnly, // h2v: dst is device, src is host
  kSrcOnly, // v2h: src is device, dst is host
};

/**
 * @brief Shared state behind one batch of pending copies.
 *
 * @tparam Desc Per-entry descriptor — V2VCopyOp / H2VCopyOp / V2HCopyOp. Same
 *         shape, deliberately distinct types; see H2VCopyOp for why.
 */
template <typename Desc>
struct BatchState {
  std::vector<Desc> pending;

  // Every entry so far agrees on `anchor` for the anchoring side(s). Maintained
  // at enqueue time so submit() needs no device lookups.
  bool homogeneous = true;
  c10::DeviceIndex anchor = -1;

  void reset() noexcept {
    pending.clear();
    homogeneous = true;
    anchor = -1;
  }
};

/**
 * @brief Update the one-device invariant for a newly enqueued (src, dst) pair.
 *
 * Short-circuits once heterogeneous. The anchoring pointer is looked up first,
 * so a disagreement costs one lookup; only kBothEnds needs a second, and only
 * while still homogeneous.
 */
template <DeviceAnchor kAnchor>
inline void update_homogeneity(bool& homogeneous, c10::DeviceIndex& anchor, const void* src, const void* dst) {
  if (!homogeneous) {
    return;
  }
  // The other end of a host direction is host memory and must not reach
  // get_torch_device_id().
  const void* device_side = (kAnchor == DeviceAnchor::kDstOnly) ? dst : src;
  const auto d = get_torch_device_id(device_side);
  if (anchor < 0) {
    anchor = d;
  } else if (anchor != d) {
    homogeneous = false;
    return;
  }
  if constexpr (kAnchor == DeviceAnchor::kBothEnds) {
    if (d != get_torch_device_id(dst)) {
      homogeneous = false;
    }
  }
}

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

/** @brief Record one contiguous slab copy. nbytes == 0 is a no-op. */
template <typename Desc, DeviceAnchor kAnchor>
inline void enqueue_one(BatchState<Desc>& st, const char* who, void* dst, const void* src, size_t nbytes) {
  RBLN_CHECK(dst != nullptr, "{}::enqueue: dst is nullptr", who);
  RBLN_CHECK(src != nullptr, "{}::enqueue: src is nullptr", who);
  if (nbytes == 0) {
    return;
  }
  update_homogeneity<kAnchor>(st.homogeneous, st.anchor, src, dst);
  st.pending.push_back({dst, src, nbytes});
}

/**
 * @brief Expand a strided description into `prod(outer_sizes)` flat entries.
 *
 * Entry k at outer index idx[] has dst_off = sum_d idx[d]*dst_byte_strides[d] and
 * likewise for src. A stride of 0 is preserved, which is what makes a broadcast
 * source replicate instead of collapsing to its first slab. The IntArrayRefs are
 * read during this call only.
 */
template <typename Desc, DeviceAnchor kAnchor>
inline void enqueue_strided_impl(
    BatchState<Desc>& st,
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
    enqueue_one<Desc, kAnchor>(st, who, dst, src, inner_block_bytes);
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
  update_homogeneity<kAnchor>(st.homogeneous, st.anchor, src, dst);

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

/**
 * @brief Destructor safety net. Never calls the backend — a rejection during
 *        unwind would terminate the process — and stays silent during unwind.
 */
template <typename Desc>
inline void warn_if_unsubmitted(const BatchState<Desc>& st, const char* who) noexcept {
  if (!st.pending.empty() && std::uncaught_exceptions() == 0) {
    RBLN_LOG_WARN("{} destroyed with {} pending entries — missing submit()", who, st.pending.size());
  }
}

} // namespace c10::rbln::detail
