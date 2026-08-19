#pragma once

// Internal to V2VBatch / H2VBatch / V2HBatch. Not installed and not for use
// outside c10/rbln/*.cpp: it exposes the pending storage the public headers hide
// behind a pimpl.
//
// Only the mechanical part of a batch is shared — pending storage, device
// homogeneity, reset and destructor behaviour. submit() is not: v2v drops to
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
