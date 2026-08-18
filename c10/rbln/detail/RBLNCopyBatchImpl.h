#pragma once

// Internal implementation detail of V2VBatch / H2VBatch / V2HBatch. NOT part of
// the installed header set and NOT for use outside c10/rbln/*.cpp: it exposes
// the batch's pending storage, which the public headers deliberately hide
// behind a pimpl so the layout can change without breaking already-compiled
// consumers.
//
// What is shared here is only the mechanical part of a batch: the pending
// vector, the row-major expansion of a strided description into flat entries,
// the device-homogeneity bookkeeping, and the destructor / reset behaviour.
//
// submit() is deliberately NOT shared. The three directions diverge there and
// have nothing to gain from a common path: v2v has to drop to per-entry copies
// (which host-bounce) when entries span devices, while h2v/v2h can split into
// one batched submit per device because host memory is reachable from all of
// them. v2v's submit also carries a runtime-rejection retry whose failure mode
// is specific to device-to-device copies. Keeping submit() per class leaves that
// logic untouched by this sharing.

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
 * The bulk runtime entrypoints target one device per call, so a batch tracks
 * whether its entries agree. Which pointer answers that differs by direction;
 * host pointers never constrain it.
 */
enum class DeviceAnchor : uint8_t {
  kBothEnds, // v2v: src and dst are both device memory
  kDstOnly, // h2v: dst is device, src is host
  kSrcOnly, // v2h: src is device, dst is host
};

/**
 * @brief Shared mechanical state behind one batch of pending copies.
 *
 * @tparam Desc Per-entry descriptor — V2VCopyOp / H2VCopyOp / V2HCopyOp. Same
 *         shape, deliberately distinct types; see H2VCopyOp for why.
 */
template <typename Desc>
struct BatchState {
  std::vector<Desc> pending;

  // True while every enqueued entry agrees on `anchor` for the anchoring
  // side(s). Maintained at enqueue time so submit() needs no extra lookups.
  bool homogeneous = true;
  c10::DeviceIndex anchor = -1;

  // Host buffers the runtime must be able to read (h2v src) or write (v2h dst)
  // for the duration of the submit call. A batch is deferred — enqueue only
  // records an address — so a caller that staged a temporary host tensor has to
  // hand ownership over here or the submit reads freed memory. Opaque because
  // this layer must not depend on ATen; the ATen-level caller supplies the
  // concrete holder.
  std::vector<std::shared_ptr<void>> keepalive;

  void reset() noexcept {
    pending.clear();
    homogeneous = true;
    anchor = -1;
    keepalive.clear();
  }
};
/**
 * @brief Update the one-device invariant for a newly enqueued (src, dst) pair.
 *
 * Once `homogeneous` is false every later call short-circuits, so a batch that
 * has already gone heterogeneous pays no further device lookups.
 *
 * The device-side pointer is looked up first; when it already disagrees with the
 * anchor the second lookup is skipped entirely. Only kBothEnds needs two
 * lookups, and only on the happy path, to catch a cross-device v2v pair.
 */
template <DeviceAnchor kAnchor>
inline void update_homogeneity(bool& homogeneous, c10::DeviceIndex& anchor, const void* src, const void* dst) {
  if (!homogeneous) {
    return;
  }
  // The anchoring device pointer. The other end (kDstOnly / kSrcOnly) is host
  // memory and must not be handed to get_torch_device_id().
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

/**
 * @brief Record one contiguous slab copy.
 *
 * nbytes == 0 is a no-op (callers with genuinely empty tensors short-circuit
 * earlier; accepting it here keeps every enqueue site free of the guard).
 */
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
 * @brief Destructor safety net shared by every batch class.
 *
 * Never issues backend calls: a rejection during stack unwind would terminate
 * the process. On a normal path with pending entries it warns so the missing
 * submit() gets caught; during unwind it stays silent.
 */
template <typename Desc>
inline void warn_if_unsubmitted(const BatchState<Desc>& st, const char* who) noexcept {
  if (!st.pending.empty() && std::uncaught_exceptions() == 0) {
    RBLN_LOG_WARN("{} destroyed with {} pending entries — missing submit()", who, st.pending.size());
  }
}

} // namespace c10::rbln::detail
