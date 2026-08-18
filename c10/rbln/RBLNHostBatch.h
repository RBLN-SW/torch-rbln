#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <memory>

namespace c10::rbln {

/**
 * @brief Buffered batch of pending host-to-device (h2v) copies.
 *
 * The write-side host counterpart of V2VBatch: enqueue() then submit()
 * record copy requests, submit() flushes them through rbln_memcpy_h2v_multi.
 * Where V2VBatch has to drop to per-entry copies once entries span devices —
 * paying a host bounce — this batch splits into one bulk submit per device,
 * because host memory is reachable from every device. Batching therefore
 * survives a heterogeneous batch; only the submit count grows.
 *
 * Overlap: destination (device) ranges must not overlap across entries, and the
 * runtime does not check this. Source (host) ranges are pure reads, so entries
 * may repeat or overlap them freely — which is what lets a stride-0 broadcast
 * source work without special handling.
 *
 * Ordering: entries are unordered with respect to each other, and a failed
 * submit may have already applied some copies (no rollback). Callers that need
 * ordering must not put the conflicting pairs in one batch.
 *
 * Lifetime: callers must invoke submit() on success. The destructor never issues
 * backend calls (a rejection during unwind would terminate the process); it just
 * warns when pending entries remain on a normal path.
 *
 * Host buffer lifetime: the runtime requires every source host buffer to stay
 * valid and unchanged until the submit call returns. Because this batch is
 * deferred, a caller that enqueued a temporary host buffer MUST hand it to
 * keep_alive() — otherwise submit() reads freed memory.
 *
 * Threading: not thread-safe — one batch per thread.
 */
class C10_RBLN_API H2VBatch {
 public:
  H2VBatch();
  ~H2VBatch();

  H2VBatch(const H2VBatch&) = delete;
  H2VBatch& operator=(const H2VBatch&) = delete;
  H2VBatch(H2VBatch&&) = delete;
  H2VBatch& operator=(H2VBatch&&) = delete;

  /**
   * @brief Enqueue one contiguous slab copy.
   *
   * @param dst    Destination device pointer (rbln virtual address).
   * @param src    Source host pointer.
   * @param nbytes Slab size in bytes. An nbytes==0 call is a no-op.
   */
  void enqueue(void* dst, const void* src, size_t nbytes);

  /**
   * @brief Tie a host buffer's lifetime to this batch, until submit() returns.
   *
   * Required for any enqueued source that the caller does not otherwise keep
   * alive across submit() — typically a staged temporary. The handle is opaque
   * because this layer must not depend on ATen; an ATen-level caller passes a
   * `std::shared_ptr<at::Tensor>` (or any holder whose destruction frees the
   * buffer).
   */
  void keep_alive(std::shared_ptr<void> holder);

  /**
   * @brief Flush queued operations to the backend. Idempotent.
   *
   * Entries are grouped per destination device; each group goes out as one
   * rbln_memcpy_h2v_multi call. Releases the keep_alive holders on return.
   */
  void submit();

  /**
   * @brief Number of pending (un-submitted) flat entries. For tests / debug.
   */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Buffered batch of pending device-to-host (v2h) copies.
 *
 * Mirror of H2VBatch with the roles swapped: the source is device memory and
 * anchors the per-device grouping, while the destination is host memory.
 * Destination (host) ranges must not overlap across entries; source (device)
 * ranges are pure reads and may repeat or overlap.
 *
 * Same deferred-submit consequence for host buffer lifetime, here on the
 * destination side: a destination the caller does not keep alive across submit()
 * must be handed to keep_alive().
 *
 * See H2VBatch for the ordering, no-rollback, destructor and threading notes.
 */
class C10_RBLN_API V2HBatch {
 public:
  V2HBatch();
  ~V2HBatch();

  V2HBatch(const V2HBatch&) = delete;
  V2HBatch& operator=(const V2HBatch&) = delete;
  V2HBatch(V2HBatch&&) = delete;
  V2HBatch& operator=(V2HBatch&&) = delete;

  /**
   * @brief Enqueue one contiguous slab copy.
   *
   * @param dst    Destination host pointer.
   * @param src    Source device pointer (rbln virtual address).
   * @param nbytes Slab size in bytes. An nbytes==0 call is a no-op.
   */
  void enqueue(void* dst, const void* src, size_t nbytes);

  /**
   * @brief Tie a host buffer's lifetime to this batch, until submit() returns.
   *        See H2VBatch::keep_alive.
   */
  void keep_alive(std::shared_ptr<void> holder);

  /**
   * @brief Flush queued operations to the backend. Idempotent.
   *
   * Entries are grouped per source device; each group goes out as one
   * rbln_memcpy_v2h_multi call. Releases the keep_alive holders on return.
   */
  void submit();

  /**
   * @brief Number of pending (un-submitted) flat entries. For tests / debug.
   */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace c10::rbln
