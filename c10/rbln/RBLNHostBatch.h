#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <memory>

namespace c10::rbln {

/**
 * @brief Buffered batch of pending host-to-device (h2v) copies.
 *
 * Entries are grouped per device and each group is flushed as one
 * rbln_memcpy_h2v_multi. Host memory is reachable from every device, so a
 * heterogeneous batch stays batched — unlike V2VBatch, which must bounce
 * through the host once entries span devices.
 *
 * Contract:
 *   - Destination (device) ranges must not overlap across entries; the runtime
 *     requires this and does not check it. Source (host) ranges are pure reads
 *     and may repeat or overlap.
 *   - Entries are unordered, and a failed submit may have applied some of them
 *     (no rollback). Conflicting pairs must not share a batch.
 *   - Every source buffer must stay valid until submit() returns.
 *   - The destructor never calls the backend (a rejection during unwind would
 *     terminate the process); it warns if entries are still pending.
 *   - Not thread-safe: one batch per thread.
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
   * @param nbytes Slab size in bytes; 0 is a no-op.
   */
  void enqueue(void* dst, const void* src, size_t nbytes);

  /**
   * @brief Enqueue `prod(outer_sizes)` copies of `inner_block_bytes`, row-major.
   *
   * Entry k at outer index idx[] is offset sum_d idx[d]*{dst,src}_byte_strides[d].
   * The three arrays must have equal length; length 0 degenerates to enqueue(). A
   * src stride of 0 is preserved, so a broadcast host source replicates.
   */
  void enqueue_strided(
      void* dst,
      const void* src,
      size_t inner_block_bytes,
      c10::IntArrayRef outer_sizes,
      c10::IntArrayRef src_byte_strides,
      c10::IntArrayRef dst_byte_strides);

  /** @brief Flush pending entries, one bulk call per device. Idempotent. */
  void submit();

  /** @brief Pending (un-submitted) entry count. Tests / debug. */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Buffered batch of pending device-to-host (v2h) copies.
 *
 * Mirror of H2VBatch: the source is device memory and anchors the per-device
 * grouping, the destination is host memory. Destination (host) ranges must not
 * overlap; source (device) ranges may. The lifetime rule applies to the destination
 * here. See H2VBatch for the rest of the contract.
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
   * @param nbytes Slab size in bytes; 0 is a no-op.
   */
  void enqueue(void* dst, const void* src, size_t nbytes);

  /** @brief Enqueue a strided range. See H2VBatch::enqueue_strided. */
  void enqueue_strided(
      void* dst,
      const void* src,
      size_t inner_block_bytes,
      c10::IntArrayRef outer_sizes,
      c10::IntArrayRef src_byte_strides,
      c10::IntArrayRef dst_byte_strides);

  /** @brief Flush pending entries, one bulk call per device. Idempotent. */
  void submit();

  /** @brief Pending (un-submitted) entry count. Tests / debug. */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace c10::rbln
