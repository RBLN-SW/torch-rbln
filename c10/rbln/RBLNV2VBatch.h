#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <memory>

namespace c10::rbln {

/**
 * @brief Buffered batch of pending device-to-device (v2v) copies.
 *
 * Isolates callers from the rebel runtime's v2v API. enqueue() / enqueue_strided()
 * record copy requests; submit() flushes them through rbln_memcpy_v2v_multi when
 * every entry shares one device — in calls of at most
 * ::rbln::kMaxV2VMultiCopies entries, the most one call dispatches on the
 * device, plus one rbln_memcpy_v2v_strided_multi call for queued strided
 * entries — or falls back to per-entry memcpy_v2v (which host-bounces
 * cross-device entries). The path is decided from bookkeeping kept at enqueue
 * time, so submit() is O(N) with no extra lookups.
 *
 * Lifetime: callers must invoke submit() on success. The destructor never issues
 * backend calls (a rejection during unwind would terminate the process); it just
 * warns when pending entries remain on a normal path.
 *
 * Threading: not thread-safe — one batch per thread.
 */
class C10_RBLN_API V2VBatch {
 public:
  V2VBatch();
  ~V2VBatch();

  V2VBatch(const V2VBatch&) = delete;
  V2VBatch& operator=(const V2VBatch&) = delete;
  V2VBatch(V2VBatch&&) = delete;
  V2VBatch& operator=(V2VBatch&&) = delete;

  /**
   * @brief Enqueue one contiguous slab copy.
   *
   * @param dst    Destination device pointer (rbln virtual address).
   * @param src    Source device pointer (rbln virtual address).
   * @param nbytes Slab size in bytes. Must be > 0; an nbytes==0 call is a no-op.
   */
  void enqueue(void* dst, const void* src, size_t nbytes);

  /**
   * @brief Enqueue a strided range — `prod(outer_sizes)` copies of
   *        `inner_block_bytes` each, walking outer_sizes in row-major order.
   *
   * For outer iteration index `idx[]` the k-th call has:
   *   dst_off = sum_d idx[d] * dst_byte_strides[d]
   *   src_off = sum_d idx[d] * src_byte_strides[d]
   *
   * outer_sizes / src_byte_strides / dst_byte_strides must have identical
   * length. When that length is zero this degenerates to a single
   * enqueue(dst, src, inner_block_bytes).
   *
   * Where the device runs strided copies as-is (memcpy_v2v_strided_supported),
   * the largest outer dim (innermost on a tie) is forwarded to the runtime as
   * one strided descriptor and only the other dims are walked here. Otherwise,
   * or with negative strides, this expands into `prod(outer_sizes)` flat entries.
   *
   * The IntArrayRefs are read during this call only; the caller does not need
   * to keep them alive past return.
   */
  void enqueue_strided(
      void* dst,
      const void* src,
      size_t inner_block_bytes,
      c10::IntArrayRef outer_sizes,
      c10::IntArrayRef src_byte_strides,
      c10::IntArrayRef dst_byte_strides);

  /**
   * @brief Flush queued operations to the backend. Idempotent.
   */
  void submit();

  /**
   * @brief Number of pending (un-submitted) copies, counting each strided
   *        chunk as one entry. For tests / debug.
   */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace c10::rbln
