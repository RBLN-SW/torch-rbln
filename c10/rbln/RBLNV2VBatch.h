#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <memory>

namespace c10::rbln {

/**
 * @brief A buffered batch of pending device-to-device (v2v) copies.
 *
 * V2VBatch isolates the rest of torch-rbln from the rebel runtime's v2v API
 * surface. Callers enqueue logical copy requests; submit() flushes them to
 * the backend.
 *
 * Today the runtime exposes only a single flat call (rbln_memcpy_v2v), so
 * submit() drains the queue by issuing one call per entry. Two future
 * additions are anticipated:
 *
 *   1. A batched API that accepts a list of (dst, src, nbytes) in one call —
 *      submit() will rewrite to issue a single bulk call.
 *   2. A strided API that accepts (sizes, strides, inner_block) — enqueue_strided
 *      will stop expanding internally and forward the description directly.
 *
 * When either lands, only this class changes. Engine / kernel code that uses
 * V2VBatch stays the same.
 *
 * Lifetime: callers MUST invoke submit() on the success path. The destructor
 * is a leak-prevention safety net only — it never issues backend calls, since
 * a backend rejection during stack unwinding would terminate the process. If
 * the destructor sees pending entries on a normal (non-exceptional) path it
 * logs a warning so the missing submit() is caught in development; during
 * exception unwind it stays silent (the real error is the in-flight throw).
 *
 * Threading: not thread-safe. Each user thread should own its own batch.
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
   * Today this expands into `prod(outer_sizes)` enqueue() calls internally.
   * When the runtime exposes a strided v2v API, this will forward the
   * description without expansion.
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
   * @brief Number of pending (un-submitted) flat entries. For tests / debug.
   */
  size_t pending_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace c10::rbln
