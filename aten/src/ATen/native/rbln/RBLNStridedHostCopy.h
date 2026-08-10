#pragma once

#include <ATen/core/Tensor.h>
#include <c10/rbln/RBLNHostBatch.h>

#include <functional>

namespace at::native::rbln {

/**
 * @brief Host-to-device strided copy primitive.
 *
 * The host-side counterpart of strided_v2v_copy: describes `src` (CPU) → `dst`
 * (RBLN) as one or more h2v copies and appends them to `batch`, with no staging
 * allocation and no CPU-side pre-copy. Handles slice/narrow, permute/transpose,
 * expand (stride==0), non-zero storage offset, and any contig / non-contig
 * combination on either side.
 *
 * Algorithm is shared with the v2v case: find the largest inner block that is
 * contiguous in BOTH tensors via common_inner_start() — the stride analysis does
 * not care which side is host memory — then describe the outer iteration. A
 * stride==0 dim is forced to the outer loop so each broadcast position gets its
 * own write instead of collapsing to the first slab.
 *
 * Preconditions (enforced via RBLN_CHECK):
 *   - dst.sizes() == src.sizes()
 *   - dst.scalar_type() == src.scalar_type()  (no dtype conversion here)
 *   - dst is on a PrivateUse1 (RBLN) device, src is on CPU
 *   - dst.numel() > 0  (caller owns the 0-numel short-circuit)
 *
 * Destination ranges must not overlap across everything queued in `batch` — the
 * runtime requires disjoint destinations and does not validate it. A single call
 * cannot produce an overlap on its own (a dst stride of 0 would, and dst is a
 * real output so it never has one), but a caller batching several pairs is
 * responsible across them.
 *
 * Host buffer lifetime: the runtime reads `src` during submit, which happens
 * later. If `src` is a temporary the caller does not otherwise hold, register it
 * with `batch.keep_alive()` — see H2VBatch.
 *
 * @param dst   Destination tensor (RBLN device).
 * @param src   Source tensor (CPU).
 * @param batch Pending-ops batch the work is appended to. The caller controls
 *              when it submits, so a multi-pair kernel can fuse every sub-copy
 *              into one backend call.
 */
void strided_h2v_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::H2VBatch& batch);

/**
 * @brief Device-to-host strided copy primitive.
 *
 * Mirror of strided_h2v_copy with the roles swapped: `src` is on the RBLN device
 * and `dst` on CPU. Same stride analysis, same preconditions with the devices
 * exchanged.
 *
 * Source ranges may repeat or overlap across the batch (the runtime treats them
 * as pure reads); destinations may not.
 *
 * Host buffer lifetime applies to `dst` here: it is written during submit, so a
 * destination the caller does not hold must go through `batch.keep_alive()`.
 */
void strided_v2h_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::V2HBatch& batch);

/**
 * @brief Submits `batch`, invoking `cpu_fallback` on a backend call failure.
 *
 * The host-batch analogue of the v2v submit_or_fallback. Only h2v/v2h backend
 * rejections (identified by the "rbln_memcpy_h2v_multi failed" /
 * "rbln_memcpy_v2h_multi failed" / per-entry "rbln_memcpy_h2v failed" /
 * "rbln_memcpy_v2h failed" substrings) route to `cpu_fallback`; every other
 * `c10::Error` propagates so caller-side bugs surface. Gated by
 * `TORCH_RBLN_DISABLE_FALLBACK=strided_copy_error` (default enabled), same knob
 * as the v2v path.
 *
 * Whether the runtime ever rejects a well-formed h2v/v2h batch is not known —
 * the analogous v2v rejection is specific to device destinations at interior
 * offsets of large untyped pool allocations, and these directions have no device
 * destination (h2v) or read-only device source (v2h). The fallback exists so
 * that if it does happen the result is a slow copy rather than a hard failure,
 * and the recorded bounce makes it visible instead of silent.
 */
void submit_or_fallback(c10::rbln::H2VBatch& batch, const char* op_name, std::function<void()> cpu_fallback);

/** @brief V2H overload of submit_or_fallback. See the H2V version. */
void submit_or_fallback(c10::rbln::V2HBatch& batch, const char* op_name, std::function<void()> cpu_fallback);

} // namespace at::native::rbln
