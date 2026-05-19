#pragma once

#include <ATen/core/Tensor.h>
#include <c10/rbln/RBLNV2VBatch.h>

namespace at::native::rbln {

/**
 * @brief Device-to-device strided copy primitive.
 *
 * Copies `src` into `dst` issuing one or more v2v calls, with no host
 * round-trip. Handles arbitrary stride combinations including slice/narrow,
 * permute/transpose, expand (stride==0), non-zero storage offset, and any
 * combination of contig / non-contig on either side.
 *
 * Algorithm: find the largest inner block that is contiguous in BOTH src and
 * dst (with stride==0 dims forced to outer iteration so broadcast positions
 * each get their own write), then enqueue a strided range describing the
 * outer iteration.
 *
 * Preconditions (enforced via RBLN_CHECK):
 *   - dst.sizes() == src.sizes()
 *   - dst.scalar_type() == src.scalar_type()
 *   - dst.device() == src.device(), both on a PrivateUse1 (RBLN) device
 *   - dst.numel() > 0  (caller is responsible for the 0-numel short-circuit)
 *
 * Aliasing / overlap is NOT checked here — `dst.copy_(src)` semantics for
 * overlapping storage are undefined upstream, so it is the higher-level
 * kernel's choice to assert or permit (cat asserts; copy_ does not).
 *
 * @param dst   Destination tensor.
 * @param src   Source tensor.
 * @param batch Pending-ops batch the work is appended to. The caller controls
 *              when it submits — for cat / index_select / multi-step copy we
 *              want one batch to span all sub-copies so future batched APIs
 *              can fuse them into one backend call.
 */
void strided_v2v_copy(
    const at::Tensor& dst,
    const at::Tensor& src,
    c10::rbln::V2VBatch& batch);

/**
 * @brief Convenience overload: allocates a temporary batch and submits it
 * inline. Use the batch-aware overload for multi-step kernels.
 */
void strided_v2v_copy(const at::Tensor& dst, const at::Tensor& src);

} // namespace at::native::rbln
