#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/Copy.h>
#include <c10/core/MemoryFormat.h>

#include <optional>

namespace at::native::rbln {

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * @param src The source tensor.
 * @param dst The destination tensor.
 * @param non_blocking If true, the copy is performed asynchronously.
 * @return The destination tensor with copied data.
 *
 * @note The source tensor data is converted to match the dtype of the destination tensor during the copy.
 */
at::Tensor _copy_from_rbln(const at::Tensor& src, const at::Tensor& dst, bool non_blocking);

/**
 * @brief Resizes the destination tensor to match the source tensor and copies data from the source tensor to the destination tensor.
 *
 * @param src The source tensor.
 * @param dst The destination tensor.
 * @return The destination tensor with copied data.
 */
at::Tensor _copy_from_and_resize_rbln(const at::Tensor& src, const at::Tensor& dst);

/**
 * @brief Native impl of aten::clone for RBLN.
 *
 * Implements PyTorch's aten::clone semantics
 * (clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor)
 * by allocating a fresh output tensor with the requested memory format and
 * filling it from `self`. The fast path — `self` is contiguous with
 * `storage_offset()==0` and spans the entire storage — bypasses the
 * aten::copy_ dispatch and goes straight through `c10::rbln::memcpy_v2v`,
 * eliminating the redispatch + TensorIterator host overhead. The general
 * (non-contiguous / partial-storage view) case falls back to the standard
 * empty + copy_ decomposition so that strided gather semantics stay correct.
 */
at::Tensor clone_rbln(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format);

/**
 * @brief Native impl of aten::_foreach_copy_ for RBLN.
 *
 * Implements ``_foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool
 * non_blocking=False) -> ()`` — copies ``src[i]`` into ``self[i]`` for every i.
 * Conversion-free, same-shape pairs are enqueued per direction and flushed in
 * ONE submit each: ``V2VBatch`` for rbln->rbln, ``H2VBatch`` for cpu->rbln,
 * ``V2HBatch`` for rbln->cpu. A scatter into N per-layer tensors therefore
 * collapses from N submits to 1 in whichever direction it runs. Cross-device is
 * a disqualifier only for rbln->rbln; the host directions split per device.
 *
 * Pairs left on a plain per-pair ``copy_``:
 *   - broadcast or dtype cast (the multi entrypoints move bytes, never convert)
 *   - a pair fanning out to descriptors below the batching floor, where one
 *     staged bulk DMA beats many small transfers
 *   - a pinned host buffer with ``non_blocking=True``, the only case where the
 *     per-pair path is genuinely asynchronous
 *
 * A destination with internal overlap (an ``expand``ed view) is rejected, as
 * ``copy_`` rejects it: batch entries are unordered, so the surviving write
 * would be arbitrary.
 *
 * Batching is order-free, so when copies alias across pairs (a destination
 * overlapping another pair's source or destination) the op falls back to a
 * sequential per-pair copy_ loop to preserve list-order semantics. Queued work
 * is flushed before any inline ``copy_`` so a rejection mid-list cannot discard
 * the pairs that preceded it.
 */
void _foreach_copy__rbln(at::TensorList self, at::TensorList src, bool non_blocking);

} // namespace at::native::rbln
