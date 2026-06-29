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
 * non_blocking=False) -> ()`` — copies ``src[i]`` into ``self[i]`` for every
 * i. All conversion-free, same-device, same-shape pairs are enqueued into a
 * single ``V2VBatch`` and flushed in ONE ``rbln_memcpy_v2v_multi`` submit (the
 * write-side mirror of cat/stack's gather), so a scatter into N separate
 * per-layer tensors collapses from N submits to 1. Non-batchable pairs
 * (broadcast / dtype cast / cross-device) fall back to a plain per-pair copy_.
 *
 * Batching is order-free, so when copies alias across pairs (a destination
 * overlapping another pair's source or destination) the op falls back to a
 * sequential per-pair copy_ loop to preserve list-order semantics.
 */
void _foreach_copy__rbln(at::TensorList self, at::TensorList src, bool non_blocking);

} // namespace at::native::rbln
