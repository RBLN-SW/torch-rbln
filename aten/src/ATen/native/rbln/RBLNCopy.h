#pragma once

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

} // namespace at::native::rbln
