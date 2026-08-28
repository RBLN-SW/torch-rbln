#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <ATen/ATen.h>

namespace at::native::rbln {

/**
 * @brief Returns a string representation of the tensor metadata.
 *
 * @param self The input tensor.
 * @return A string representation of the tensor metadata.
 */
C10_RBLN_API std::string get_tensor_metadata_string(const at::Tensor& self);

/**
 * @brief Creates and returns a CPU copy of the given RBLN tensor.
 *
 * The returned CPU tensor retains the same dtype, sizes, strides,
 * and storage offset as the input RBLN tensor.
 *
 * @param self The input RBLN tensor on a PrivateUse1 device.
 * @return A CPU tensor that is a copy of the input RBLN tensor.
 */
C10_RBLN_API at::Tensor get_cpu_copy_of_rbln_tensor(const at::Tensor& self);

/**
 * @brief Creates a tensor from a raw data pointer.
 *
 * @param data_ptr The raw data pointer address.
 * @param sizes The sizes of the tensor.
 * @param dtype The data type of the tensor.
 * @return The created tensor.
 */
C10_RBLN_API at::Tensor create_tensor_from_ptr(uint64_t data_ptr, c10::IntArrayRef sizes, c10::ScalarType dtype);

// Like create_tensor_from_ptr, but the tensor owns `base_ptr`: the last reference frees it with
// c10::rbln::free_nothrow. `data_ptr` may point inside the allocation at `base_ptr`.
C10_RBLN_API at::Tensor create_owning_tensor_from_ptr(
    uint64_t base_ptr,
    uint64_t data_ptr,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype);

/**
 * @brief True iff `a` and `b` reference the same logical view: same storage
 * data pointer, same storage_offset, and same strides.
 *
 * Used by kernels that may receive `out == self` via in-place dispatch
 * (e.g. `index_copy_`) or that want to short-circuit self-aliased copies
 * inside the strided v2v engine. Callers are responsible for any size
 * comparison they care about.
 */
inline bool is_same_view(const at::Tensor& a, const at::Tensor& b) {
  if (!a.has_storage() || !b.has_storage()) {
    return false;
  }
  if (a.storage().data() != b.storage().data()) {
    return false;
  }
  if (a.storage_offset() != b.storage_offset()) {
    return false;
  }
  return a.strides() == b.strides();
}

} // namespace at::native::rbln
