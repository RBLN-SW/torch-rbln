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

} // namespace at::native::rbln
