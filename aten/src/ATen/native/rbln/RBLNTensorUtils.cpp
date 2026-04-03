#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHooksInterface.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/accumulate.h>

namespace at::native::rbln {

std::string get_tensor_metadata_string(const at::Tensor& self) {
  const auto self_device = self.device();
  RBLN_CHECK(
      self_device.is_cpu() || self_device.is_privateuseone(),
      "Only cpu and privateuseone devices are supported, but got {}",
      c10::str(self_device));

  const auto* self_data = self.data_ptr();
  const auto self_dtype = self.scalar_type();
  const auto self_numel = self.numel();
  const auto self_sizes = self.sizes();
  const auto self_strides = self.strides();
  const auto self_storage_offset = self.storage_offset();
  const auto self_is_contiguous = self.is_contiguous();

  const auto tensor_metadata_string = fmt::format(
      "Tensor(data={}, device={}, dtype={}, numel={}, sizes={}, strides={}, storage_offset={}, is_contiguous={})",
      fmt::ptr(self_data),
      c10::str(self_device),
      c10::str(self_dtype),
      self_numel,
      c10::str(self_sizes),
      c10::str(self_strides),
      self_storage_offset,
      self_is_contiguous);
  return tensor_metadata_string;
}

at::Tensor get_cpu_copy_of_rbln_tensor(const at::Tensor& self) {
  RBLN_CHECK(self.device().is_privateuseone());
  RBLN_CHECK(self.numel() > 0);

  const auto self_dtype = self.scalar_type();
  const auto self_sizes = self.sizes();
  const auto self_strides = self.strides();

  const auto self_element_size = self.element_size();
  const auto data_nbytes = at::detail::computeStorageNbytes(self_sizes, self_strides, self_element_size);
  const auto data_numel = static_cast<int64_t>(data_nbytes / self_element_size);

  auto cpu_tensor =
      at::empty({data_numel}, self_dtype, std::nullopt, c10::Device(c10::kCPU), false, c10::MemoryFormat::Contiguous);
  RBLN_CHECK(cpu_tensor.storage().nbytes() == data_nbytes);
  auto* dst_data = cpu_tensor.storage().mutable_data();
  const auto* src_data = self.data_ptr();
  c10::rbln::memcpy_v2h(dst_data, src_data, data_nbytes);

  const auto cpu_copy_of_rbln_tensor = cpu_tensor.as_strided(self_sizes, self_strides, 0);
  RBLN_CHECK(cpu_copy_of_rbln_tensor.device().is_cpu());
  RBLN_CHECK(cpu_copy_of_rbln_tensor.scalar_type() == self_dtype);
  RBLN_CHECK(cpu_copy_of_rbln_tensor.sizes() == self_sizes);
  RBLN_CHECK(cpu_copy_of_rbln_tensor.strides() == self_strides);
  RBLN_CHECK(cpu_copy_of_rbln_tensor.storage_offset() == 0);
  RBLN_CHECK(cpu_copy_of_rbln_tensor.storage().nbytes() == data_nbytes);
  return cpu_copy_of_rbln_tensor;
}

at::Tensor create_tensor_from_ptr(uint64_t data_ptr, c10::IntArrayRef sizes, c10::ScalarType dtype) {
  auto* data = reinterpret_cast<void*>(data_ptr); // NOLINT(performance-no-int-to-ptr)
  const auto device = c10::rbln::get_rbln_hooks()->getDeviceFromPtr(data);
  const auto options = c10::TensorOptions().dtype(dtype).device(device);
  return at::from_blob(data, sizes, options);
}

} // namespace at::native::rbln
