#include <ATen/native/Resize.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

namespace at::native::rbln {

namespace {

c10::TensorImpl* resize_impl_rbln_(
    c10::TensorImpl* self_,
    c10::IntArrayRef sizes,
    std::optional<c10::IntArrayRef> strides_opt) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("self_data={}", fmt::ptr(self_->data()));

  if (self_->sizes() == sizes && (!strides_opt.has_value() || self_->strides() == strides_opt.value())) {
    RBLN_LOG_DEBUG("Tensor sizes and strides are already the same");
    return self_;
  }

  const auto type_meta = self_->dtype();
  const auto itemsize = type_meta.itemsize();
  const auto storage_offset = self_->storage_offset();

  size_t new_nbytes = 0;
  if (strides_opt.has_value()) {
    const auto strides = strides_opt.value();
    RBLN_LOG_DEBUG("Setting sizes to {} and strides to {}", c10::str(sizes), c10::str(strides));
    self_->set_sizes_and_strides(sizes, strides);

    new_nbytes = at::detail::computeStorageNbytes(sizes, strides, itemsize, storage_offset);
  } else {
    RBLN_LOG_DEBUG("Setting sizes to {}", c10::str(sizes));
    self_->set_sizes_contiguous(sizes);

    new_nbytes = at::detail::computeStorageNbytesContiguous(sizes, itemsize, storage_offset);
  }

  const auto& storage = self_->unsafe_storage();
  const void* old_data = (storage) ? storage.data() : nullptr;
  const size_t old_nbytes = (storage) ? storage.nbytes() : 0;
  RBLN_LOG_DEBUG("old_nbytes={}, new_nbytes={}", old_nbytes, new_nbytes);

  if ((self_->numel() > 0) && (new_nbytes > old_nbytes)) {
    RBLN_LOG_DEBUG("Resizing storage at {} to {} bytes", fmt::ptr(&storage), new_nbytes);
    resize_bytes_nocuda(storage, new_nbytes);
    const auto storage_nbytes = storage.nbytes();
    RBLN_CHECK(storage_nbytes == new_nbytes, "Failed to resize storage to requested nbytes");

    const auto* new_data = storage.data();
    RBLN_CHECK(new_data != old_data, "Failed to allocate new memory");
  }

  return self_;
}

} // namespace

const at::Tensor& resize_rbln_(
    const at::Tensor& self,
    c10::IntArrayRef sizes,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("self_data={}", fmt::ptr(self.data_ptr()));

  auto* self_ = self.unsafeGetTensorImpl();

  RBLN_LOG_DEBUG("Resizing to {}", c10::str(sizes));
  resize_impl_rbln_(self_, sizes, std::nullopt);
  if (memory_format_opt.has_value()) {
    const auto memory_format = memory_format_opt.value();
    RBLN_CHECK(memory_format != c10::MemoryFormat::Preserve, "Preserve memory format is not supported");

    RBLN_LOG_DEBUG("Restriding for {} memory format", c10::str(memory_format));
    self_->empty_tensor_restride(memory_format);
  }
  RBLN_CHECK(self_->sizes() == sizes, "Failed to resize tensor to requested sizes");

  return self;
}

at::Tensor& set_storage_rbln_(
    at::Tensor& self,
    c10::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG(
      "self_data={}, storage={}, storage_offset={}, sizes={}, strides={}",
      fmt::ptr(self.data_ptr()),
      fmt::ptr(&storage),
      storage_offset,
      c10::str(sizes),
      c10::str(strides));

  auto* self_ = self.unsafeGetTensorImpl();

  const auto& old_storage = self_->unsafe_storage();
  const void* old_data = (old_storage) ? old_storage.data() : nullptr;

  at::native::checkSetStorage(self, std::move(storage), storage_offset, sizes, strides);

  RBLN_LOG_DEBUG("Setting storage offset to {}", storage_offset);
  self_->set_storage_offset(storage_offset);
  RBLN_LOG_DEBUG("Setting sizes to {} and strides to {}", c10::str(sizes), c10::str(strides));
  std::optional<c10::IntArrayRef> strides_opt =
      (strides.data() != nullptr) ? std::optional<c10::IntArrayRef>(strides) : std::nullopt;
  resize_impl_rbln_(self_, sizes, strides_opt);

  return self;
}

} // namespace at::native::rbln
