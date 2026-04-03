#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

class RBLNTensorUtilsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
    ASSERT_EQ(c10::get_privateuse1_backend(true), "rbln");
    ASSERT_GE(c10::rbln::get_device_count(), 1);
  }

  void SetUp() override {
    c10::rbln::set_device_index(initial_device_index_);
    ASSERT_EQ(c10::rbln::get_device_index(), initial_device_index_);
  }

  const c10::DeviceIndex initial_device_index_ = 0;
  const c10::TensorOptions options_rbln_fp16_ = c10::TensorOptions().device(c10::kPrivateUse1).dtype(c10::kHalf);
  const c10::TensorOptions options_rbln_fp32_ = c10::TensorOptions().device(c10::kPrivateUse1).dtype(c10::kFloat);
};

TEST_F(RBLNTensorUtilsTest, GetCPUCopyOfContiguousRBLNTensor) {
  const auto sizes = std::vector<int64_t>{16, 64};
  const auto contiguous_rbln_tensor = at::randn(sizes, options_rbln_fp16_);
  EXPECT_EQ(contiguous_rbln_tensor.storage_offset(), 0);
  EXPECT_TRUE(contiguous_rbln_tensor.is_contiguous());
  const auto contiguous_storage_nbytes = at::detail::computeStorageNbytes(
      contiguous_rbln_tensor.sizes(),
      contiguous_rbln_tensor.strides(),
      contiguous_rbln_tensor.element_size(),
      contiguous_rbln_tensor.storage_offset());
  EXPECT_EQ(contiguous_rbln_tensor.storage().nbytes(), contiguous_storage_nbytes);

  // CPU copy uses compact storage (optimization: copies only data, not unused offset region)
  const auto data_only_nbytes = at::detail::computeStorageNbytes(
      contiguous_rbln_tensor.sizes(),
      contiguous_rbln_tensor.strides(),
      contiguous_rbln_tensor.element_size()); // No storage_offset

  const auto cpu_copy = at::native::rbln::get_cpu_copy_of_rbln_tensor(contiguous_rbln_tensor);
  EXPECT_TRUE(cpu_copy.device().is_cpu());
  EXPECT_EQ(cpu_copy.dtype(), contiguous_rbln_tensor.dtype());
  EXPECT_EQ(cpu_copy.sizes(), contiguous_rbln_tensor.sizes());
  EXPECT_EQ(cpu_copy.strides(), contiguous_rbln_tensor.strides());
  EXPECT_EQ(cpu_copy.storage_offset(), 0); // Always 0 (compact storage)
  EXPECT_EQ(cpu_copy.numel(), contiguous_rbln_tensor.numel());
  EXPECT_EQ(cpu_copy.is_contiguous(), contiguous_rbln_tensor.is_contiguous());
  EXPECT_EQ(cpu_copy.storage().nbytes(), data_only_nbytes);

  // Verify data correctness
  const auto cpu_tensor = contiguous_rbln_tensor.to(c10::kCPU);
  EXPECT_TRUE(cpu_copy.equal(cpu_tensor));
}

TEST_F(RBLNTensorUtilsTest, GetCPUCopyOfTransposedRBLNTensor) {
  const auto sizes = std::vector<int64_t>{16, 64};
  const auto contiguous_rbln_tensor = at::randn(sizes, options_rbln_fp16_);
  EXPECT_EQ(contiguous_rbln_tensor.storage_offset(), 0);
  EXPECT_TRUE(contiguous_rbln_tensor.is_contiguous());
  const auto contiguous_storage_nbytes = at::detail::computeStorageNbytes(
      contiguous_rbln_tensor.sizes(),
      contiguous_rbln_tensor.strides(),
      contiguous_rbln_tensor.element_size(),
      contiguous_rbln_tensor.storage_offset());
  EXPECT_EQ(contiguous_rbln_tensor.storage().nbytes(), contiguous_storage_nbytes);

  const auto transposed_rbln_tensor = contiguous_rbln_tensor.transpose(0, 1);
  EXPECT_EQ(transposed_rbln_tensor.sizes(), std::vector<int64_t>({64, 16}));
  EXPECT_EQ(transposed_rbln_tensor.storage_offset(), 0);
  EXPECT_FALSE(transposed_rbln_tensor.is_contiguous());
  EXPECT_EQ(transposed_rbln_tensor.storage().nbytes(), contiguous_storage_nbytes);
  const auto transposed_storage_nbytes = at::detail::computeStorageNbytes(
      transposed_rbln_tensor.sizes(),
      transposed_rbln_tensor.strides(),
      transposed_rbln_tensor.element_size(),
      transposed_rbln_tensor.storage_offset());
  EXPECT_EQ(transposed_storage_nbytes, contiguous_storage_nbytes);

  // CPU copy uses compact storage (optimization: copies only data, not unused offset region)
  const auto data_only_nbytes = at::detail::computeStorageNbytes(
      transposed_rbln_tensor.sizes(),
      transposed_rbln_tensor.strides(),
      transposed_rbln_tensor.element_size()); // No storage_offset

  const auto cpu_copy = at::native::rbln::get_cpu_copy_of_rbln_tensor(transposed_rbln_tensor);
  EXPECT_TRUE(cpu_copy.device().is_cpu());
  EXPECT_EQ(cpu_copy.dtype(), transposed_rbln_tensor.dtype());
  EXPECT_EQ(cpu_copy.sizes(), transposed_rbln_tensor.sizes());
  EXPECT_EQ(cpu_copy.strides(), transposed_rbln_tensor.strides());
  EXPECT_EQ(cpu_copy.storage_offset(), 0); // Always 0 (compact storage)
  EXPECT_EQ(cpu_copy.numel(), transposed_rbln_tensor.numel());
  EXPECT_EQ(cpu_copy.is_contiguous(), transposed_rbln_tensor.is_contiguous());
  EXPECT_EQ(cpu_copy.storage().nbytes(), data_only_nbytes);

  // Verify data correctness
  const auto cpu_tensor = transposed_rbln_tensor.to(c10::kCPU);
  EXPECT_TRUE(cpu_copy.equal(cpu_tensor));
}

TEST_F(RBLNTensorUtilsTest, GetCPUCopyOfSlicedRBLNTensor) {
  const auto sizes = std::vector<int64_t>{16, 64};
  const auto contiguous_rbln_tensor = at::randn(sizes, options_rbln_fp16_);
  EXPECT_EQ(contiguous_rbln_tensor.storage_offset(), 0);
  EXPECT_TRUE(contiguous_rbln_tensor.is_contiguous());
  const auto contiguous_storage_nbytes = at::detail::computeStorageNbytes(
      contiguous_rbln_tensor.sizes(),
      contiguous_rbln_tensor.strides(),
      contiguous_rbln_tensor.element_size(),
      contiguous_rbln_tensor.storage_offset());
  EXPECT_EQ(contiguous_rbln_tensor.storage().nbytes(), contiguous_storage_nbytes);

  const auto sliced_rbln_tensor = contiguous_rbln_tensor.slice(0, 4, 8);
  EXPECT_EQ(sliced_rbln_tensor.sizes(), std::vector<int64_t>({4, 64}));
  EXPECT_NE(sliced_rbln_tensor.storage_offset(), 0);
  EXPECT_TRUE(sliced_rbln_tensor.is_contiguous());
  EXPECT_EQ(sliced_rbln_tensor.storage().nbytes(), contiguous_storage_nbytes);
  const auto sliced_storage_nbytes = at::detail::computeStorageNbytes(
      sliced_rbln_tensor.sizes(),
      sliced_rbln_tensor.strides(),
      sliced_rbln_tensor.element_size(),
      sliced_rbln_tensor.storage_offset());
  EXPECT_LT(sliced_storage_nbytes, contiguous_storage_nbytes);

  // CPU copy uses compact storage (optimization: copies only data, not unused offset region)
  const auto data_only_nbytes = at::detail::computeStorageNbytes(
      sliced_rbln_tensor.sizes(),
      sliced_rbln_tensor.strides(),
      sliced_rbln_tensor.element_size()); // No storage_offset

  const auto cpu_copy = at::native::rbln::get_cpu_copy_of_rbln_tensor(sliced_rbln_tensor);
  EXPECT_TRUE(cpu_copy.device().is_cpu());
  EXPECT_EQ(cpu_copy.dtype(), sliced_rbln_tensor.dtype());
  EXPECT_EQ(cpu_copy.sizes(), sliced_rbln_tensor.sizes());
  EXPECT_EQ(cpu_copy.strides(), sliced_rbln_tensor.strides());
  EXPECT_EQ(cpu_copy.storage_offset(), 0); // Always 0 (compact storage)
  EXPECT_EQ(cpu_copy.numel(), sliced_rbln_tensor.numel());
  EXPECT_EQ(cpu_copy.is_contiguous(), sliced_rbln_tensor.is_contiguous());
  EXPECT_EQ(cpu_copy.storage().nbytes(), data_only_nbytes);

  // Verify data correctness
  const auto cpu_tensor = sliced_rbln_tensor.to(c10::kCPU);
  EXPECT_TRUE(cpu_copy.equal(cpu_tensor));
}

TEST_F(RBLNTensorUtilsTest, GetCPUCopyOfEmptyRBLNTensor) {
  const auto sizes = std::vector<int64_t>{0};
  const auto empty_rbln_tensor = at::randn(sizes, options_rbln_fp16_);
  EXPECT_EQ(empty_rbln_tensor.storage_offset(), 0);
  EXPECT_TRUE(empty_rbln_tensor.is_contiguous());
  EXPECT_EQ(empty_rbln_tensor.numel(), 0);
  EXPECT_EQ(empty_rbln_tensor.storage().nbytes(), 0);

  EXPECT_THROW(at::native::rbln::get_cpu_copy_of_rbln_tensor(empty_rbln_tensor), c10::Error);
}

TEST_F(RBLNTensorUtilsTest, CreateTensorFromPtrFrom2DTensor) {
  const auto device = c10::Device(c10::kPrivateUse1, 0);
  const auto sizes = std::vector<int64_t>{2, 3};
  const auto options = c10::TensorOptions().dtype(c10::kHalf).device(device);
  auto source_tensor = at::ones(sizes, options);

  auto data_ptr = reinterpret_cast<uint64_t>(source_tensor.data_ptr());

  auto result_tensor = at::native::rbln::create_tensor_from_ptr(data_ptr, sizes, c10::kHalf);
  EXPECT_EQ(result_tensor.device(), device);
  EXPECT_EQ(result_tensor.dtype(), c10::kHalf);
  EXPECT_EQ(result_tensor.sizes(), sizes);
  EXPECT_EQ(result_tensor.data_ptr(), source_tensor.data_ptr());

  // Verify data is shared
  source_tensor[0][0] = 5.0f;
  const auto result_cpu = result_tensor.to(c10::kCPU);
  const auto source_cpu = source_tensor.to(c10::kCPU);
  EXPECT_FLOAT_EQ(result_cpu[0][0].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(result_cpu[0][0].item<float>(), source_cpu[0][0].item<float>());
}

TEST_F(RBLNTensorUtilsTest, CreateTensorFromPtrFrom3DTensor) {
  const auto device = c10::Device(c10::kPrivateUse1, 0);
  const auto sizes = std::vector<int64_t>{2, 3, 4};
  const auto options = c10::TensorOptions().dtype(c10::kFloat).device(device);
  auto source_tensor = at::randn(sizes, options);

  auto data_ptr = reinterpret_cast<uint64_t>(source_tensor.data_ptr());

  const auto result_tensor = at::native::rbln::create_tensor_from_ptr(data_ptr, sizes, c10::kFloat);
  EXPECT_EQ(result_tensor.device(), device);
  EXPECT_EQ(result_tensor.dtype(), c10::kFloat);
  EXPECT_EQ(result_tensor.sizes(), sizes);
  EXPECT_EQ(result_tensor.data_ptr(), source_tensor.data_ptr());

  // Verify data is shared
  source_tensor[0][0][0] = 99.0f;
  const auto result_cpu = result_tensor.to(c10::kCPU);
  const auto source_cpu = source_tensor.to(c10::kCPU);
  EXPECT_FLOAT_EQ(result_cpu[0][0][0].item<float>(), 99.0f);
  EXPECT_FLOAT_EQ(result_cpu[0][0][0].item<float>(), source_cpu[0][0][0].item<float>());
}
