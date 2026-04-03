#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

class RBLNCopyOpTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
    ASSERT_EQ(c10::get_privateuse1_backend(true), "rbln");
  }
  void SetUp() override {}

  at::TensorOptions cpu_options{at::TensorOptions().dtype(at::kFloat).device(at::kCPU)};
  at::TensorOptions rbln_options{at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1)};
};

TEST_F(RBLNCopyOpTest, CPUtoRBLN) {
  auto src = at::ones({1, 64});
  auto dst = src.to(at::kPrivateUse1);

  for (int i = 0; i < src.size(0); i++) {
    for (int j = 0; j < src.size(1); j++) {
      EXPECT_EQ(src[i][j].item().toFloat(), dst[i][j].item().toFloat());
    }
  }
}

TEST_F(RBLNCopyOpTest, RBLNtoCPU) {
  auto src = at::ones({1, 64}, rbln_options);
  auto dst = src.to(at::kCPU);

  for (int i = 0; i < src.size(0); i++) {
    for (int j = 0; j < src.size(1); j++) {
      EXPECT_EQ(src[i][j].item().toFloat(), dst[i][j].item().toFloat());
    }
  }
}

TEST_F(RBLNCopyOpTest, RBLNtoRBLN) {
  auto src = at::randn({1, 64}, rbln_options);
  auto dst = at::empty({1, 64}, rbln_options);
  dst.copy_(src);

  for (int i = 0; i < src.size(0); i++) {
    for (int j = 0; j < src.size(1); j++) {
      EXPECT_EQ(src[i][j].item().toFloat(), dst[i][j].item().toFloat());
    }
  }
}

TEST_F(RBLNCopyOpTest, CPUtoRBLN_UsingStorageOffset) {
  auto src = at::ones({2, 64});
  auto expected = src.slice(0, 1);
  auto dst = expected.to(at::kPrivateUse1);

  EXPECT_EQ(0, dst.storage_offset());
  EXPECT_EQ(expected.sizes(), dst.sizes());
  EXPECT_EQ(expected.strides(), dst.strides());
  EXPECT_EQ(expected.numel(), dst.numel());

  for (int i = 0; i < expected.size(0); i++) {
    for (int j = 0; j < expected.size(1); j++) {
      EXPECT_EQ(expected[i][j].item().toFloat(), dst[i][j].item().toFloat());
    }
  }
}

TEST_F(RBLNCopyOpTest, RBLNtoCPU_UsingStorageOffset) {
  auto src = at::ones({2, 64}, rbln_options);
  auto expected = src.slice(0, 1);
  auto dst = expected.to(at::kCPU);

  EXPECT_EQ(0, dst.storage_offset());
  EXPECT_EQ(expected.sizes(), dst.sizes());
  EXPECT_EQ(expected.strides(), dst.strides());
  EXPECT_EQ(expected.numel(), dst.numel());

  for (int i = 0; i < expected.size(0); i++) {
    for (int j = 0; j < expected.size(1); j++) {
      EXPECT_EQ(expected[i][j].item().toFloat(), dst[i][j].item().toFloat());
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
