#include <c10/rbln/RBLNFunctions.h>
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

// ======== Async (non_blocking) copy tests ========

TEST_F(RBLNCopyOpTest, CPUtoRBLN_NonBlocking) {
  auto src = at::randn({4, 64});
  auto dst = src.to(at::kPrivateUse1, /*non_blocking=*/true);
  c10::rbln::synchronize(0);

  auto readback = dst.to(at::kCPU);
  EXPECT_TRUE(at::allclose(src, readback));
}

TEST_F(RBLNCopyOpTest, RBLNtoCPU_NonBlocking) {
  auto src_cpu = at::randn({4, 64});
  auto src = src_cpu.to(at::kPrivateUse1);

  auto dst = src.to(at::kCPU, /*non_blocking=*/true);
  c10::rbln::synchronize(0);

  EXPECT_TRUE(at::allclose(src_cpu, dst));
}

TEST_F(RBLNCopyOpTest, RBLNtoCPU_NonBlocking_LargeData) {
  auto src_cpu = at::randn({256, 1024});
  auto src = src_cpu.to(at::kPrivateUse1);

  auto dst = src.to(at::kCPU, /*non_blocking=*/true);
  c10::rbln::synchronize(0);

  EXPECT_TRUE(at::allclose(src_cpu, dst));
}

TEST_F(RBLNCopyOpTest, SyncAfterAsync_Consistent) {
  // Async H2V followed by sync V2H — sync should auto-drain pending
  auto src = at::randn({4, 64});
  auto rbln_tensor = src.to(at::kPrivateUse1, /*non_blocking=*/true);

  // Sync copy back — should auto-drain the async H2V first
  auto dst = rbln_tensor.to(at::kCPU);

  EXPECT_TRUE(at::allclose(src, dst));
}

TEST_F(RBLNCopyOpTest, NonBlocking_NonContiguous_FallsBackToSync) {
  // Non-contiguous tensors should fall back to sync copy
  auto src = at::randn({4, 64}, rbln_options);
  auto sliced = src.slice(1, 0, 32);  // non-contiguous

  // This should work (falls back to sync internally)
  auto dst = sliced.to(at::kCPU, /*non_blocking=*/true);
  auto expected = sliced.to(at::kCPU);

  EXPECT_TRUE(at::allclose(expected, dst));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
