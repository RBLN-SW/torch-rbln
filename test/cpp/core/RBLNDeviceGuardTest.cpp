#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

class RBLNDeviceGuardTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
    ASSERT_EQ(c10::get_privateuse1_backend(true), "rbln");
  }

  void SetUp() override {
    c10::rbln::set_device_index(initial_device_index_);
    ASSERT_EQ(c10::rbln::get_device_index(), initial_device_index_);
  }

  const c10::DeviceIndex initial_device_index_ = 0;
};

TEST_F(RBLNDeviceGuardTest, DeviceGuard) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; device_index++) {
    const auto original_device_index = c10::rbln::get_device_index();
    const auto original_device = c10::Device(c10::kPrivateUse1, original_device_index);
    {
      const auto device = c10::Device(c10::kPrivateUse1, device_index);
      const auto guard = c10::DeviceGuard(device);
      EXPECT_EQ(guard.original_device(), original_device);
      EXPECT_EQ(guard.current_device(), device);
      EXPECT_EQ(c10::rbln::get_device_index(), device_index);
    }
    EXPECT_EQ(c10::rbln::get_device_index(), original_device_index);
  }
}

static void TestNestedGuard(c10::DeviceIndex device_index, c10::DeviceIndex device_count) {
  if (device_count == 0) {
    return;
  }

  const auto original_device_index = c10::rbln::get_device_index();
  const auto original_device = c10::Device(c10::kPrivateUse1, original_device_index);
  {
    const auto device = c10::Device(c10::kPrivateUse1, device_index);
    const auto guard = c10::DeviceGuard(device);
    EXPECT_EQ(guard.original_device(), original_device);
    EXPECT_EQ(guard.current_device(), device);
    EXPECT_EQ(c10::rbln::get_device_index(), device_index);
    TestNestedGuard(++device_index, --device_count);
  }
  EXPECT_EQ(c10::rbln::get_device_index(), original_device_index);
}

TEST_F(RBLNDeviceGuardTest, NestedDeviceGuard) {
  const auto device_count = c10::rbln::get_device_count();
  ASSERT_GE(device_count, 1);
  TestNestedGuard(0, device_count);
}

TEST_F(RBLNDeviceGuardTest, MixedDeviceGuard) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);

  const auto original_rbln_device_index = c10::rbln::get_device_index();
  const auto original_rbln_device = c10::Device(c10::kPrivateUse1, original_rbln_device_index);
  for (c10::DeviceIndex rbln_device_index = 0; rbln_device_index < device_count; rbln_device_index++) {
    const auto rbln_device = c10::Device(c10::kPrivateUse1, rbln_device_index);
    const auto cpu_device = c10::Device(c10::kCPU);
    {
      const auto cpu_guard = c10::DeviceGuard(cpu_device);
      EXPECT_EQ(cpu_guard.original_device(), cpu_device);
      EXPECT_EQ(cpu_guard.current_device(), cpu_device);
      EXPECT_EQ(c10::rbln::get_device_index(), original_rbln_device_index);
      {
        const auto rbln_guard = c10::DeviceGuard(rbln_device);
        EXPECT_EQ(rbln_guard.original_device(), original_rbln_device);
        EXPECT_EQ(rbln_guard.current_device(), rbln_device);
        EXPECT_EQ(c10::rbln::get_device_index(), rbln_device_index);
        EXPECT_EQ(cpu_guard.original_device(), cpu_device);
        EXPECT_EQ(cpu_guard.current_device(), cpu_device);
      }
      EXPECT_EQ(cpu_guard.original_device(), cpu_device);
      EXPECT_EQ(cpu_guard.current_device(), cpu_device);
      EXPECT_EQ(c10::rbln::get_device_index(), original_rbln_device_index);
    }
    EXPECT_EQ(c10::rbln::get_device_index(), original_rbln_device_index);

    {
      const auto rbln_guard = c10::DeviceGuard(rbln_device);
      EXPECT_EQ(rbln_guard.original_device(), original_rbln_device);
      EXPECT_EQ(rbln_guard.current_device(), rbln_device);
      EXPECT_EQ(c10::rbln::get_device_index(), rbln_device_index);
      {
        const auto cpu_guard = c10::DeviceGuard(cpu_device);
        EXPECT_EQ(cpu_guard.original_device(), cpu_device);
        EXPECT_EQ(cpu_guard.current_device(), cpu_device);
        EXPECT_EQ(c10::rbln::get_device_index(), rbln_device_index);
        EXPECT_EQ(rbln_guard.original_device(), original_rbln_device);
        EXPECT_EQ(rbln_guard.current_device(), rbln_device);
      }
      EXPECT_EQ(rbln_guard.original_device(), original_rbln_device);
      EXPECT_EQ(rbln_guard.current_device(), rbln_device);
      EXPECT_EQ(c10::rbln::get_device_index(), rbln_device_index);
    }
    EXPECT_EQ(c10::rbln::get_device_index(), original_rbln_device_index);
  }
}
