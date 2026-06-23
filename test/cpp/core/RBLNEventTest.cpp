#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

class RBLNEventTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
  }

  void SetUp() override {
    c10::rbln::set_device_index(0);
  }
};

TEST_F(RBLNEventTest, NeverRecordedEventIsComplete) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  void* event = nullptr;
  EXPECT_TRUE(impl.queryEvent(event));
  impl.synchronizeEvent(event); // must be a no-op, not a crash
  impl.destroyEvent(event, 0);
}

TEST_F(RBLNEventTest, RecordSynchronizeOrdersAsyncCopy) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  constexpr c10::DeviceIndex device_index = 0;
  constexpr size_t nbytes = 4096;

  std::vector<int8_t> src_cpu(nbytes);
  for (size_t i = 0; i < nbytes; ++i) {
    src_cpu[i] = static_cast<int8_t>(i % 127);
  }
  auto* rbln_data = c10::rbln::malloc(device_index, nbytes);
  ASSERT_NE(rbln_data, nullptr);
  c10::rbln::memcpy_h2v_async(rbln_data, src_cpu.data(), nbytes);

  std::vector<int8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);

  void* event = nullptr;
  const auto stream = impl.getStream(c10::Device(c10::kPrivateUse1, device_index));
  impl.record(&event, stream, device_index, c10::EventFlag::PYTORCH_DEFAULT);
  impl.synchronizeEvent(event);
  EXPECT_EQ(dst_cpu, src_cpu);
  EXPECT_TRUE(impl.queryEvent(event));

  // Recording again must reuse the handle, and block() must drain too.
  c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);
  impl.record(&event, stream, device_index, c10::EventFlag::PYTORCH_DEFAULT);
  impl.block(event, stream);
  EXPECT_EQ(dst_cpu, src_cpu);

  impl.destroyEvent(event, device_index);
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNEventTest, SynchronizeDeviceDrainsQueue) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  constexpr c10::DeviceIndex device_index = 0;
  constexpr size_t nbytes = 4096;

  std::vector<int8_t> src_cpu(nbytes, 42);
  auto* rbln_data = c10::rbln::malloc(device_index, nbytes);
  ASSERT_NE(rbln_data, nullptr);
  c10::rbln::memcpy_h2v_async(rbln_data, src_cpu.data(), nbytes);

  std::vector<int8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);
  impl.synchronizeDevice(device_index);
  EXPECT_EQ(dst_cpu, src_cpu);

  c10::rbln::free(rbln_data);
}
