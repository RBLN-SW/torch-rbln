#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// Exercises the event half of RBLNGuardImpl through VirtualGuardImpl (the same path
// torch.Event takes): record, block, query, synchronize, destroy. Host-observable
// data assertions use synchronizeDevice.
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
  impl.synchronizeEvent(event); // no-op, not a crash
  impl.block(event, impl.getStream(c10::Device(c10::kPrivateUse1, 0))); // no-op
  impl.destroyEvent(event, 0);
}

TEST_F(RBLNEventTest, RecordSynchronizeCompletesAsyncCopy) {
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
  impl.synchronizeEvent(event); // the recorded work has completed
  EXPECT_TRUE(impl.queryEvent(event)); // ... and query agrees
  impl.synchronizeDevice(device_index); // finalize the pending D2H transfer
  EXPECT_EQ(dst_cpu, src_cpu);

  // Re-record reuses the handle and snapshots the newer copy.
  std::fill(dst_cpu.begin(), dst_cpu.end(), int8_t{0});
  c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);
  impl.record(&event, stream, device_index, c10::EventFlag::PYTORCH_DEFAULT);
  impl.synchronizeEvent(event);
  impl.synchronizeDevice(device_index);
  EXPECT_EQ(dst_cpu, src_cpu);

  impl.destroyEvent(event, device_index);
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNEventTest, WaitEventAcrossStreamsDoesNotError) {
  // stream B waits on an event recorded on stream A. Asserts the plumbing works end
  // to end and completion is observable.
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  constexpr c10::DeviceIndex device_index = 0;
  constexpr size_t nbytes = 4096;
  const auto device = c10::Device(c10::kPrivateUse1, device_index);

  std::vector<int8_t> src_cpu(nbytes, 7);
  auto* rbln_data = c10::rbln::malloc(device_index, nbytes); // also materializes the context
  ASSERT_NE(rbln_data, nullptr);

  const auto stream_a = impl.getNewStream(device, 0);
  const auto stream_b = impl.getNewStream(device, 0);

  const auto prev = impl.exchangeStream(stream_a);
  c10::rbln::memcpy_h2v_async(rbln_data, src_cpu.data(), nbytes);
  void* event = nullptr;
  impl.record(&event, stream_a, device_index, c10::EventFlag::PYTORCH_DEFAULT);
  impl.block(event, stream_b); // B waits for A's event; must not error
  impl.exchangeStream(prev);

  impl.synchronizeStream(stream_a);
  impl.synchronizeStream(stream_b);
  EXPECT_TRUE(impl.queryEvent(event));

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
