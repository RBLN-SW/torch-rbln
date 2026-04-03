#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

class RBLNAllocatorTest : public ::testing::Test {
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

  // Returns the registered allocator cast to DeviceAllocator.
  // Asserts (not just expects) so callers can assume the result is non-null.
  static c10::DeviceAllocator* GetDeviceAllocator() {
    auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
    EXPECT_NE(allocator, nullptr);
    auto* device_allocator = dynamic_cast<c10::DeviceAllocator*>(allocator);
    EXPECT_NE(device_allocator, nullptr);
    return device_allocator;
  }

  const c10::DeviceIndex initial_device_index_ = 0;
  const size_t size_0b_ = 0;
  const size_t size_1gib_ = 1ULL << 30;
  const size_t size_16gib_ = 1ULL << 34; // The memory capacity of ATOM is 15.7 GiB.
};

TEST_F(RBLNAllocatorTest, Allocate) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);

  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    c10::rbln::set_device_index(device_index);
    const auto current_device_index = c10::rbln::get_device_index();
    EXPECT_EQ(current_device_index, device_index);

    {
      const auto data = allocator->allocate(15 * size_1gib_);
      EXPECT_TRUE(data.get() != nullptr);
      const auto data_device = data.device();
      EXPECT_TRUE(data_device.is_privateuseone());
      EXPECT_EQ(data_device.index(), current_device_index);

      // If memory is allocated lazily, the following assertion may fail because CPU memory is allocated instead of NPU
      // memory.
      if (c10::rbln::is_eager_malloc()) {
        EXPECT_THROW(allocator->allocate(size_1gib_), c10::Error);
      }
    }
    const auto data = allocator->allocate(size_1gib_);
    EXPECT_TRUE(data.get() != nullptr);
    const auto data_device = data.device();
    EXPECT_TRUE(data_device.is_privateuseone());
    EXPECT_EQ(data_device.index(), current_device_index);
  }
}

TEST_F(RBLNAllocatorTest, AllocateZeroBytes) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);

  EXPECT_EQ(allocator->allocate(size_0b_), nullptr);
}

TEST_F(RBLNAllocatorTest, AllocateInvalidSize) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);

  // If memory is allocated lazily, the following assertion may fail because CPU memory is allocated instead of NPU
  // memory.
  if (c10::rbln::is_eager_malloc()) {
    EXPECT_THROW(allocator->allocate(size_16gib_), c10::Error);
  }
}

// Verify the registered allocator is a DeviceAllocator (the prerequisite for all
// torch.accelerator memory APIs).
TEST_F(RBLNAllocatorTest, IsDeviceAllocator) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  EXPECT_NE(dynamic_cast<c10::DeviceAllocator*>(allocator), nullptr);
}

TEST_F(RBLNAllocatorTest, Initialized) {
  auto* device_allocator = GetDeviceAllocator();
  EXPECT_TRUE(device_allocator->initialized());
}

TEST_F(RBLNAllocatorTest, EmptyCache) {
  auto* device_allocator = GetDeviceAllocator();
  EXPECT_NO_THROW(device_allocator->emptyCache());
}

// recordStream must be a safe no-op — RBLN has no stream-based async execution.
TEST_F(RBLNAllocatorTest, RecordStreamIsNoOp) {
  auto* device_allocator = GetDeviceAllocator();
  const auto stream = c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::kPrivateUse1, initial_device_index_));
  EXPECT_NO_THROW(device_allocator->recordStream(c10::DataPtr{}, stream));
}

TEST_F(RBLNAllocatorTest, GetDeviceStats) {
  auto* device_allocator = GetDeviceAllocator();

  // Query stats for the device initialised in SetUp (device 0).
  // Other devices may not have an active runtime context, so querying them
  // would raise INIT_INVALID_ARGUMENT.
  c10::CachingDeviceAllocator::DeviceStats stats{};
  ASSERT_NO_THROW(stats = device_allocator->getDeviceStats(initial_device_index_));

  // All byte counters must be non-negative.
  constexpr size_t kAggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  EXPECT_GE(stats.allocated_bytes[kAggregate].current, 0);
  EXPECT_GE(stats.allocated_bytes[kAggregate].peak, 0);
  EXPECT_GE(stats.reserved_bytes[kAggregate].current, 0);
  EXPECT_GE(stats.reserved_bytes[kAggregate].peak, 0);
  EXPECT_GE(stats.active_bytes[kAggregate].current, 0);
  EXPECT_GE(stats.active_bytes[kAggregate].peak, 0);
  EXPECT_GE(stats.inactive_split_bytes[kAggregate].current, 0);
  EXPECT_GE(stats.inactive_split_bytes[kAggregate].peak, 0);

  // Scalar counters must be non-negative.
  EXPECT_GE(stats.num_alloc_retries, 0);
  EXPECT_GE(stats.num_ooms, 0);
  EXPECT_GE(stats.num_device_alloc, 0);
  EXPECT_GE(stats.num_device_free, 0);

  // Peak must be at least as large as current.
  EXPECT_GE(stats.allocated_bytes[kAggregate].peak, stats.allocated_bytes[kAggregate].current);
  EXPECT_GE(stats.reserved_bytes[kAggregate].peak, stats.reserved_bytes[kAggregate].current);
  EXPECT_GE(stats.active_bytes[kAggregate].peak, stats.active_bytes[kAggregate].current);
}

TEST_F(RBLNAllocatorTest, GetDeviceStatsInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();

  // Negative index should throw.
  EXPECT_THROW(device_allocator->getDeviceStats(-1), c10::Error);
  // Out-of-range index should throw.
  EXPECT_THROW(device_allocator->getDeviceStats(device_count), c10::Error);
}

TEST_F(RBLNAllocatorTest, ResetAccumulatedStats) {
  auto* device_allocator = GetDeviceAllocator();
  EXPECT_NO_THROW(device_allocator->resetAccumulatedStats(initial_device_index_));
}

TEST_F(RBLNAllocatorTest, ResetAccumulatedStatsInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();

  EXPECT_THROW(device_allocator->resetAccumulatedStats(-1), c10::Error);
  EXPECT_THROW(device_allocator->resetAccumulatedStats(device_count), c10::Error);
}

TEST_F(RBLNAllocatorTest, ResetPeakStats) {
  auto* device_allocator = GetDeviceAllocator();
  EXPECT_NO_THROW(device_allocator->resetPeakStats(initial_device_index_));
}

TEST_F(RBLNAllocatorTest, ResetPeakStatsInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();

  EXPECT_THROW(device_allocator->resetPeakStats(-1), c10::Error);
  EXPECT_THROW(device_allocator->resetPeakStats(device_count), c10::Error);
}

// copy_data with nbytes==0 must be a no-op — no crash, no side effects.
TEST_F(RBLNAllocatorTest, CopyDataZeroBytes) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  char src = 'A';
  char dst = 'B';
  EXPECT_NO_THROW(allocator->copy_data(&dst, &src, 0));
  EXPECT_EQ(dst, 'B');
}

TEST_F(RBLNAllocatorTest, RawDeleterIsNonNull) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  EXPECT_NE(allocator->raw_deleter(), nullptr);
}
