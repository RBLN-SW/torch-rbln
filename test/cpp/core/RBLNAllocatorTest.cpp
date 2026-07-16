#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHooksInterface.h>
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
  // CUDA parity: initialized() reflects per-process allocator state, so it is true only
  // after this process has actually allocated (a device/mapping existing is not enough).
  // The uninitialized-process case (false before any allocation) is covered in a fresh
  // subprocess by test/rbln/test_runtime_unavailable.py.
  auto* device_allocator = GetDeviceAllocator();
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_TRUE(device_allocator->initialized());
}

// The per-process context flag backs initialized() and hasPrimaryContext(): a bit is
// set on the first successful allocation on a device. (The uninitialized case — false
// before any allocation — needs a fresh process; see test_runtime_unavailable.py.)
TEST_F(RBLNAllocatorTest, DeviceContextTracksAllocation) {
  EXPECT_FALSE(c10::rbln::device_context_initialized(-1)); // negative → always false, nothrow

  const auto idx = c10::rbln::get_device_index();
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_TRUE(c10::rbln::device_context_initialized(idx));
  EXPECT_TRUE(c10::rbln::any_device_context_initialized());

  // hasPrimaryContext() is per-device and mirrors the flag (CUDA parity).
  auto* hooks = c10::rbln::get_rbln_hooks();
  EXPECT_TRUE(hooks->hasPrimaryContext(idx));
  EXPECT_FALSE(hooks->hasPrimaryContext(-1));
}

// The tracker spans the full valid DeviceIndex range across both mask words (63 = word 0,
// 64/126 = word 1); the max value (127) is out of range, and negatives are false. This
// guards against the earlier single-word tracker that silently dropped indices 64+. Marks
// are process-global/sticky, so this only touches high, otherwise-unused indices.
TEST_F(RBLNAllocatorTest, DeviceContextTrackerBounds) {
  EXPECT_FALSE(c10::rbln::device_context_initialized(-1));
  for (const c10::DeviceIndex i : {c10::DeviceIndex{63}, c10::DeviceIndex{64}, c10::DeviceIndex{126}}) {
    EXPECT_FALSE(c10::rbln::device_context_initialized(i)); // unset before mark
    c10::rbln::mark_device_context_initialized(i);
    EXPECT_TRUE(c10::rbln::device_context_initialized(i));
  }
  // 127 == numeric_limits<DeviceIndex>::max() is never a valid device index → ignored.
  c10::rbln::mark_device_context_initialized(127);
  EXPECT_FALSE(c10::rbln::device_context_initialized(127));
}

// A shutting-down runtime has no usable context, so hasPrimaryContext() must report
// false even for a device this process allocated on (folds in the liveness check).
TEST_F(RBLNAllocatorTest, HasPrimaryContextFalseDuringShutdown) {
  const auto idx = c10::rbln::get_device_index();
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  auto* hooks = c10::rbln::get_rbln_hooks();
  EXPECT_TRUE(hooks->hasPrimaryContext(idx));

  c10::rbln::set_runtime_shutting_down(true);
  EXPECT_FALSE(hooks->hasPrimaryContext(idx));
  c10::rbln::set_runtime_shutting_down(false); // restore (process-global flag)
  EXPECT_TRUE(hooks->hasPrimaryContext(idx));
}

TEST_F(RBLNAllocatorTest, EmptyCache) {
  auto* device_allocator = GetDeviceAllocator();
  // Allocate first so this exercises a real flush, not the uninitialized no-op path
  // (keeps the test independent of allocations done by earlier tests).
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_NO_THROW(device_allocator->emptyCache());
}

// Regression guard for the device-less empty_cache() "span all devices" contract:
// its device selection (initialized_device_indices) must cover *every* initialized
// device, not just the current one (a current-device-only regression drops the rest).
// The per-device release itself is unobservable here — the runtime exposes memory
// stats for node 0 only — so we assert the selection includes a non-current device.
TEST_F(RBLNAllocatorTest, EmptyCacheSpansNonCurrentInitializedDevice) {
  if (c10::rbln::get_device_count() < 2) {
    GTEST_SKIP() << "needs >= 2 devices to exercise a non-current device";
  }
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  // Initialize a non-current device (index 1), then make device 0 current again.
  c10::rbln::set_device_index(1);
  {
    const auto d1 = allocator->allocate(1024);
    EXPECT_NE(d1.get(), nullptr);
  }
  c10::rbln::set_device_index(0);
  {
    const auto d0 = allocator->allocate(1024);
    EXPECT_NE(d0.get(), nullptr);
  }

  const auto indices = c10::rbln::initialized_device_indices();
  const auto contains = [&](c10::DeviceIndex i) {
    for (const auto x : indices) {
      if (x == i) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(contains(0));
  EXPECT_TRUE(contains(1)) << "empty_cache() would skip non-current initialized device 1";
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
  // Allocate first so this exercises a real reset, not the uninitialized no-op path.
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_NO_THROW(device_allocator->resetAccumulatedStats(initial_device_index_));
}

TEST_F(RBLNAllocatorTest, ResetAccumulatedStatsInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();
  // Once the allocator is in use, an invalid device index is a real error (CUDA parity).
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);

  EXPECT_THROW(device_allocator->resetAccumulatedStats(-1), c10::Error);
  EXPECT_THROW(device_allocator->resetAccumulatedStats(device_count), c10::Error);
}

TEST_F(RBLNAllocatorTest, ResetPeakStats) {
  auto* device_allocator = GetDeviceAllocator();
  // Allocate first so this exercises a real reset, not the uninitialized no-op path.
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_NO_THROW(device_allocator->resetPeakStats(initial_device_index_));
}

TEST_F(RBLNAllocatorTest, ResetPeakStatsInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();
  // Once the allocator is in use, an invalid device index is a real error (CUDA parity).
  const auto data = c10::GetAllocator(c10::kPrivateUse1)->allocate(1024);
  EXPECT_NE(data.get(), nullptr);

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
