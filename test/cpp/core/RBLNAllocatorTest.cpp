#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHooksInterface.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>

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
    // The allocator reports stats only for a device this process has allocated on.
    {
      const auto primer = c10::GetAllocator(c10::kPrivateUse1)->allocate(1);
    }
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

  // The caching allocator's torch.cuda.memory_stats()-style map of the initial device.
  std::map<std::string, uint64_t> Stats() const {
    return c10::rbln::memory_stats(c10::Device(c10::kPrivateUse1, initial_device_index_));
  }

  static uint64_t Stat(const std::map<std::string, uint64_t>& stats, const std::string& key) {
    const auto it = stats.find(key);
    EXPECT_NE(it, stats.end()) << "no stat " << key;
    return it == stats.end() ? 0 : it->second;
  }

  const c10::DeviceIndex initial_device_index_ = 0;
  const size_t size_0b_ = 0;
  const size_t size_1gib_ = 1ULL << 30;
  // Large blocks are whole multiples of 2 MiB, so this request takes a 4 MiB block.
  static constexpr size_t kLargeRequest = size_t{3} << 20;
  static constexpr size_t kLargeBlock = size_t{4} << 20;
};

TEST_F(RBLNAllocatorTest, Allocate) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);

  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    c10::rbln::set_device_index(device_index);
    const auto current_device_index = c10::rbln::get_device_index();
    EXPECT_EQ(current_device_index, device_index);

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
  auto* allocator = GetDeviceAllocator();
  const auto before = Stats();

  // Every allocation is device memory, so one larger than the device fails at once: once
  // with the cache as it was, and once more after emptying it.
  const auto total = allocator->getMemoryInfo(initial_device_index_).second;
  EXPECT_THROW(allocator->allocate(total + size_1gib_), c10::Error);

  const auto after = Stats();
  EXPECT_EQ(Stat(after, "num_ooms") - Stat(before, "num_ooms"), 1u);
  EXPECT_EQ(Stat(after, "num_alloc_retries") - Stat(before, "num_alloc_retries"), 1u);
}

// A freed block goes back to the cache, and the next request of its size on the same
// stream gets it again without asking the device for more memory.
TEST_F(RBLNAllocatorTest, FreedBlockIsReusedOnItsStream) {
  auto* allocator = GetDeviceAllocator();
  allocator->emptyCache();

  void* first = nullptr;
  {
    const auto data = allocator->allocate(kLargeRequest);
    ASSERT_NE(data.get(), nullptr);
    first = data.get();
  }
  const auto before = Stats();
  const auto again = allocator->allocate(kLargeRequest);
  EXPECT_EQ(again.get(), first);
  EXPECT_EQ(Stat(Stats(), "num_device_alloc"), Stat(before, "num_device_alloc"));
}

// A block freed on one stream is not handed to another stream's request, which could
// otherwise overwrite it before the first stream is done with it.
TEST_F(RBLNAllocatorTest, FreedBlockIsNotReusedOnAnotherStream) {
  auto* allocator = GetDeviceAllocator();
  allocator->emptyCache();

  void* first = nullptr;
  {
    const auto data = allocator->allocate(kLargeRequest);
    ASSERT_NE(data.get(), nullptr);
    first = data.get();
  }
  const auto default_stream = c10::rbln::get_current_stream(initial_device_index_);
  c10::rbln::set_current_stream(c10::rbln::get_stream_from_pool(initial_device_index_));
  {
    const auto other = allocator->allocate(kLargeRequest);
    EXPECT_NE(other.get(), first);
  }
  c10::rbln::set_current_stream(default_stream);
}

// A block another stream used is reused once that stream has reached the point of the free.
TEST_F(RBLNAllocatorTest, RecordedBlockIsReusedOnceTheStreamReachesTheFree) {
  auto* allocator = GetDeviceAllocator();
  allocator->emptyCache();
  const auto pool_stream = c10::rbln::get_stream_from_pool(initial_device_index_);

  void* first = nullptr;
  {
    const auto data = allocator->allocate(kLargeRequest);
    ASSERT_NE(data.get(), nullptr);
    first = data.get();
    allocator->recordStream(data, pool_stream);
  }
  c10::rbln::synchronize_stream(pool_stream);
  const auto again = allocator->allocate(kLargeRequest);
  EXPECT_EQ(again.get(), first);
}

TEST_F(RBLNAllocatorTest, StatsCountBlocksAndSegments) {
  auto* allocator = GetDeviceAllocator();
  allocator->emptyCache();
  const auto before = Stats();
  const auto delta = [&before](const std::map<std::string, uint64_t>& now, const std::string& key) {
    return Stat(now, key) - Stat(before, key);
  };

  {
    const auto data = allocator->allocate(kLargeRequest);
    ASSERT_NE(data.get(), nullptr);
    const auto during = Stats();
    EXPECT_EQ(delta(during, "allocated_bytes.all.current"), kLargeBlock);
    EXPECT_EQ(delta(during, "allocated_bytes.large_pool.current"), kLargeBlock);
    EXPECT_EQ(delta(during, "requested_bytes.all.current"), kLargeRequest);
    EXPECT_EQ(delta(during, "reserved_bytes.all.current"), kLargeBlock);
    EXPECT_EQ(delta(during, "allocation.all.current"), 1u);
    EXPECT_EQ(delta(during, "segment.all.current"), 1u);
    EXPECT_GE(Stat(during, "allocated_bytes.all.peak"), Stat(during, "allocated_bytes.all.current"));
  }

  // Freed, the block stays reserved in the cache.
  const auto after_free = Stats();
  EXPECT_EQ(delta(after_free, "allocated_bytes.all.current"), 0u);
  EXPECT_EQ(delta(after_free, "reserved_bytes.all.current"), kLargeBlock);
  EXPECT_EQ(delta(after_free, "allocation.all.freed"), 1u);
}

// empty_cache hands back every segment with no live block in it, and only those.
TEST_F(RBLNAllocatorTest, EmptyCacheReleasesWholeFreeSegments) {
  auto* allocator = GetDeviceAllocator();
  allocator->emptyCache();
  const auto base = Stat(Stats(), "reserved_bytes.all.current");

  {
    const auto freed = allocator->allocate(kLargeRequest);
  }
  EXPECT_EQ(Stat(Stats(), "reserved_bytes.all.current"), base + kLargeBlock);
  allocator->emptyCache();
  EXPECT_EQ(Stat(Stats(), "reserved_bytes.all.current"), base);

  // Two small blocks share a segment; freeing one leaves the segment in use.
  const auto live = allocator->allocate(1024);
  {
    const auto freed = allocator->allocate(1024);
  }
  const auto reserved = Stat(Stats(), "reserved_bytes.all.current");
  EXPECT_GT(reserved, base);
  allocator->emptyCache();
  EXPECT_EQ(Stat(Stats(), "reserved_bytes.all.current"), reserved);
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

// Regression guard for the device-less empty_cache() "span all devices" contract
// (CUDA/XPU parity): emptyCache() must release *every* initialized device, not just the
// current one. Device 0 is left non-current with a cached (freed-but-reserved) block, so a
// current-device-only regression leaves that block intact — asserted released here. The
// selection seam is additionally checked via initialized_device_indices().
TEST_F(RBLNAllocatorTest, EmptyCacheSpansNonCurrentInitializedDevice) {
  if (c10::rbln::get_device_count() < 2) {
    GTEST_SKIP() << "needs >= 2 devices to exercise a non-current device";
  }
  constexpr size_t kAggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  auto* allocator = GetDeviceAllocator();

  // Build a cached block on device 0, then measure its reserved bytes while still current.
  c10::rbln::set_device_index(0);
  {
    const auto d0 = allocator->allocate(32ULL << 20);
    EXPECT_NE(d0.get(), nullptr);
  }
  const auto reserved_before = allocator->getDeviceStats(0).reserved_bytes[kAggregate].current;
  ASSERT_GT(reserved_before, 0) << "the allocation reserved no device memory on device 0";

  // Make device 1 current so device 0 is the non-current initialized device.
  c10::rbln::set_device_index(1);
  {
    const auto d1 = allocator->allocate(1024);
    EXPECT_NE(d1.get(), nullptr);
  }

  EXPECT_NO_THROW(allocator->emptyCache());

  // Device 0 (non-current) must have been released too, not just the current device.
  EXPECT_LT(allocator->getDeviceStats(0).reserved_bytes[kAggregate].current, reserved_before)
      << "emptyCache() left non-current device 0 reserved (current-device-only regression)";

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
  EXPECT_TRUE(contains(1)) << "initialized_device_indices() dropped non-current device 1";
}

// recordStream of an empty DataPtr has no block to record and must be a safe no-op.
TEST_F(RBLNAllocatorTest, RecordStreamOfNullIsNoOp) {
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

TEST_F(RBLNAllocatorTest, GetMemoryInfo) {
  auto* device_allocator = GetDeviceAllocator();
  const auto [free_bytes, total_bytes] = device_allocator->getMemoryInfo(initial_device_index_);
  EXPECT_GT(total_bytes, 0u);
  EXPECT_LE(free_bytes, total_bytes);

  // The device-wide figure is the sum of its chiplets, NPU by NPU.
  const auto per_chiplet = c10::rbln::mem_get_info_per_chiplet(c10::Device(c10::kPrivateUse1, initial_device_index_));
  uint64_t npu_total = 0;
  uint64_t chiplet_total = 0;
  for (const auto& [key, value] : per_chiplet) {
    if (key.find(".chiplet.") != std::string::npos) {
      if (key.size() > 6 && key.compare(key.size() - 6, 6, ".total") == 0) {
        chiplet_total += value;
      }
    } else if (key.size() > 6 && key.compare(key.size() - 6, 6, ".total") == 0) {
      npu_total += value;
    }
  }
  EXPECT_EQ(npu_total, total_bytes);
  EXPECT_EQ(chiplet_total, total_bytes);
}

TEST_F(RBLNAllocatorTest, GetMemoryInfoInvalidIndex) {
  auto* device_allocator = GetDeviceAllocator();
  const auto device_count = c10::rbln::get_device_count();

  EXPECT_THROW(device_allocator->getMemoryInfo(-1), c10::Error);
  EXPECT_THROW(device_allocator->getMemoryInfo(device_count), c10::Error);
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
