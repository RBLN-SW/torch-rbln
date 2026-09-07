#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNPinnedAllocator.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

class RBLNPinnedAllocatorTest : public ::testing::Test {};

TEST_F(RBLNPinnedAllocatorTest, AllocateRegistersPinnedRange) {
  auto* allocator = c10::rbln::get_pinned_memory_allocator();
  ASSERT_NE(allocator, nullptr);

  constexpr size_t nbytes = 4097; // deliberately unaligned size
  auto data_ptr = allocator->allocate(nbytes);
  auto* base = static_cast<uint8_t*>(data_ptr.get());
  ASSERT_NE(base, nullptr);
  EXPECT_EQ(data_ptr.device().type(), c10::DeviceType::CPU);

  // Base, interior, and last byte resolve as pinned; one-past-end does not.
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(base));
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(base + nbytes / 2));
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(base + nbytes - 1));
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base + nbytes));

  // The memory must be writable end to end.
  std::memset(base, 0xAB, nbytes);
  EXPECT_EQ(base[nbytes - 1], 0xAB);
}

TEST_F(RBLNPinnedAllocatorTest, FreeUnregistersPinnedRange) {
  auto* allocator = c10::rbln::get_pinned_memory_allocator();
  void* base = nullptr;
  {
    auto data_ptr = allocator->allocate(1024);
    base = data_ptr.get();
    EXPECT_TRUE(c10::rbln::is_pinned_ptr(base));
  }
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base));
}

TEST_F(RBLNPinnedAllocatorTest, NullAndForeignPointersAreNotPinned) {
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(nullptr));
  int local = 0;
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(&local));
}

TEST_F(RBLNPinnedAllocatorTest, ZeroByteAllocationIsNull) {
  auto data_ptr = c10::rbln::get_pinned_memory_allocator()->allocate(0);
  EXPECT_EQ(data_ptr.get(), nullptr);
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(data_ptr.get()));
}

// --- Runtime registration (needs a device) -------------------------------------------

class RBLNPinnedAllocatorRegisterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (c10::rbln::get_device_count_nothrow() < 1) {
      GTEST_SKIP() << "no RBLN device";
    }
    // Initializes device 0's context; registration needs a live one.
    device_mem_ = c10::rbln::malloc(0, 4096);
    ASSERT_NE(device_mem_, nullptr);
    if (!c10::rbln::pinned_ptr_registered_on(probe(), 0)) {
      // Either the runtime has no host registration (UMD < 3.5 / flag off) or it failed;
      // both are the documented best-effort outcome, nothing to assert against.
      GTEST_SKIP() << "host-memory registration unavailable on this runtime";
    }
  }
  void TearDown() override {
    if (device_mem_ != nullptr) {
      c10::rbln::free(device_mem_);
    }
  }
  // A pinned allocation made while device 0 is initialized: registered eagerly if the
  // runtime supports it.
  const void* probe() {
    if (probe_.get() == nullptr) {
      probe_ = c10::rbln::get_pinned_memory_allocator()->allocate(4096);
    }
    return probe_.get();
  }
  void* device_mem_ = nullptr;
  c10::DataPtr probe_;
};

TEST_F(RBLNPinnedAllocatorRegisterTest, AllocateRegistersOnInitializedDevice) {
  auto data_ptr = c10::rbln::get_pinned_memory_allocator()->allocate(3 * 4096);
  auto* base = static_cast<const uint8_t*>(data_ptr.get());
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(base, 0));
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(base + 4096 + 17, 0)) << "interior pointers resolve too";
  EXPECT_FALSE(c10::rbln::pinned_ptr_registered_on(base, 999)) << "never registered on a device that does not exist";
  // Idempotent: asking again for a covered device changes nothing and does not throw.
  c10::rbln::ensure_pinned_registered(base, 0);
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(base, 0));
}

TEST_F(RBLNPinnedAllocatorRegisterTest, HugeAllocationIsRegistered) {
  // >= 2 MiB takes the huge-page branch; the registration must cover the rounded size.
  constexpr size_t nbytes = (2u << 20) + 4096;
  auto data_ptr = c10::rbln::get_pinned_memory_allocator()->allocate(nbytes);
  auto* base = static_cast<const uint8_t*>(data_ptr.get());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % (2u << 20), 0u) << "2 MiB aligned";
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(base, 0));
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(base + nbytes - 1, 0));
}

TEST_F(RBLNPinnedAllocatorRegisterTest, FreeUnregisters) {
  const void* base = nullptr;
  {
    auto data_ptr = c10::rbln::get_pinned_memory_allocator()->allocate(4096);
    base = data_ptr.get();
    ASSERT_TRUE(c10::rbln::pinned_ptr_registered_on(base, 0));
  }
  EXPECT_FALSE(c10::rbln::pinned_ptr_registered_on(base, 0));
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base));
}

TEST_F(RBLNPinnedAllocatorRegisterTest, PageableAndNullAreIgnored) {
  int local = 0;
  c10::rbln::ensure_pinned_registered(&local, 0); // must not throw
  c10::rbln::ensure_pinned_registered(nullptr, 0);
  EXPECT_FALSE(c10::rbln::pinned_ptr_registered_on(&local, 0));
  EXPECT_FALSE(c10::rbln::pinned_ptr_registered_on(nullptr, 0));
}

TEST_F(RBLNPinnedAllocatorRegisterTest, RegisteredCopyRoundTrips) {
  constexpr size_t nbytes = 1u << 20;
  auto src = c10::rbln::get_pinned_memory_allocator()->allocate(nbytes);
  auto dst = c10::rbln::get_pinned_memory_allocator()->allocate(nbytes);
  ASSERT_TRUE(c10::rbln::pinned_ptr_registered_on(src.get(), 0));
  auto* s = static_cast<uint8_t*>(src.get());
  auto* d = static_cast<uint8_t*>(dst.get());
  for (size_t i = 0; i < nbytes; ++i) {
    s[i] = static_cast<uint8_t>(i * 7 + 3);
  }
  std::memset(d, 0, nbytes);
  void* dev = c10::rbln::malloc(0, nbytes);
  ASSERT_NE(dev, nullptr);
  c10::rbln::memcpy_h2v(dev, s, nbytes);
  c10::rbln::memcpy_v2h(d, dev, nbytes);
  c10::rbln::free(dev);
  EXPECT_EQ(std::memcmp(s, d, nbytes), 0);
}

// --- register_host_memory: caller-owned memory ---------------------------------------

TEST(RBLNRegisterHostMemoryTest, RegisteredRangeIsPinnedUntilUnregistered) {
  std::vector<uint8_t> buf(3 * 4096);
  auto* base = buf.data();
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base));
  c10::rbln::register_host_memory(base, buf.size());
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(base));
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(base + buf.size() - 1));
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base + buf.size()));
  c10::rbln::unregister_host_memory(base);
  EXPECT_FALSE(c10::rbln::is_pinned_ptr(base));
}

TEST(RBLNRegisterHostMemoryTest, RejectsOverlapNullAndUnknown) {
  std::vector<uint8_t> buf(2 * 4096);
  auto* base = buf.data();
  c10::rbln::register_host_memory(base, buf.size());
  EXPECT_THROW(c10::rbln::register_host_memory(base + 4096, 4096), c10::Error);
  EXPECT_THROW(c10::rbln::register_host_memory(base - 1, 2), c10::Error);
  EXPECT_THROW(c10::rbln::unregister_host_memory(base + 4096), c10::Error);
  c10::rbln::unregister_host_memory(base);
  EXPECT_THROW(c10::rbln::unregister_host_memory(base), c10::Error);
  EXPECT_THROW(c10::rbln::register_host_memory(nullptr, 4096), c10::Error);
  EXPECT_THROW(c10::rbln::register_host_memory(base, 0), c10::Error);
}

TEST(RBLNRegisterHostMemoryTest, AllocatorMemoryIsNotExternal) {
  auto data_ptr = c10::rbln::get_pinned_memory_allocator()->allocate(4096);
  EXPECT_THROW(c10::rbln::register_host_memory(data_ptr.get(), 4096), c10::Error);
  EXPECT_THROW(c10::rbln::unregister_host_memory(data_ptr.get()), c10::Error);
  EXPECT_TRUE(c10::rbln::is_pinned_ptr(data_ptr.get()));
}

TEST_F(RBLNPinnedAllocatorRegisterTest, ExternalRangeIsRegisteredWithTheDevice) {
  std::vector<uint8_t> storage(3 * 4096 + 4095);
  // Page-align the start so the device-VA path applies.
  auto base = reinterpret_cast<uintptr_t>(storage.data());
  base = (base + 4095) & ~uintptr_t{4095};
  auto* data = reinterpret_cast<uint8_t*>(base);
  c10::rbln::register_host_memory(data, 3 * 4096);
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(data, 0));
  EXPECT_TRUE(c10::rbln::pinned_ptr_registered_on(data + 4096, 0));
  c10::rbln::unregister_host_memory(data);
  EXPECT_FALSE(c10::rbln::pinned_ptr_registered_on(data, 0));
}
