#include <c10/rbln/RBLNPinnedAllocator.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

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
