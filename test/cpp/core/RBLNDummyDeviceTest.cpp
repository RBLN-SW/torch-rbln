// Tests for the host-backed dummy device contract (RBLN_DUMMY_DEVICE=1).
//
// Dummy mode is an explicit opt-in that lets torch-rbln construct device
// tensors and run memory transfers on host memory so a model can be
// traced/compiled on a host with no NPU. It is forced regardless of physical
// NPU presence, so this test runs in any configuration.
//
// The env MUST be set before the DeviceMappingManager singleton initializes
// (it reads RBLN_DUMMY_DEVICE once at init). A file-scope static initializer
// runs before main(), and thus before any c10::rbln call in a test body, so
// the singleton observes the flag.
//
// Contract with RBLN_DUMMY_DEVICE=1:
//   - device_count() reports N >= 1 logical device(s) (vs. the no-device path,
//     which reports 0); physical_device_count() reports 0 without querying the
//     runtime (no NPU);
//   - allocation, h2v/v2h/v2v transfers, and host-pointer borrows succeed on
//     host memory rather than failing at the point of use;
//   - get_torch_device_id() resolves a pointer (incl. interior/view) to its
//     owning device; free() rejects stale/double frees;
//   - synchronize is a no-op (host transfers are synchronous).
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace {
// Force dummy mode before main(), i.e. before the singleton initializes. Two
// logical devices (RBLN_DEVICE_MAP) so the pointer->device registry is testable.
[[maybe_unused]] const int kSetDummyEnv = []() {
  setenv("RBLN_DUMMY_DEVICE", "1", /*overwrite=*/1);
  setenv("RBLN_DEVICE_MAP", "[0],[1]", /*overwrite=*/1);
  return 0;
}();
} // namespace

TEST(RBLNDummyDeviceTest, ReportsLogicalButNoPhysicalDevice) {
  EXPECT_TRUE(c10::rbln::is_dummy_device());
  EXPECT_EQ(c10::rbln::get_device_count(), 2); // RBLN_DEVICE_MAP group count
  // physical count must not query the runtime (no NPU); reports 0.
  EXPECT_EQ(c10::rbln::get_physical_device_count(), 0);
}

TEST(RBLNDummyDeviceTest, AllocateAndTransferOnHost) {
  // Allocation succeeds (host-backed) instead of throwing as it would with no
  // logical device.
  constexpr size_t kBytes = 4 * sizeof(float);
  void* dev = nullptr;
  ASSERT_NO_THROW(dev = c10::rbln::malloc(/*device_index=*/0, kBytes));
  ASSERT_NE(dev, nullptr);

  const float src[4] = {1.0F, -2.0F, 3.5F, 42.0F};
  float dst[4] = {0, 0, 0, 0};

  ASSERT_NO_THROW(c10::rbln::memcpy_h2v(dev, src, kBytes));
  ASSERT_NO_THROW(c10::rbln::memcpy_v2h(dst, dev, kBytes));
  EXPECT_EQ(0, std::memcmp(src, dst, kBytes));

  // Same-"device" v2v is a plain host memmove.
  void* dev2 = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev2, nullptr);
  ASSERT_NO_THROW(c10::rbln::memcpy_v2v(dev2, dev, kBytes));
  float dst2[4] = {0, 0, 0, 0};
  c10::rbln::memcpy_v2h(dst2, dev2, kBytes);
  EXPECT_EQ(0, std::memcmp(src, dst2, kBytes));

  c10::rbln::free(dev);
  c10::rbln::free(dev2);
}

TEST(RBLNDummyDeviceTest, GetTorchDeviceIdReportsOwningDevice) {
  // The owning device is the one a pointer was allocated on, not the current
  // device — and an interior/view pointer resolves to the same allocation.
  auto* d0 = static_cast<char*>(c10::rbln::malloc(/*device_index=*/0, 64));
  auto* d1 = static_cast<char*>(c10::rbln::malloc(/*device_index=*/1, 64));
  ASSERT_NE(d0, nullptr);
  ASSERT_NE(d1, nullptr);
  EXPECT_EQ(c10::rbln::get_torch_device_id(d0), 0);
  EXPECT_EQ(c10::rbln::get_torch_device_id(d1), 1);
  EXPECT_EQ(c10::rbln::get_torch_device_id(d1 + 16), 1); // interior/view pointer
  EXPECT_EQ(c10::rbln::get_torch_device_id(d1 + 63), 1); // last byte
  c10::rbln::free(d0);
  c10::rbln::free(d1);
  // After free the pointer is unknown -> throws (parity with the real lookup).
  EXPECT_THROW(c10::rbln::get_torch_device_id(d0), c10::Error);
}

TEST(RBLNDummyDeviceTest, FreeRejectsStaleAndDoubleFree) {
  void* p = c10::rbln::malloc(/*device_index=*/0, 32);
  ASSERT_NE(p, nullptr);
  ASSERT_NO_THROW(c10::rbln::free(p)); // first free succeeds
  EXPECT_THROW(c10::rbln::free(p), c10::Error); // double free rejected (no abort)
  int stack_var = 0;
  EXPECT_THROW(c10::rbln::free(&stack_var), c10::Error); // unknown pointer rejected
}

TEST(RBLNDummyDeviceTest, GetMemoryInfoUnavailable) {
  void* p = c10::rbln::malloc(/*device_index=*/0, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_THROW(c10::rbln::get_memory_info(p), c10::Error);
  c10::rbln::free(p);
}

TEST(RBLNDummyDeviceTest, V2VHandlesOverlap) {
  // copy_ can alias; v2v must use memmove, not memcpy (overlap is otherwise UB).
  constexpr size_t kN = 8;
  auto* buf = static_cast<int32_t*>(c10::rbln::malloc(/*device_index=*/0, kN * sizeof(int32_t)));
  ASSERT_NE(buf, nullptr);
  for (int32_t i = 0; i < static_cast<int32_t>(kN); ++i) {
    buf[i] = i;
  }
  // Shift right by one within the same buffer (dst overlaps src).
  c10::rbln::memcpy_v2v(buf + 1, buf, (kN - 1) * sizeof(int32_t));
  const int32_t expected[kN] = {0, 0, 1, 2, 3, 4, 5, 6};
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_EQ(buf[i], expected[i]) << "at index " << i;
  }
  // Self-copy must be a safe no-op.
  EXPECT_NO_THROW(c10::rbln::memcpy_v2v(buf, buf, kN * sizeof(int32_t)));
  c10::rbln::free(buf);
}

TEST(RBLNDummyDeviceTest, BorrowHostPtrIsIdentity) {
  constexpr size_t kBytes = 2 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);

  const auto borrow = c10::rbln::borrow_host_ptr(dev, kBytes);
  // The host pointer is the device pointer itself; borrow_id is the no-op
  // sentinel.
  EXPECT_EQ(borrow.host_ptr, reinterpret_cast<uintptr_t>(dev));
  EXPECT_EQ(borrow.borrow_id, 0U);
  ASSERT_NO_THROW(c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/true));

  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, SynchronizeIsNoOp) {
  EXPECT_NO_THROW(c10::rbln::synchronize(/*device_index=*/0));
}
