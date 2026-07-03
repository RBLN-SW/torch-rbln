// NOTE: excluded from the build (see test/cpp/CMakeLists.txt). These tests
// assert the old torch host-backing contract (host malloc, interior-pointer
// registry, get_memory_info-unavailable, direct pointer deref) that no longer
// exists now that dummy allocations route through rebel's v-memory. They fail
// at runtime and need a rewrite against the vmem contract; until then the dummy
// behavior is covered by test/rbln/test_dummy_device.py.
//
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
#include <optional>
#include <vector>

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
  // Host pointer is the device pointer itself; borrow_id is non-zero per the
  // BorrowedHostPtr contract (return_borrowed no-ops in dummy mode).
  EXPECT_EQ(borrow.host_ptr, reinterpret_cast<uintptr_t>(dev));
  EXPECT_NE(borrow.borrow_id, 0U);
  ASSERT_NO_THROW(c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/true));

  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, CopyAndBorrowRejectOutOfBounds) {
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);
  int32_t host[8] = {};

  // A transfer larger than the allocation must throw, not OOB-access host memory.
  EXPECT_THROW(c10::rbln::memcpy_h2v(dev, host, kBytes * 2), c10::Error);
  EXPECT_THROW(c10::rbln::memcpy_v2h(host, dev, kBytes * 2), c10::Error);
  // A borrow beyond the allocation must throw.
  EXPECT_THROW(c10::rbln::borrow_host_ptr(dev, kBytes * 2), c10::Error);
  // An unknown device pointer must throw.
  int32_t stack = 0;
  EXPECT_THROW(c10::rbln::memcpy_v2h(host, &stack, sizeof(int32_t)), c10::Error);

  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, SynchronizeIsNoOp) {
  EXPECT_NO_THROW(c10::rbln::synchronize(/*device_index=*/0));
}

TEST(RBLNDummyDeviceTest, AsyncTransfersOnHost) {
  // The async transfers are synchronous host memmoves in dummy mode (no handle,
  // no runtime); same data and bounds contract as the sync variants.
  constexpr size_t kBytes = 4 * sizeof(float);
  void* a = c10::rbln::malloc(/*device_index=*/0, kBytes);
  void* b = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  const float src[4] = {1.0F, -2.0F, 3.5F, 42.0F};
  float dst[4] = {0, 0, 0, 0};
  ASSERT_NO_THROW(c10::rbln::memcpy_h2v_async(a, src, kBytes));
  ASSERT_NO_THROW(c10::rbln::memcpy_v2v_async(b, a, kBytes));
  ASSERT_NO_THROW(c10::rbln::memcpy_v2h_async(dst, b, kBytes));
  EXPECT_EQ(0, std::memcmp(src, dst, kBytes));

  EXPECT_THROW(c10::rbln::memcpy_h2v_async(a, src, kBytes * 2), c10::Error);

  c10::rbln::free(a);
  c10::rbln::free(b);
}

TEST(RBLNDummyDeviceTest, V2VMultiOnHost) {
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  auto* src = static_cast<int32_t*>(c10::rbln::malloc(/*device_index=*/0, kBytes));
  auto* d0 = static_cast<int32_t*>(c10::rbln::malloc(/*device_index=*/0, kBytes));
  auto* d1 = static_cast<int32_t*>(c10::rbln::malloc(/*device_index=*/0, kBytes));
  ASSERT_NE(src, nullptr);
  ASSERT_NE(d0, nullptr);
  ASSERT_NE(d1, nullptr);
  for (int32_t i = 0; i < 4; ++i) {
    src[i] = i + 1;
  }

  EXPECT_NO_THROW(c10::rbln::memcpy_v2v_multi({})); // empty is a no-op
  std::vector<c10::rbln::V2VCopyOp> copies = {{d0, src, kBytes}, {d1, src, kBytes}};
  ASSERT_NO_THROW(c10::rbln::memcpy_v2v_multi(copies));
  EXPECT_EQ(0, std::memcmp(src, d0, kBytes));
  EXPECT_EQ(0, std::memcmp(src, d1, kBytes));

  // An out-of-bounds entry is rejected.
  std::vector<c10::rbln::V2VCopyOp> bad = {{d0, src, kBytes * 2}};
  EXPECT_THROW(c10::rbln::memcpy_v2v_multi(bad), c10::Error);

  c10::rbln::free(src);
  c10::rbln::free(d0);
  c10::rbln::free(d1);
}

TEST(RBLNDummyDeviceTest, TryBorrowAndAcquireAreIdentity) {
  constexpr size_t kBytes = 2 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);
  const auto addr = reinterpret_cast<uintptr_t>(dev);

  // try_borrow / acquire / try_acquire are all identity host views with a
  // non-zero borrow id in dummy mode.
  const auto tb = c10::rbln::try_borrow_host_ptr(dev, kBytes);
  ASSERT_TRUE(tb.has_value());
  EXPECT_EQ(tb->host_ptr, addr);
  EXPECT_NE(tb->borrow_id, 0U);
  c10::rbln::return_borrowed(tb->borrow_id, /*updated=*/false);

  const auto acq = c10::rbln::acquire_host_ptr_for_overwrite(dev, kBytes);
  EXPECT_EQ(acq.host_ptr, addr);
  EXPECT_NE(acq.borrow_id, 0U);
  c10::rbln::return_borrowed(acq.borrow_id, /*updated=*/true);

  const auto ta = c10::rbln::try_acquire_host_ptr_for_overwrite(dev, kBytes);
  ASSERT_TRUE(ta.has_value());
  EXPECT_EQ(ta->host_ptr, addr);

  // try_* return nullopt (no throw) for an out-of-range request.
  EXPECT_FALSE(c10::rbln::try_borrow_host_ptr(dev, kBytes * 2).has_value());
  EXPECT_FALSE(c10::rbln::try_acquire_host_ptr_for_overwrite(dev, kBytes * 2).has_value());

  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, SetDeviceLayoutLikeIsNoOp) {
  // No device-side layout in dummy mode: same-device live allocations no-op;
  // cross-device or stale pointers are rejected (no runtime call).
  void* a = c10::rbln::malloc(/*device_index=*/0, 64);
  void* b = c10::rbln::malloc(/*device_index=*/0, 64);
  void* c = c10::rbln::malloc(/*device_index=*/1, 64);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  EXPECT_NO_THROW(c10::rbln::set_device_layout_like(a, b));
  EXPECT_THROW(c10::rbln::set_device_layout_like(a, c), c10::Error);
  c10::rbln::free(a);
  c10::rbln::free(b);
  c10::rbln::free(c);
  EXPECT_THROW(c10::rbln::set_device_layout_like(a, a), c10::Error); // stale
}

TEST(RBLNDummyDeviceTest, SetFileOffloadingIsNoOp) {
  // File offloading is a runtime feature; enable/disable are no-ops in dummy mode.
  EXPECT_NO_THROW(c10::rbln::set_file_offloading_enabled(true));
  EXPECT_NO_THROW(c10::rbln::set_file_offloading_enabled(false));
}
