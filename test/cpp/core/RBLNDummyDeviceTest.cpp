// Tests for the dummy device contract (RBLN_DUMMY_DEVICE=1).
//
// Dummy mode presents host-backed logical device(s) with no NPU, so a model can
// be constructed and compiled without hardware. Allocations and transfers are
// backed by rebel's v-memory (a host mirror) through a dummy runtime context —
// torch adds no host-memory shim of its own. The returned device pointer is an
// opaque v-memory handle, NOT host-dereferenceable, so these tests move data
// only through memcpy_h2v / memcpy_v2h / memcpy_v2v. Kernel/graph execution
// still requires a real NPU (guarded elsewhere).
//
// The env MUST be set before the DeviceMappingManager singleton initializes (it
// reads the flags once at init). A file-scope static initializer sets them
// before main(). RBLN_TARGET_SOC is set too: with no NPU to probe, dummy
// registration resolves the target SoC from it.
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
// Force dummy mode before main(). Two logical devices (RBLN_DEVICE_MAP) so the
// owning-device lookup is testable across devices.
[[maybe_unused]] const int kSetDummyEnv = []() {
  setenv("RBLN_DUMMY_DEVICE", "1", /*overwrite=*/1);
  setenv("RBLN_DEVICE_MAP", "[0],[1]", /*overwrite=*/1);
  setenv("RBLN_TARGET_SOC", "RBLN-CA25", /*overwrite=*/1);
  return 0;
}();
} // namespace

TEST(RBLNDummyDeviceTest, ReportsLogicalButNoPhysicalDevice) {
  EXPECT_TRUE(c10::rbln::is_dummy_device());
  EXPECT_EQ(c10::rbln::get_device_count(), 2); // RBLN_DEVICE_MAP group count
  // physical count must not query the runtime (no NPU); reports 0.
  EXPECT_EQ(c10::rbln::get_physical_device_count(), 0);
}

TEST(RBLNDummyDeviceTest, AllocateAndTransferViaVMemory) {
  // Allocation succeeds (v-memory host mirror) instead of throwing as it would
  // with no logical device. Data moves via h2v/v2h/v2v, never a raw deref.
  constexpr size_t kBytes = 4 * sizeof(float);
  void* dev = nullptr;
  ASSERT_NO_THROW(dev = c10::rbln::malloc(/*device_index=*/0, kBytes));
  ASSERT_NE(dev, nullptr);

  const float src[4] = {1.0F, -2.0F, 3.5F, 42.0F};
  float dst[4] = {0, 0, 0, 0};
  ASSERT_NO_THROW(c10::rbln::memcpy_h2v(dev, src, kBytes));
  ASSERT_NO_THROW(c10::rbln::memcpy_v2h(dst, dev, kBytes));
  EXPECT_EQ(0, std::memcmp(src, dst, kBytes));

  // v2v between two device buffers round-trips the same bytes.
  void* dev2 = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev2, nullptr);
  ASSERT_NO_THROW(c10::rbln::memcpy_v2v(dev2, dev, kBytes));
  float dst2[4] = {0, 0, 0, 0};
  c10::rbln::memcpy_v2h(dst2, dev2, kBytes);
  EXPECT_EQ(0, std::memcmp(src, dst2, kBytes));

  c10::rbln::free(dev);
  c10::rbln::free(dev2);
}

TEST(RBLNDummyDeviceTest, GetTorchDeviceIdResolvesOwningDevice) {
  // The owning device is the one a pointer was allocated on, and an interior
  // (view) pointer resolves to the same allocation via the v-memory range map.
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
}

TEST(RBLNDummyDeviceTest, FreeRejectsDoubleAndStale) {
  void* p = c10::rbln::malloc(/*device_index=*/0, 32);
  ASSERT_NE(p, nullptr);
  ASSERT_NO_THROW(c10::rbln::free(p)); // first free succeeds
  EXPECT_THROW(c10::rbln::free(p), c10::Error); // double free rejected (no abort)
  int stack_var = 0;
  EXPECT_THROW(c10::rbln::free(&stack_var), c10::Error); // unknown pointer rejected
}

TEST(RBLNDummyDeviceTest, TransfersAndBorrowsRejectOutOfBounds) {
  // Transfers/borrows past the allocation are rejected, not silently OOB.
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);
  int32_t host[8] = {};
  EXPECT_THROW(c10::rbln::memcpy_h2v(dev, host, kBytes * 2), c10::Error);
  EXPECT_THROW(c10::rbln::memcpy_v2h(host, dev, kBytes * 2), c10::Error);
  EXPECT_THROW(c10::rbln::borrow_host_ptr(dev, kBytes * 2), c10::Error);
  // An unknown device pointer is rejected too.
  int32_t stack = 0;
  EXPECT_THROW(c10::rbln::memcpy_v2h(host, &stack, sizeof(int32_t)), c10::Error);
  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, AsyncTransfersRoundTrip) {
  // Async transfers complete on the dummy context (no handle/runtime); same data
  // and bounds contract as the sync variants.
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
  c10::rbln::synchronize(/*device_index=*/0);
  EXPECT_EQ(0, std::memcmp(src, dst, kBytes));
  EXPECT_THROW(c10::rbln::memcpy_h2v_async(a, src, kBytes * 2), c10::Error);
  c10::rbln::free(a);
  c10::rbln::free(b);
}

TEST(RBLNDummyDeviceTest, BorrowAndAcquireProvideHostAccess) {
  // A borrow yields a valid host pointer (non-zero id) into the v-memory host
  // mirror; writing through it with return_borrowed(updated=true) is reflected on
  // a subsequent v2h read. borrow/try_borrow/acquire/try_acquire all succeed.
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);

  const auto borrow = c10::rbln::borrow_host_ptr(dev, kBytes);
  EXPECT_NE(borrow.borrow_id, 0U);
  ASSERT_NE(borrow.host_ptr, 0U);
  auto* hp = reinterpret_cast<int32_t*>(borrow.host_ptr);
  for (int32_t i = 0; i < 4; ++i) {
    hp[i] = i + 10;
  }
  ASSERT_NO_THROW(c10::rbln::return_borrowed(borrow.borrow_id, /*updated=*/true));
  int32_t out[4] = {};
  c10::rbln::memcpy_v2h(out, dev, kBytes);
  const int32_t expected[4] = {10, 11, 12, 13};
  EXPECT_EQ(0, std::memcmp(expected, out, kBytes));

  const auto tb = c10::rbln::try_borrow_host_ptr(dev, kBytes);
  ASSERT_TRUE(tb.has_value());
  EXPECT_NE(tb->borrow_id, 0U);
  c10::rbln::return_borrowed(tb->borrow_id, /*updated=*/false);

  const auto acq = c10::rbln::acquire_host_ptr_for_overwrite(dev, kBytes);
  EXPECT_NE(acq.borrow_id, 0U);
  c10::rbln::return_borrowed(acq.borrow_id, /*updated=*/true);

  const auto ta = c10::rbln::try_acquire_host_ptr_for_overwrite(dev, kBytes);
  ASSERT_TRUE(ta.has_value());
  c10::rbln::return_borrowed(ta->borrow_id, /*updated=*/false);

  // try_* return nullopt (no throw) for an out-of-range request.
  EXPECT_FALSE(c10::rbln::try_borrow_host_ptr(dev, kBytes * 2).has_value());
  EXPECT_FALSE(c10::rbln::try_acquire_host_ptr_for_overwrite(dev, kBytes * 2).has_value());

  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, V2VMultiTransfers) {
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  void* src = c10::rbln::malloc(/*device_index=*/0, kBytes);
  void* d0 = c10::rbln::malloc(/*device_index=*/0, kBytes);
  void* d1 = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(d0, nullptr);
  ASSERT_NE(d1, nullptr);
  const int32_t values[4] = {1, 2, 3, 4};
  c10::rbln::memcpy_h2v(src, values, kBytes);

  EXPECT_NO_THROW(c10::rbln::memcpy_v2v_multi({})); // empty is a no-op
  std::vector<c10::rbln::V2VCopyOp> copies = {{d0, src, kBytes}, {d1, src, kBytes}};
  ASSERT_NO_THROW(c10::rbln::memcpy_v2v_multi(copies));
  int32_t out0[4] = {}, out1[4] = {};
  c10::rbln::memcpy_v2h(out0, d0, kBytes);
  c10::rbln::memcpy_v2h(out1, d1, kBytes);
  EXPECT_EQ(0, std::memcmp(values, out0, kBytes));
  EXPECT_EQ(0, std::memcmp(values, out1, kBytes));

  c10::rbln::free(src);
  c10::rbln::free(d0);
  c10::rbln::free(d1);
}

TEST(RBLNDummyDeviceTest, SynchronizeIsNoOp) {
  EXPECT_NO_THROW(c10::rbln::synchronize(/*device_index=*/0));
}

TEST(RBLNDummyDeviceTest, SetDeviceLayoutLikeRaisesGracefullyWithoutMaterializedRef) {
  // Not special-cased for dummy: without a materialized ref (physical view) the op raises
  // a catchable c10::Error -- same precondition as a real NPU, not a no-op or a crash.
  void* a = c10::rbln::malloc(/*device_index=*/0, 64);
  void* b = c10::rbln::malloc(/*device_index=*/0, 64);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_THROW(c10::rbln::set_device_layout_like(a, b), c10::Error);
  c10::rbln::free(a);
  c10::rbln::free(b);
}

TEST(RBLNDummyDeviceTest, SetFileOffloadingIsNoOp) {
  // File offloading is a runtime flag; enabling/disabling it must not throw on
  // the dummy runtime context.
  EXPECT_NO_THROW(c10::rbln::set_file_offloading_enabled(true));
  EXPECT_NO_THROW(c10::rbln::set_file_offloading_enabled(false));
}
