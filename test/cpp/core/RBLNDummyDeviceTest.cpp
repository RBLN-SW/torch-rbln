// Tests for the dummy device contract (RBLN_DUMMY_DEVICE=1).
//
// Dummy mode presents host-backed logical device(s) with no NPU, so a model can
// be constructed and compiled without hardware. Allocations and transfers go
// through the runtime's dummy device, which backs device memory with host
// memory — torch adds no host-memory shim of its own. The returned device
// pointer is a handle, NOT host-dereferenceable, so these tests move data only
// through memcpy_h2v / memcpy_v2h / memcpy_v2v. Kernel/graph execution still
// requires a real NPU (guarded elsewhere).
//
// The env MUST be set before the DeviceMappingManager singleton initializes (it
// reads the flags once at init). A file-scope static initializer sets them
// before main(). RBLN_FORCE_NPU_NAME is set too: with no NPU to probe, dummy
// registration resolves the target SoC from it.
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNCachingAllocator.h>
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
  setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25", /*overwrite=*/1);
  return 0;
}();
} // namespace

TEST(RBLNDummyDeviceTest, ReportsLogicalButNoPhysicalDevice) {
  EXPECT_TRUE(c10::rbln::is_dummy_device());
  EXPECT_EQ(c10::rbln::get_device_count(), 2); // RBLN_DEVICE_MAP group count
  // physical count must not query the runtime (no NPU); reports 0.
  EXPECT_EQ(c10::rbln::get_physical_device_count(), 0);
}

TEST(RBLNDummyDeviceTest, AllocateAndTransfer) {
  // Allocation succeeds (host-backed dummy device) instead of throwing as it would
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
  // (view) pointer resolves to the same allocation via the segment map.
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

TEST(RBLNDummyDeviceTest, TransfersRejectOutOfBounds) {
  // A small block lies in a segment of kSmallSegment bytes, so a longer transfer runs past
  // the device memory it starts in and is rejected, not silently OOB.
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  constexpr size_t kPastSegment = c10::rbln::caching::kSmallSegment + 1;
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);
  std::vector<uint8_t> host(kPastSegment);
  EXPECT_THROW(c10::rbln::memcpy_h2v(dev, host.data(), kPastSegment), c10::Error);
  EXPECT_THROW(c10::rbln::memcpy_v2h(host.data(), dev, kPastSegment), c10::Error);
  EXPECT_THROW(c10::rbln::fill_zeros(dev, kPastSegment), c10::Error);
  // An unknown device pointer is rejected too.
  int32_t stack = 0;
  EXPECT_THROW(c10::rbln::memcpy_v2h(host.data(), &stack, sizeof(int32_t)), c10::Error);
  c10::rbln::free(dev);
}

TEST(RBLNDummyDeviceTest, AsyncTransfersRoundTrip) {
  // Async transfers complete on the dummy device; same data and bounds contract as
  // the sync variants.
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
  std::vector<uint8_t> past_segment(c10::rbln::caching::kSmallSegment + 1);
  EXPECT_THROW(c10::rbln::memcpy_h2v_async(a, past_segment.data(), past_segment.size()), c10::Error);
  c10::rbln::free(a);
  c10::rbln::free(b);
}

TEST(RBLNDummyDeviceTest, FillZerosClearsDeviceMemory) {
  constexpr size_t kBytes = 4 * sizeof(int32_t);
  void* dev = c10::rbln::malloc(/*device_index=*/0, kBytes);
  ASSERT_NE(dev, nullptr);
  const int32_t values[4] = {10, 11, 12, 13};
  c10::rbln::memcpy_h2v(dev, values, kBytes);

  ASSERT_NO_THROW(c10::rbln::fill_zeros(dev, kBytes));
  int32_t out[4] = {1, 1, 1, 1};
  c10::rbln::memcpy_v2h(out, dev, kBytes);
  const int32_t expected[4] = {0, 0, 0, 0};
  EXPECT_EQ(0, std::memcmp(expected, out, kBytes));

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
