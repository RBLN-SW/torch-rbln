// Tests for the no-device contract: torch-rbln must remain usable for
// compile-only work on a host with no physical NPU (device_count() == 0),
// mirroring how torch.cuda.device_count() returns 0 on a CPU-only host.
//
// The contract only manifests with zero devices, so the test skips itself
// unless it is actually running in that configuration (e.g. a CI lane on a
// hardware-less build box).
//
// Contract with device_count() == 0:
//   - device enumeration reports 0 without throwing (vs. the production path
//     that asserts at least one logical device when an NPU is present);
//   - selecting a current logical device is permitted bookkeeping (no throw),
//     so DTensor / DeviceMesh can place tensors on rbln:0..N-1 during tracing;
//   - actually allocating device memory fails at the point of use.
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

TEST(RBLNNoDeviceTest, NoDeviceContract) {
  if (c10::rbln::get_device_count() != 0) {
    GTEST_SKIP() << "Requires a host with no physical NPU (device_count != 0).";
  }

  // Enumeration reports 0 and did not throw — this is the core regression this
  // change fixes (previously DeviceMappingManager asserted >= 1 logical device).
  EXPECT_EQ(c10::rbln::get_device_count(), 0);

  // Selecting a current device is bookkeeping only and must not throw, even
  // though no device is assigned.
  EXPECT_NO_THROW(c10::rbln::set_device_index(0));
  EXPECT_EQ(c10::rbln::get_device_index(), 0);
  EXPECT_NO_THROW(c10::rbln::set_device_index(3));
  EXPECT_EQ(c10::rbln::get_device_index(), 3);

  // Operations that actually use a device must fail at the point of use rather
  // than silently succeeding. They resolve a logical device via the shared
  // to_device_id() chokepoint, so the guard covers them uniformly (allocation,
  // synchronize, and any other index-based runtime call).
  EXPECT_THROW(c10::rbln::malloc(/*device_index=*/0, /*nbytes=*/1024), c10::Error);
  EXPECT_THROW(c10::rbln::synchronize(/*device_index=*/0), c10::Error);
}
