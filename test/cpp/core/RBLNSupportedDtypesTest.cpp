#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>

class RBLNSupportedDtypesTest : public ::testing::Test {
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

  const c10::DeviceIndex initial_device_index_ = 0;
};

namespace {

template <typename Arr, typename T>
bool contains(const Arr& arr, const T& value) {
  return std::find(arr.begin(), arr.end(), value) != arr.end();
}

} // namespace

TEST_F(RBLNSupportedDtypesTest, DispatchCatalogContents) {
  constexpr std::array kExpected = {c10::kHalf, c10::kBFloat16, c10::kFloat};
  for (const auto scalar_type : kExpected) {
    EXPECT_TRUE(contains(c10::rbln::kDispatchDtypes, scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, SdpaCatalogContents) {
  constexpr std::array kExpected = {c10::kHalf, c10::kBFloat16};
  for (const auto scalar_type : kExpected) {
    EXPECT_TRUE(contains(c10::rbln::kSdpaDtypes, scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, AmpCatalogIsEmpty) {
  // AMP autocast is not implemented yet (no AutocastPrivateUse1 cast policy),
  // so the catalog advertises no dtypes and PyTorch disables autocast instead
  // of dispatching to a missing kernel. See RBLNSupportedDtypes.h.
  EXPECT_TRUE(c10::rbln::kAmpDtypes.empty());
}

TEST_F(RBLNSupportedDtypesTest, SdpaCatalogIsFloatOnly) {
  for (const auto scalar_type : c10::rbln::kSdpaDtypes) {
    EXPECT_TRUE(c10::isFloatingType(scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, CapabilityCatalogContents) {
  // Device-resident dtypes only (fp16/bf16/fp32); other dtypes are CPU-backed.
  constexpr std::array kExpected = {c10::kHalf, c10::kBFloat16, c10::kFloat};
  for (const auto scalar_type : kExpected) {
    EXPECT_TRUE(contains(c10::rbln::kCapabilityDtypes, scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, IsDispatchDtypeAcceptsAdmitted) {
  for (const auto scalar_type : c10::rbln::kDispatchDtypes) {
    EXPECT_TRUE(c10::rbln::is_dispatch_dtype(scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, IsDispatchDtypeRejectsUnsupported) {
  constexpr std::array kUnsupported = {
      c10::kInt,
      c10::kLong,
      c10::kBool,
  };
  for (const auto scalar_type : kUnsupported) {
    EXPECT_FALSE(c10::rbln::is_dispatch_dtype(scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, DispatchExtensionAdmitsOnlyWhatIsSet) {
  EXPECT_TRUE(c10::rbln::is_dispatch_dtype_rt(c10::kFloat));
  EXPECT_FALSE(c10::rbln::is_dispatch_dtype_rt(c10::kInt));

  c10::rbln::set_dispatch_extension({c10::kInt, c10::kBool});
  EXPECT_TRUE(c10::rbln::is_dispatch_dtype_rt(c10::kInt));
  EXPECT_TRUE(c10::rbln::is_dispatch_dtype_rt(c10::kBool));
  EXPECT_FALSE(c10::rbln::is_dispatch_dtype_rt(c10::kLong));

  // A dtype outside kDispatchExtensionDtypes is rejected and the previous
  // extension stays in place.
  EXPECT_THROW(
      c10::rbln::set_dispatch_extension({c10::kLong, c10::kComplexFloat}),
      c10::Error);
  EXPECT_TRUE(c10::rbln::is_dispatch_dtype_rt(c10::kInt));
  EXPECT_FALSE(c10::rbln::is_dispatch_dtype_rt(c10::kLong));

  c10::rbln::set_dispatch_extension({});
  EXPECT_FALSE(c10::rbln::is_dispatch_dtype_rt(c10::kInt));
}

static_assert(
    !c10::rbln::kDispatchDtypes.empty(),
    "dispatch dtype catalog must not be empty");
static_assert(
    !c10::rbln::kSdpaDtypes.empty(),
    "SDPA dtype catalog must not be empty");
static_assert(
    c10::rbln::kAmpDtypes.empty(),
    "AMP dtype catalog is intentionally empty until autocast is implemented");
static_assert(
    !c10::rbln::kCapabilityDtypes.empty(),
    "capability dtype catalog must not be empty");
static_assert(
    c10::rbln::is_dispatch_dtype(c10::kHalf),
    "is_dispatch_dtype must be constexpr");
