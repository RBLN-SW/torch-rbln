#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <gtest/gtest.h>

#include <algorithm>

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

template <typename Arr, typename T>
bool contains(const Arr& arr, const T& value) {
  return std::find(arr.begin(), arr.end(), value) != arr.end();
}

TEST_F(RBLNSupportedDtypesTest, DispatchCatalogContents) {
  constexpr c10::ScalarType kExpected[] = {c10::kHalf, c10::kBFloat16};
  for (const auto scalar_type : kExpected) {
    EXPECT_TRUE(contains(c10::rbln::kDispatchDtypes, scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, SdpaCatalogContents) {
  constexpr c10::ScalarType kExpected[] = {c10::kHalf, c10::kBFloat16};
  for (const auto scalar_type : kExpected) {
    EXPECT_TRUE(contains(c10::rbln::kSdpaDtypes, scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, AmpCatalogIsEmpty) {
  // AMP autocast is not implemented yet (no AutocastPrivateUse1 cast policy), so
  // the catalog advertises no dtypes and torch disables autocast instead of
  // dispatching to a missing kernel. See RBLNSupportedDtypes.h.
  EXPECT_TRUE(c10::rbln::kAmpDtypes.empty());
}

TEST_F(RBLNSupportedDtypesTest, SdpaCatalogIsFloatOnly) {
  for (const auto scalar_type : c10::rbln::kSdpaDtypes) {
    EXPECT_TRUE(c10::isFloatingType(scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, IsDispatchDtypeAcceptsAdmitted) {
  for (const auto scalar_type : c10::rbln::kDispatchDtypes) {
    EXPECT_TRUE(c10::rbln::is_dispatch_dtype(scalar_type));
  }
}

TEST_F(RBLNSupportedDtypesTest, IsDispatchDtypeRejectsUnsupported) {
  constexpr c10::ScalarType kUnsupported[] = {
      c10::kFloat,
      c10::kInt,
      c10::kLong,
      c10::kBool,
  };
  for (const auto scalar_type : kUnsupported) {
    EXPECT_FALSE(c10::rbln::is_dispatch_dtype(scalar_type));
  }
}

static_assert(!c10::rbln::kDispatchDtypes.empty(), "dispatch dtype catalog must not be empty");
static_assert(!c10::rbln::kSdpaDtypes.empty(), "SDPA dtype catalog must not be empty");
static_assert(c10::rbln::kAmpDtypes.empty(), "AMP dtype catalog is intentionally empty until autocast is implemented");
static_assert(c10::rbln::is_dispatch_dtype(c10::kHalf), "is_dispatch_dtype must be constexpr");
static_assert(!c10::rbln::is_dispatch_dtype(c10::kFloat), "is_dispatch_dtype must be constexpr");
