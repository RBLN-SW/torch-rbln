#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <set>

// Exercises the stream half of RBLNGuardImpl through VirtualGuardImpl (the same
// path torch.Stream takes). Stream creation needs the device initialized, so each
// test forces it with a scratch allocation.
class RBLNStreamTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
  }

  void SetUp() override {
    c10::rbln::set_device_index(0);
    scratch_ = c10::rbln::malloc(0, 4096); // initialize the device
    ASSERT_NE(scratch_, nullptr);
    reset_to_default();
  }

  void TearDown() override {
    reset_to_default();
    if (scratch_ != nullptr) {
      c10::rbln::free(scratch_);
      scratch_ = nullptr;
    }
  }

  // Leave the thread's current stream at the default so tests don't leak state.
  void reset_to_default() {
    const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
    impl.exchangeStream(impl.getDefaultStream(device_));
  }

  void* scratch_ = nullptr;
  const c10::Device device_{c10::kPrivateUse1, 0};
};

TEST_F(RBLNStreamTest, DefaultStreamIsZero) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  EXPECT_EQ(impl.getDefaultStream(device_).id(), 0);
  EXPECT_EQ(impl.getDefaultStream(device_).device(), device_);
}

TEST_F(RBLNStreamTest, CurrentDefaultsToDefault) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  EXPECT_EQ(impl.getStream(device_), impl.getDefaultStream(device_));
}

TEST_F(RBLNStreamTest, NewStreamsAreDistinct) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  const auto s1 = impl.getNewStream(device_, 0);
  const auto s2 = impl.getNewStream(device_, 0);
  EXPECT_NE(s1.id(), 0);
  EXPECT_NE(s2.id(), 0);
  EXPECT_NE(s1, s2);
  EXPECT_EQ(s1.device(), device_);
}

TEST_F(RBLNStreamTest, ExchangeStreamSetsAndRestores) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  const auto s = impl.getNewStream(device_, 0);
  const auto def = impl.getDefaultStream(device_);
  const auto prev = impl.exchangeStream(s);
  EXPECT_EQ(prev, def);
  EXPECT_EQ(impl.getStream(device_), s);
  const auto back = impl.exchangeStream(def);
  EXPECT_EQ(back, s);
  EXPECT_EQ(impl.getStream(device_), def);
}

TEST_F(RBLNStreamTest, PriorityIsAcceptedButIgnored) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  // RBLN has no stream priorities; any value is accepted and yields a usable stream.
  const auto high = impl.getNewStream(device_, -1);
  const auto low = impl.getNewStream(device_, 1);
  EXPECT_NE(high.id(), 0);
  EXPECT_NE(low.id(), 0);
}

TEST_F(RBLNStreamTest, IdleStreamQueryIsTrue) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  const auto s = impl.getNewStream(device_, 0);
  EXPECT_TRUE(impl.queryStream(s));
  impl.synchronizeStream(s); // idle -> no-op, must not crash
  EXPECT_TRUE(impl.queryStream(s));
}

TEST_F(RBLNStreamTest, GlobalPoolRoundRobins) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  const auto first = impl.getStreamFromGlobalPool(device_, false);
  std::set<c10::StreamId> seen{first.id()};
  bool cycled = false;
  for (int i = 0; i < 128; ++i) {
    const auto s = impl.getStreamFromGlobalPool(device_, false);
    EXPECT_NE(s.id(), 0);
    if (s.id() == first.id()) {
      cycled = true;
      break;
    }
    seen.insert(s.id());
  }
  EXPECT_TRUE(cycled); // a fixed-size pool cycles back to the first stream
  EXPECT_GT(seen.size(), 1u); // and holds more than one stream
}

TEST_F(RBLNStreamTest, StreamIdRoundTrips) {
  const c10::impl::VirtualGuardImpl impl(c10::kPrivateUse1);
  const auto s = impl.getNewStream(device_, 0);
  const auto data = s.pack3();
  const auto rebuilt = c10::Stream::unpack3(data.stream_id, data.device_index, data.device_type);
  EXPECT_EQ(rebuilt, s);
  EXPECT_EQ(rebuilt.id(), s.id());
  EXPECT_EQ(rebuilt.device(), device_);
}
