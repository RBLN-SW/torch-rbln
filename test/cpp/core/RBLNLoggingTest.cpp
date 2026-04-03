#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHooksInterface.h>
#include <c10/rbln/RBLNLogging.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

class RBLNLoggingTest : public ::testing::Test {
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

TEST_F(RBLNLoggingTest, InitialDepthIsZero) {
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, SingleScopeDepthTracking) {
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
  {
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, NestedScopeDepthTracking) {
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
  {
    c10::rbln::RBLNScopeGuard guard1(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
    {
      c10::rbln::RBLNScopeGuard guard2(__FILE__, __LINE__, __func__);
      EXPECT_EQ(c10::rbln::get_scope_depth(), 2);
      {
        c10::rbln::RBLNScopeGuard guard3(__FILE__, __LINE__, __func__);
        EXPECT_EQ(c10::rbln::get_scope_depth(), 3);
      }
      EXPECT_EQ(c10::rbln::get_scope_depth(), 2);
    }
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, EarlyReturnDepthReset) {
  auto func = [](bool exit_early) {
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    if (exit_early) {
      return;
    }
  };
  func(true);
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
  func(false);
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, ExceptionSafetyDepthReset) {
  try {
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
    throw std::runtime_error("test");
  } catch (const std::runtime_error&) {
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, NestedExceptionSafetyDepthReset) {
  try {
    c10::rbln::RBLNScopeGuard guard1(__FILE__, __LINE__, __func__);
    {
      c10::rbln::RBLNScopeGuard guard2(__FILE__, __LINE__, __func__);
      EXPECT_EQ(c10::rbln::get_scope_depth(), 2);
      throw std::runtime_error("test");
    }
  } catch (const std::runtime_error&) {
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, ThreadLocalIndependence) {
  {
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);

    std::thread t([] {
      EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
      {
        c10::rbln::RBLNScopeGuard guard1(__FILE__, __LINE__, __func__);
        EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
        {
          c10::rbln::RBLNScopeGuard guard2(__FILE__, __LINE__, __func__);
          EXPECT_EQ(c10::rbln::get_scope_depth(), 2);
        }
        EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
      }
      EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
    });
    t.join();

    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, SequentialScopesNonNested) {
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
    {
      c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
      EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
    }
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, DepthVisibleAcrossCallStack) {
  auto func = []() {
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 2);
  };

  {
    c10::rbln::RBLNScopeGuard guard(__FILE__, __LINE__, __func__);
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
    func();
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, ScopeGuardMacroIsDebugOnly) {
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
  {
    RBLN_SCOPE_GUARD();
#ifdef NDEBUG
    EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
#else
    EXPECT_EQ(c10::rbln::get_scope_depth(), 1);
#endif
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, MultipleThreadsConcurrent) {
  constexpr int num_threads = 4;
  constexpr int nesting_depth = 10;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([nesting_depth] {
      EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
      std::vector<std::unique_ptr<c10::rbln::RBLNScopeGuard>> scopes;
      scopes.reserve(nesting_depth);
      for (int d = 0; d < nesting_depth; ++d) {
        scopes.push_back(std::make_unique<c10::rbln::RBLNScopeGuard>(__FILE__, __LINE__, __func__));
        EXPECT_EQ(c10::rbln::get_scope_depth(), d + 1);
      }
      while (!scopes.empty()) {
        int expected = static_cast<int>(scopes.size()) - 1;
        scopes.pop_back();
        EXPECT_EQ(c10::rbln::get_scope_depth(), expected);
      }
      EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(c10::rbln::get_scope_depth(), 0);
}

TEST_F(RBLNLoggingTest, GetNewGeneratorDoesNotAbort) {
  auto generator = c10::rbln::get_rbln_hooks()->getNewGenerator(initial_device_index_);
  (void)generator;
  SUCCEED();
}
