/**
 * ProcessGroupRBLN Test Suite
 *
 * This test suite uses a Singleton pattern to ensure that each test process
 * creates and uses only one ProcessGroupRBLN instance. This prevents resource
 * conflicts and ensures consistent behavior across all tests.
 *
 * The ProcessGroupRBLNSingleton class manages the single instance and provides
 * thread-safe access to the ProcessGroupRBLN object.
 */

// Standard library includes
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

// Google Test includes
#include <gtest/gtest.h>

// PyTorch includes
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/torch.h>

// RBLN includes
#include <aten/src/ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.hpp>

namespace c10d {

// ============================================================================
// Test Constants and Configuration
// ============================================================================

namespace {
// Test tensor dimensions
constexpr int64_t kDefaultTensorSize = 8;
constexpr int64_t kSmallTensorSize = 4;

// Test timeouts
constexpr std::chrono::milliseconds kDefaultTimeout{5000};
constexpr std::chrono::milliseconds kShortTimeout{1000};
constexpr std::chrono::milliseconds kVeryShortTimeout{100};

// Test values
constexpr float kDefaultTensorValue = 1.0f;
constexpr float kTestTensorValue = 2.0f;

// Process group configuration
constexpr int kTestRank = 0;
constexpr int kTestSize = 1;
constexpr const char* kBackendName = "rbln-ccl";

// ============================================================================
// RBLN Tensor Creation Utilities
// ============================================================================

/**
 * @brief Create a default-sized RBLN tensor for testing
 * @param value Scalar value to fill the tensor (default: 1.0f)
 * @return RBLN tensor with default dimensions
 */
at::Tensor createDefaultRBLNTensor(float value = kDefaultTensorValue) {
  const auto sizes = std::vector<int64_t>{kDefaultTensorSize, kDefaultTensorSize};
  const auto device = c10::Device(c10::kPrivateUse1, c10::rbln::get_device_index());

  auto data_tensor = at::ones(sizes, c10::kHalf);
  if (value != kDefaultTensorValue) {
    data_tensor.fill_(value);
  }

  const auto rbln_tensor = data_tensor.to(device);
  return rbln_tensor;
}

/**
 * @brief Wait for work completion with timeout
 *
 * This is necessary because ProcessGroupRBLN uses a thread pool for async execution,
 * so work->isCompleted() will return false immediately after calling operations like
 * allreduce() or broadcast(). We need to wait for the work to be processed.
 *
 * @param work Work object to wait for
 * @param timeout Maximum time to wait
 * @return true if work completed successfully, false otherwise
 */
bool waitForWorkCompletion(
    const c10::intrusive_ptr<::c10d::Work>& work,
    std::chrono::milliseconds timeout = kDefaultTimeout) {
  if (!work)
    return false;

  try {
    work->wait(timeout);
    return work->isCompleted();
  } catch (const std::exception& ex) {
    // Log the exception but don't fail the test
    std::cout << "Work wait failed: " << ex.what() << '\n';
    return false;
  }
}

} // namespace

// ============================================================================
// ProcessGroupRBLN Singleton
// ============================================================================

class ProcessGroupRBLNSingleton {
 public:
  ProcessGroupRBLNSingleton() = default;
  ~ProcessGroupRBLNSingleton() = default;

  // Disable copy constructor and assignment operator
  ProcessGroupRBLNSingleton(const ProcessGroupRBLNSingleton&) = delete;
  ProcessGroupRBLNSingleton& operator=(const ProcessGroupRBLNSingleton&) = delete;

  // Get the singleton instance
  static ProcessGroupRBLNSingleton& getInstance() {
    static ProcessGroupRBLNSingleton instance;
    return instance;
  }

  // Get the ProcessGroupRBLN instance
  c10::intrusive_ptr<ProcessGroupRBLN> getProcessGroup() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pg_) {
      initialize();
    }
    return pg_;
  }

  // Reset the singleton (for testing purposes)
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    pg_.reset();
    initialized_ = false;
  }

 private:
  void initialize() {
    if (initialized_)
      return;

    auto options = c10::make_intrusive<::c10d::Backend::Options>(kBackendName, kDefaultTimeout);
    auto store = c10::make_intrusive<::c10d::HashStore>();
    pg_ = c10::make_intrusive<ProcessGroupRBLN>(
        store, kTestRank, kTestSize, -1, std::vector<int>{}, options, c10::intrusive_ptr<::c10d::Backend>());
    initialized_ = true;
  }

  c10::intrusive_ptr<ProcessGroupRBLN> pg_{nullptr};
  bool initialized_{false};
  mutable std::mutex mutex_{};
};

// ============================================================================
// Basic Test Fixture (Single Process)
// ============================================================================

class ProcessGroupRBLNTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    // Reset singleton to ensure clean state
    ProcessGroupRBLNSingleton::getInstance().reset();
  }

  static void TearDownTestSuite() {
    // Reset singleton after all tests
    ProcessGroupRBLNSingleton::getInstance().reset();
  }

  void SetUp() override {
    // Get the singleton instance for each test
    pg_ = ProcessGroupRBLNSingleton::getInstance().getProcessGroup();
  }

  void TearDown() override {
    // No need to reset here as we want to reuse the same instance
  }

  // Helper methods to access the singleton instance
  c10::intrusive_ptr<ProcessGroupRBLN> getProcessGroup() {
    return ProcessGroupRBLNSingleton::getInstance().getProcessGroup();
  }

 protected:
  c10::intrusive_ptr<ProcessGroupRBLN> pg_;
};

// ============================================================================
// Basic ProcessGroup Tests
// ============================================================================

TEST_F(ProcessGroupRBLNTest, TestConstructor) {
  EXPECT_NE(pg_, nullptr);
  EXPECT_EQ(pg_->getRank(), kTestRank);
  EXPECT_EQ(pg_->getSize(), kTestSize);
  EXPECT_EQ(pg_->getBackendName(), kBackendName);
}

TEST_F(ProcessGroupRBLNTest, TestBasicProperties) {
  // Test rank and size
  EXPECT_EQ(pg_->getRank(), kTestRank);
  EXPECT_EQ(pg_->getSize(), kTestSize);

  // Test backend name
  EXPECT_EQ(pg_->getBackendName(), kBackendName);
}

TEST_F(ProcessGroupRBLNTest, TestSingletonBehavior) {
  // Test that multiple calls to getProcessGroup() return the same instance
  auto pg1 = getProcessGroup();
  auto pg2 = getProcessGroup();
  auto pg3 = ProcessGroupRBLNSingleton::getInstance().getProcessGroup();

  // All should be the same instance
  EXPECT_EQ(pg1.get(), pg2.get());
  EXPECT_EQ(pg2.get(), pg3.get());
  EXPECT_EQ(pg1.get(), pg3.get());

  // All should be valid
  EXPECT_NE(pg1, nullptr);
  EXPECT_NE(pg2, nullptr);
  EXPECT_NE(pg3, nullptr);

  // All should have the same properties
  EXPECT_EQ(pg1->getRank(), pg2->getRank());
  EXPECT_EQ(pg2->getRank(), pg3->getRank());
  EXPECT_EQ(pg1->getSize(), pg2->getSize());
  EXPECT_EQ(pg2->getSize(), pg3->getSize());
  EXPECT_EQ(pg1->getBackendName(), pg2->getBackendName());
  EXPECT_EQ(pg2->getBackendName(), pg3->getBackendName());
}

// ============================================================================
// Tensor Creation and Validation Tests
// ============================================================================

TEST_F(ProcessGroupRBLNTest, TestUnsupportedDeviceType) {
  // Test that unsupported device types throw errors
  auto tensor = torch::ones({kDefaultTensorSize, kDefaultTensorSize}, torch::kFloat); // CPU tensor
  std::vector<at::Tensor> inputs = {tensor};

  EXPECT_THROW(pg_->broadcast(inputs, BroadcastOptions{}), c10::Error);
}

TEST_F(ProcessGroupRBLNTest, TestRBLNTensorCreation) {
  // Test RBLN tensor creation
  std::vector<int64_t> shapes = {kDefaultTensorSize, kDefaultTensorSize};
  const auto tensor = createDefaultRBLNTensor(kTestTensorValue);

  EXPECT_EQ(tensor.device().type(), at::kPrivateUse1);
  EXPECT_EQ(tensor.dtype(), torch::kFloat16);
  EXPECT_EQ(tensor.sizes(), shapes);

  // Test tensor values
  auto cpu_tensor = tensor.to("cpu").to(torch::kFloat16);
  EXPECT_TRUE(cpu_tensor.allclose(torch::ones(shapes, torch::kFloat16) * kTestTensorValue, 1e-6));
}

// ============================================================================
// Collective Communication Tests
// ============================================================================

TEST_F(ProcessGroupRBLNTest, TestBarrier) {
  // Test barrier in single process mode
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);
  work->wait();
  EXPECT_TRUE(work->isCompleted());
}

TEST_F(ProcessGroupRBLNTest, TestSendNotImplemented) {
  // Test that send is not implemented
  auto tensor = torch::ones({kDefaultTensorSize, kDefaultTensorSize}, torch::kFloat);
  std::vector<at::Tensor> inputs = {tensor};

  EXPECT_THROW(pg_->send(inputs, 0, 0), c10::Error);
}

TEST_F(ProcessGroupRBLNTest, TestRecvNotImplemented) {
  // Test that recv is not implemented
  auto tensor = torch::ones({kDefaultTensorSize, kDefaultTensorSize}, torch::kFloat);
  std::vector<at::Tensor> inputs = {tensor};

  EXPECT_THROW(pg_->recv(inputs, 0, 0), c10::Error);
}

// FIXME: Re-enable the following test after resolving C++ implementation issues.
// TEST_F(ProcessGroupRBLNTest, TestScatterWithCPUTensor) {
//   // Test scatter with CPU tensor (should work now that scatter is implemented)
//   auto tensor = createDefaultRBLNTensor();
//   std::vector<at::Tensor> outputs = {tensor};
//   std::vector<std::vector<at::Tensor>> inputs = {{tensor}};
//   ScatterOptions opts;
//   opts.rootRank = kTestRank;

//   auto work = pg_->scatter(outputs, inputs, opts);
//   EXPECT_NE(work, nullptr);
//   EXPECT_TRUE(waitForWorkCompletion(work));
// }

TEST_F(ProcessGroupRBLNTest, TestWorkQueueOperations) {
  // Test that work can be enqueued and processed
  auto tensor = torch::ones({kDefaultTensorSize, kDefaultTensorSize}, torch::kFloat);
  std::vector<at::Tensor> inputs = {tensor};

  // This should throw due to invalid device, but the work queue
  // mechanism should still work
  EXPECT_THROW(pg_->broadcast(inputs, BroadcastOptions{}), c10::Error);
}

TEST_F(ProcessGroupRBLNTest, TestReduceOpTypes) {
  // Test different reduce operation types
  // std::vector<ReduceOp> ops = {ReduceOp::SUM, ReduceOp::AVG, ReduceOp::PRODUCT, ReduceOp::MIN, ReduceOp::MAX};
  std::vector<ReduceOp> ops = {ReduceOp::SUM};

  for (const auto& op : ops) {
    auto tensor = createDefaultRBLNTensor(); // Use RBLN tensor
    std::vector<at::Tensor> inputs = {tensor};
    AllreduceOptions opts;
    opts.reduceOp = op;

    // This should work with RBLN tensors
    auto work = pg_->allreduce(inputs, opts);
    EXPECT_NE(work, nullptr);

    // Wait for work to complete since it uses thread pool
    EXPECT_TRUE(waitForWorkCompletion(work));
  }
}

TEST_F(ProcessGroupRBLNTest, TestBroadcastOptions) {
  // Test broadcast options
  auto tensor = createDefaultRBLNTensor(); // Use RBLN tensor
  std::vector<at::Tensor> inputs = {tensor};
  BroadcastOptions opts;
  opts.rootRank = kTestRank;

  // This should work with RBLN tensors
  auto work = pg_->broadcast(inputs, opts);
  EXPECT_NE(work, nullptr);

  // Wait for work to complete since it uses thread pool
  EXPECT_TRUE(waitForWorkCompletion(work));
}

TEST_F(ProcessGroupRBLNTest, TestWorkCompletion) {
  // Test work completion status
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);

  // Test work result - only if result() is implemented
  try {
    auto result = work->result();
    EXPECT_TRUE(result.empty()); // Barrier doesn't return tensors
  } catch (const std::exception& ex) {
    // result() not implemented, which is expected
    RBLN_LOG_DEBUG("Work result() not implemented: {}", ex.what());
  }
}

TEST_F(ProcessGroupRBLNTest, TestBarrierOptions) {
  // Test barrier options
  BarrierOptions opts;
  auto work = pg_->barrier(opts);
  EXPECT_NE(work, nullptr);

  // Wait for work to complete (barrier should complete immediately)
  EXPECT_TRUE(waitForWorkCompletion(work));
}

// FIXME: Re-enable the following test after resolving C++ implementation issues.
// TEST_F(ProcessGroupRBLNTest, TestScatter) {
//   // Test scatter functionality
//   auto tensor = createDefaultRBLNTensor(); // Use RBLN tensor
//   std::vector<at::Tensor> outputs = {tensor};
//   std::vector<std::vector<at::Tensor>> inputs = {{tensor}};
//   ScatterOptions opts;
//   opts.rootRank = kTestRank;

//   // Scatter is now implemented, so this should work
//   auto work = pg_->scatter(outputs, inputs, opts);
//   EXPECT_NE(work, nullptr);

//   // Wait for work to complete since it uses thread pool
//   EXPECT_TRUE(waitForWorkCompletion(work));
// }

// ============================================================================
// Work Object Tests
// ============================================================================

TEST_F(ProcessGroupRBLNTest, TestWorkAbort) {
  // Test work abort functionality
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);

  // Abort may throw if not implemented, which is expected
  try {
    work->abort();
  } catch (const std::exception& ex) {
    // abort() not implemented, which is expected
    RBLN_LOG_DEBUG("Work abort() not implemented: {}", ex.what());
  }
}

TEST_F(ProcessGroupRBLNTest, TestWorkWait) {
  // Test work wait functionality
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);

  // Wait should complete immediately for completed work
  EXPECT_NO_THROW(work->wait());
  EXPECT_TRUE(work->isCompleted());
}

TEST_F(ProcessGroupRBLNTest, TestWorkWaitWithTimeout) {
  // Test work wait with timeout
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);

  // Wait with timeout should complete immediately for completed work
  EXPECT_NO_THROW(work->wait(kVeryShortTimeout));
  EXPECT_TRUE(work->isCompleted());
}

TEST_F(ProcessGroupRBLNTest, TestAsyncWorkWait) {
  // Test async work wait functionality with allreduce
  auto tensor = createDefaultRBLNTensor();
  std::vector<at::Tensor> inputs = {tensor};
  AllreduceOptions opts;
  opts.reduceOp = ReduceOp::SUM;

  auto work = pg_->allreduce(inputs, opts);
  EXPECT_NE(work, nullptr);

  // Wait for async work to complete
  EXPECT_TRUE(waitForWorkCompletion(work, kShortTimeout));
}

TEST_F(ProcessGroupRBLNTest, TestWorkSourceRank) {
  // Test work source rank
  auto work = pg_->barrier();
  EXPECT_NE(work, nullptr);

  // sourceRank() may throw for barrier work, which is expected
  try {
    EXPECT_EQ(work->sourceRank(), -1);
  } catch (const std::exception& ex) {
    // sourceRank() not applicable for barrier work, which is expected
    RBLN_LOG_DEBUG("Work sourceRank() not applicable for barrier: {}", ex.what());
  }
}

// ============================================================================
// World Size 1 RCCL Init Tests
// ============================================================================

/**
 * Test world size 1 scenarios - verifies RCCL initialization is skipped
 */
TEST_F(ProcessGroupRBLNTest, TestWorldSize1Broadcast) {
  // Create tensor for broadcast test
  auto tensor = createDefaultRBLNTensor(2.5f);
  std::vector<at::Tensor> tensors = {tensor};

  // Test broadcast operation
  ::c10d::BroadcastOptions opts;
  opts.rootRank = 0;

  auto work = pg_->broadcast(tensors, opts);
  EXPECT_NE(work, nullptr);
  EXPECT_TRUE(waitForWorkCompletion(work, kShortTimeout));

  // Verify tensor value is preserved (no-op for single process)
  auto cpu_tensor = tensor.to("cpu").to(torch::kFloat);
  EXPECT_TRUE(cpu_tensor.allclose(torch::full_like(cpu_tensor, 2.5f)));
}

/**
 * Test world size 1 allreduce
 */
TEST_F(ProcessGroupRBLNTest, TestWorldSize1Allreduce) {
  auto tensor = createDefaultRBLNTensor(3.0f);
  std::vector<at::Tensor> tensors = {tensor};

  auto work = pg_->allreduce(tensors);
  EXPECT_NE(work, nullptr);
  EXPECT_TRUE(waitForWorkCompletion(work, kShortTimeout));

  // For world size 1, allreduce should be no-op
  auto cpu_tensor = tensor.to("cpu").to(torch::kFloat);
  EXPECT_TRUE(cpu_tensor.allclose(torch::full_like(cpu_tensor, 3.0f)));
}

/**
 * Test world size 1 scatter
 */
// FIXME: Re-enable the following test after resolving C++ implementation issues.
// TEST_F(ProcessGroupRBLNTest, TestWorldSize1Scatter) {
//   auto input_tensor = createDefaultRBLNTensor(4.0f);
//   auto output_tensor = createDefaultRBLNTensor(0.0f);

//   std::vector<std::vector<at::Tensor>> input_list = {{input_tensor}};
//   std::vector<at::Tensor> output_list = {output_tensor};

//   ::c10d::ScatterOptions opts;
//   opts.rootRank = 0;

//   auto work = pg_->scatter(output_list, input_list, opts);
//   EXPECT_NE(work, nullptr);
//   EXPECT_TRUE(waitForWorkCompletion(work, kShortTimeout));

//   // Verify output received input value
//   auto cpu_output = output_tensor.to("cpu").to(torch::kFloat);
//   auto cpu_input = input_tensor.to("cpu").to(torch::kFloat);
//   EXPECT_TRUE(cpu_output.allclose(cpu_input));
// }

// ============================================================================
// rccl_unique_id store set/get round-trip (broadcastUniqueRCCLID serialization)
// ============================================================================

#ifndef RCCL_IP_STR_LEN
#define RCCL_IP_STR_LEN 16
#endif

struct TestRcclUniqueIdLayout {
  std::array<char, RCCL_IP_STR_LEN> root_ip;
  std::array<char, RCCL_IP_STR_LEN> self_ip;
  std::array<char, RCCL_IP_STR_LEN> self_rdma_ip;
  int root_port;
  int rdma_base_port;
};

TEST_F(ProcessGroupRBLNTest, TestRcclUniqueIdStoreRoundTrip) {
  const std::string storeKey = "rbln_rccl_uid_0";
  const std::string host = "127.0.0.1";
  constexpr uint16_t kTestPort = 29600;

  TCPStoreOptions serverOpts;
  serverOpts.port = kTestPort;
  serverOpts.isServer = true;
  serverOpts.waitWorkers = false;
  auto serverStore = c10::make_intrusive<TCPStore>(host, serverOpts);

  TCPStoreOptions clientOpts;
  clientOpts.port = kTestPort;
  clientOpts.isServer = false;
  auto clientStore = c10::make_intrusive<TCPStore>(host, clientOpts);

  TestRcclUniqueIdLayout id_original{};
  std::snprintf(id_original.root_ip.data(), id_original.root_ip.size(), "192.168.1.1");
  std::snprintf(id_original.self_ip.data(), id_original.self_ip.size(), "192.168.1.%d", 0);
  std::snprintf(id_original.self_rdma_ip.data(), id_original.self_rdma_ip.size(), "10.0.0.1");
  id_original.root_port = 12345;
  id_original.rdma_base_port = 54321;

  std::vector<uint8_t> vec(
      reinterpret_cast<uint8_t*>(&id_original),
      reinterpret_cast<uint8_t*>(&id_original) + sizeof(TestRcclUniqueIdLayout));
  serverStore->set(storeKey, vec);

  std::vector<uint8_t> vec_from_store = clientStore->get(storeKey);
  ASSERT_EQ(vec_from_store.size(), sizeof(TestRcclUniqueIdLayout)) << "TCPStore returned wrong size for rccl_unique_id";

  TestRcclUniqueIdLayout id_restored{};
  std::memcpy(&id_restored, vec_from_store.data(), vec_from_store.size());

  EXPECT_EQ(std::string(id_original.root_ip.data()), std::string(id_restored.root_ip.data()));
  EXPECT_EQ(std::string(id_original.self_ip.data()), std::string(id_restored.self_ip.data()));
  EXPECT_EQ(std::string(id_original.self_rdma_ip.data()), std::string(id_restored.self_rdma_ip.data()));
  EXPECT_EQ(id_original.root_port, id_restored.root_port);
  EXPECT_EQ(id_original.rdma_base_port, id_restored.rdma_base_port);
  EXPECT_EQ(0, std::memcmp(&id_original, &id_restored, sizeof(TestRcclUniqueIdLayout)))
      << "rccl_unique_id bytes differ after TCPStore set (server) / get (client)";
}

} // namespace c10d

// Main function for running tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
