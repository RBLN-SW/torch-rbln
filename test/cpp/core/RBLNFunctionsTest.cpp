#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

class RBLNFunctionsTest : public ::testing::Test {
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
  const size_t size_0b_ = 0;
  const size_t size_1gib_ = 1ULL << 30;
  const size_t size_16gib_ = 1ULL << 34; // The memory capacity of ATOM is 15.7 GiB.
};

TEST_F(RBLNFunctionsTest, GetDeviceCount) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
}

TEST_F(RBLNFunctionsTest, GetAndSetDeviceIndex) {
  const auto current_device_index = c10::rbln::get_device_index();
  EXPECT_EQ(current_device_index, initial_device_index_);

  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    c10::rbln::set_device_index(device_index);
    const auto current_device_index = c10::rbln::get_device_index();
    EXPECT_EQ(current_device_index, device_index);
  }
}

TEST_F(RBLNFunctionsTest, SetNegativeDeviceIndex) {
  const auto current_device_index = c10::rbln::get_device_index();
  EXPECT_EQ(current_device_index, initial_device_index_);

  const c10::DeviceIndex negative_index = -1;
  c10::rbln::set_device_index(negative_index);
  EXPECT_EQ(c10::rbln::get_device_index(), current_device_index);
}

TEST_F(RBLNFunctionsTest, SetInvalidDeviceIndex) {
  const auto current_device_index = c10::rbln::get_device_index();
  EXPECT_EQ(current_device_index, initial_device_index_);

  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  const auto exceeded_index = device_count;
  EXPECT_THROW(c10::rbln::set_device_index(exceeded_index), c10::Error);
  EXPECT_EQ(c10::rbln::get_device_index(), current_device_index);
}

TEST_F(RBLNFunctionsTest, ExchangeDeviceIndex) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    const auto original_device_index = c10::rbln::get_device_index();
    const auto previous_device_index = c10::rbln::exchange_device_index(device_index);
    EXPECT_EQ(previous_device_index, original_device_index);
    EXPECT_EQ(c10::rbln::get_device_index(), device_index);
  }
}

TEST_F(RBLNFunctionsTest, ExchangeNegativeDeviceIndex) {
  const auto original_device_index = c10::rbln::get_device_index();

  const c10::DeviceIndex negative_index = -1;
  const auto previous_device_index = c10::rbln::exchange_device_index(negative_index);
  EXPECT_EQ(previous_device_index, original_device_index);
  EXPECT_EQ(c10::rbln::get_device_index(), original_device_index);
}

TEST_F(RBLNFunctionsTest, ExchangeInvalidDeviceIndex) {
  const auto original_device_index = c10::rbln::get_device_index();

  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  const c10::DeviceIndex exceeded_index = device_count;
  EXPECT_THROW(c10::rbln::exchange_device_index(exceeded_index), c10::Error);
  EXPECT_EQ(c10::rbln::get_device_index(), original_device_index);
}

TEST_F(RBLNFunctionsTest, IsEagerMalloc) {
  const auto is_eager_malloc = c10::rbln::is_eager_malloc();

  const auto* env = std::getenv("TORCH_RBLN_EAGER_MALLOC");
  if ((env != nullptr) && (std::string(env) == "1")) {
    EXPECT_TRUE(is_eager_malloc);
  } else {
    EXPECT_FALSE(is_eager_malloc);
  }
}

TEST_F(RBLNFunctionsTest, MallocAndFree) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    const auto data_15gib = c10::rbln::malloc(device_index, 15 * size_1gib_);
    EXPECT_TRUE(data_15gib != nullptr);

    // If memory is allocated lazily, the following assertion may fail because CPU memory is allocated instead of NPU
    // memory.
    if (c10::rbln::is_eager_malloc()) {
      EXPECT_THROW(c10::rbln::malloc(device_index, size_1gib_), c10::Error);
    }

    c10::rbln::free(data_15gib);

    // Double free
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
    EXPECT_THROW(c10::rbln::free(data_15gib), c10::Error);

    const auto data_1gib = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(data_1gib != nullptr);
    c10::rbln::free(data_1gib);
  }
}

TEST_F(RBLNFunctionsTest, MallocInvalidSize) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    EXPECT_THROW(c10::rbln::malloc(device_index, size_0b_), c10::Error);

    // If memory is allocated lazily, the following assertion may fail because CPU memory is allocated instead of NPU
    // memory.
    if (c10::rbln::is_eager_malloc()) {
      EXPECT_THROW(c10::rbln::malloc(device_index, size_16gib_), c10::Error);
    }
  }
}

TEST_F(RBLNFunctionsTest, FreeNullPtr) {
  void* data = nullptr;
  EXPECT_THROW(c10::rbln::free(data), c10::Error);
  EXPECT_EQ(data, nullptr);
}

TEST_F(RBLNFunctionsTest, SameDeviceMemcpy) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    std::vector<int8_t> src_cpu(size_1gib_, 1);
    const void* src_cpu_data = src_cpu.data();
    std::vector<int8_t> dst_cpu(size_1gib_, 0);
    void* dst_cpu_data = dst_cpu.data();

    const auto src_rbln_data = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(src_rbln_data != nullptr);
    auto dst_rbln_data = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(dst_rbln_data != nullptr);

    c10::rbln::memcpy_h2v(src_rbln_data, src_cpu_data, size_1gib_);
    c10::rbln::memcpy_v2v(dst_rbln_data, src_rbln_data, size_1gib_);
    c10::rbln::memcpy_v2h(dst_cpu_data, dst_rbln_data, size_1gib_);

    EXPECT_EQ(dst_cpu, src_cpu);

    c10::rbln::free(src_rbln_data);
    c10::rbln::free(dst_rbln_data);
  }
}

TEST_F(RBLNFunctionsTest, CrossDeviceMemcpy) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  if (device_count < 2) {
    GTEST_SKIP() << "Skipping: cross-device memcpy requires at least 2 devices.";
  }
  for (c10::DeviceIndex src_device_index = 0; src_device_index < device_count; ++src_device_index) {
    for (c10::DeviceIndex dst_device_index = 0; dst_device_index < device_count; ++dst_device_index) {
      if (src_device_index != dst_device_index) {
        std::vector<int8_t> src_cpu(size_1gib_, 1);
        const void* src_cpu_data = src_cpu.data();
        std::vector<int8_t> dst_cpu(size_1gib_, 0);
        void* dst_cpu_data = dst_cpu.data();

        const auto src_rbln_data = c10::rbln::malloc(src_device_index, size_1gib_);
        EXPECT_TRUE(src_rbln_data != nullptr);
        auto dst_rbln_data = c10::rbln::malloc(dst_device_index, size_1gib_);
        EXPECT_TRUE(dst_rbln_data != nullptr);

        c10::rbln::memcpy_h2v(src_rbln_data, src_cpu_data, size_1gib_);
        c10::rbln::memcpy_v2v(dst_rbln_data, src_rbln_data, size_1gib_);
        c10::rbln::memcpy_v2h(dst_cpu_data, dst_rbln_data, size_1gib_);
        EXPECT_EQ(dst_cpu, src_cpu);

        c10::rbln::free(src_rbln_data);
        c10::rbln::free(dst_rbln_data);
      }
    }
  }
}

// Empty input must be a clean no-op — no runtime call, no error.
TEST_F(RBLNFunctionsTest, MemcpyV2VMultiEmptyIsNoop) {
  std::vector<c10::rbln::V2VCopyOp> copies;
  EXPECT_NO_THROW(c10::rbln::memcpy_v2v_multi(copies));
}

// Bulk dispatch: many independent slab copies into adjacent dst regions land
// at the right offsets and preserve content. Validates the new
// rbln_memcpy_v2v_multi entrypoint that V2VBatch::submit() now routes through.
TEST_F(RBLNFunctionsTest, MemcpyV2VMultiBasic) {
  constexpr size_t blk = 32;
  constexpr size_t nblk = 64;
  constexpr size_t total = blk * nblk;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 17) % 127);
  }
  std::vector<int8_t> dst_initial(total, 0);

  for (c10::DeviceIndex device_index = 0; device_index < c10::rbln::get_device_count(); ++device_index) {
    c10::rbln::set_device_index(device_index);

    auto* src_rbln = static_cast<int8_t*>(c10::rbln::malloc(device_index, total));
    auto* dst_rbln = static_cast<int8_t*>(c10::rbln::malloc(device_index, total));
    ASSERT_NE(src_rbln, nullptr);
    ASSERT_NE(dst_rbln, nullptr);
    c10::rbln::memcpy_h2v(src_rbln, src_host.data(), total);
    c10::rbln::memcpy_h2v(dst_rbln, dst_initial.data(), total);

    std::vector<c10::rbln::V2VCopyOp> copies;
    copies.reserve(nblk);
    for (size_t i = 0; i < nblk; ++i) {
      copies.push_back({dst_rbln + i * blk, src_rbln + i * blk, blk});
    }
    c10::rbln::memcpy_v2v_multi(copies);

    std::vector<int8_t> dst_host(total);
    c10::rbln::memcpy_v2h(dst_host.data(), dst_rbln, total);
    EXPECT_EQ(dst_host, src_host);

    c10::rbln::free(src_rbln);
    c10::rbln::free(dst_rbln);
  }
}

// nullptr / 0-byte entries are rejected with a c10::Error before reaching the
// runtime — mirrors the per-call memcpy_v2v contract.
TEST_F(RBLNFunctionsTest, MemcpyV2VMultiRejectsInvalidEntries) {
  constexpr size_t n = 16;
  std::vector<int8_t> src_host(n, 7);
  auto* src_rbln = c10::rbln::malloc(0, n);
  auto* dst_rbln = c10::rbln::malloc(0, n);
  c10::rbln::memcpy_h2v(src_rbln, src_host.data(), n);

  EXPECT_THROW(c10::rbln::memcpy_v2v_multi({{dst_rbln, src_rbln, 0}}), c10::Error);
  EXPECT_THROW(c10::rbln::memcpy_v2v_multi({{dst_rbln, nullptr, n}}), c10::Error);
  EXPECT_THROW(c10::rbln::memcpy_v2v_multi({{nullptr, src_rbln, n}}), c10::Error);

  c10::rbln::free(src_rbln);
  c10::rbln::free(dst_rbln);
}

TEST_F(RBLNFunctionsTest, GetUninitializedMemoryInfo) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    const auto data = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(data != nullptr);

    const auto memory_info = c10::rbln::get_memory_info(data);
    EXPECT_EQ(memory_info.torch_device_id, static_cast<uint32_t>(device_index));
    EXPECT_EQ(memory_info.user_dtype, ::rbln::DataType::Undefined);
    EXPECT_EQ(memory_info.user_shape, std::vector<int64_t>{});
    EXPECT_EQ(memory_info.physical_shape, std::vector<int64_t>{});

    c10::rbln::free(data);
  }
}
