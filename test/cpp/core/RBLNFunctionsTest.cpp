#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

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

TEST_F(RBLNFunctionsTest, MallocAndFree) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    const auto data = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(data != nullptr);
    c10::rbln::free(data);

    // Double free
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
    EXPECT_THROW(c10::rbln::free(data), c10::Error);
  }
}

TEST_F(RBLNFunctionsTest, MallocInvalidSize) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    EXPECT_THROW(c10::rbln::malloc(device_index, size_0b_), c10::Error);

    // Every allocation is device memory, so one larger than the device throws at once.
    const auto total_bytes = c10::rbln::mem_get_info(c10::Device(c10::kPrivateUse1, device_index)).second;
    EXPECT_THROW(c10::rbln::malloc(device_index, total_bytes + size_1gib_), c10::Error);
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

TEST_F(RBLNFunctionsTest, GetTorchDeviceId) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    auto* data = static_cast<char*>(c10::rbln::malloc(device_index, size_1gib_));
    EXPECT_TRUE(data != nullptr);

    EXPECT_EQ(c10::rbln::get_torch_device_id(data), device_index);
    // An interior pointer, as a view's data_ptr() is, resolves to the same device.
    EXPECT_EQ(c10::rbln::get_torch_device_id(data + 4096), device_index);

    c10::rbln::free(data);
  }
}

TEST_F(RBLNFunctionsTest, GetTorchDeviceIdRejectsHostPointer) {
  int host = 0;
  EXPECT_THROW(c10::rbln::get_torch_device_id(&host), c10::Error);
}

TEST_F(RBLNFunctionsTest, GetTorchDeviceIdNullPtr) {
  void* data = nullptr;
  EXPECT_THROW(c10::rbln::get_torch_device_id(data), c10::Error);
}

TEST_F(RBLNFunctionsTest, Synchronize) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    // synchronize with no pending transfers should be a no-op
    EXPECT_NO_THROW(c10::rbln::synchronize(device_index));
  }
}

TEST_F(RBLNFunctionsTest, AsyncMemcpyH2VAndV2H) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    constexpr size_t nbytes = 4096;
    std::vector<int8_t> src_cpu(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      src_cpu[i] = static_cast<int8_t>(i % 127);
    }

    auto rbln_data = c10::rbln::malloc(device_index, nbytes);
    EXPECT_TRUE(rbln_data != nullptr);

    // Async H2V
    c10::rbln::memcpy_h2v_async(rbln_data, src_cpu.data(), nbytes);
    c10::rbln::synchronize(device_index);

    // Async V2H
    std::vector<int8_t> dst_cpu(nbytes, 0);
    c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);
    c10::rbln::synchronize(device_index);

    EXPECT_EQ(dst_cpu, src_cpu);

    c10::rbln::free(rbln_data);
  }
}

TEST_F(RBLNFunctionsTest, AsyncMemcpyV2V) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    constexpr size_t nbytes = 4096;
    std::vector<int8_t> src_cpu(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      src_cpu[i] = static_cast<int8_t>((i * 3) % 127);
    }

    auto src_rbln = c10::rbln::malloc(device_index, nbytes);
    auto dst_rbln = c10::rbln::malloc(device_index, nbytes);
    EXPECT_TRUE(src_rbln != nullptr);
    EXPECT_TRUE(dst_rbln != nullptr);

    c10::rbln::memcpy_h2v_async(src_rbln, src_cpu.data(), nbytes);
    c10::rbln::memcpy_v2v_async(dst_rbln, src_rbln, nbytes);

    std::vector<int8_t> dst_cpu(nbytes, 0);
    c10::rbln::memcpy_v2h_async(dst_cpu.data(), dst_rbln, nbytes);
    c10::rbln::synchronize(device_index);

    EXPECT_EQ(dst_cpu, src_cpu);

    c10::rbln::free(src_rbln);
    c10::rbln::free(dst_rbln);
  }
}

// Unaligned-size variants: 4097 is not 64-aligned, so the V2H path takes the host-bounce
// finalize rather than a direct DMA. Exercises the ordering contract the aligned tests skip.
TEST_F(RBLNFunctionsTest, AsyncMemcpyH2VAndV2HUnaligned) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    constexpr size_t nbytes = 4097;
    std::vector<int8_t> src_cpu(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      src_cpu[i] = static_cast<int8_t>(i % 127);
    }

    auto rbln_data = c10::rbln::malloc(device_index, nbytes);
    EXPECT_TRUE(rbln_data != nullptr);

    c10::rbln::memcpy_h2v_async(rbln_data, src_cpu.data(), nbytes);
    c10::rbln::synchronize(device_index);

    std::vector<int8_t> dst_cpu(nbytes, 0);
    c10::rbln::memcpy_v2h_async(dst_cpu.data(), rbln_data, nbytes);
    c10::rbln::synchronize(device_index);

    EXPECT_EQ(dst_cpu, src_cpu);

    c10::rbln::free(rbln_data);
  }
}

TEST_F(RBLNFunctionsTest, AsyncMemcpyV2VUnaligned) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    constexpr size_t nbytes = 4097;
    std::vector<int8_t> src_cpu(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      src_cpu[i] = static_cast<int8_t>((i * 3) % 127);
    }

    auto src_rbln = c10::rbln::malloc(device_index, nbytes);
    auto dst_rbln = c10::rbln::malloc(device_index, nbytes);
    EXPECT_TRUE(src_rbln != nullptr);
    EXPECT_TRUE(dst_rbln != nullptr);

    c10::rbln::memcpy_h2v_async(src_rbln, src_cpu.data(), nbytes);
    c10::rbln::memcpy_v2v_async(dst_rbln, src_rbln, nbytes);

    std::vector<int8_t> dst_cpu(nbytes, 0);
    c10::rbln::memcpy_v2h_async(dst_cpu.data(), dst_rbln, nbytes);
    c10::rbln::synchronize(device_index);

    EXPECT_EQ(dst_cpu, src_cpu);

    c10::rbln::free(src_rbln);
    c10::rbln::free(dst_rbln);
  }
}

// ---------------------------------------------------------------------------
// fill_zeros: a device-side fill of a byte range, ordered on the current stream.
// ---------------------------------------------------------------------------

TEST_F(RBLNFunctionsTest, FillZerosClearsTheWholeRange) {
  const size_t nbytes = 4096;
  std::vector<int8_t> ones(nbytes, 1);
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);
  c10::rbln::memcpy_h2v(rbln_data, ones.data(), nbytes);

  c10::rbln::fill_zeros(rbln_data, nbytes);

  std::vector<int8_t> dst_cpu(nbytes, 1);
  c10::rbln::memcpy_v2h(dst_cpu.data(), rbln_data, nbytes);
  EXPECT_EQ(dst_cpu, std::vector<int8_t>(nbytes, 0));
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, FillZerosLeavesBytesOutsideTheRange) {
  // An interior range, as zero_ on a view passes: only [offset, offset + length) changes.
  const size_t nbytes = 1024;
  const size_t offset = 100;
  const size_t length = 300;
  std::vector<int8_t> src_cpu(nbytes);
  for (size_t i = 0; i < nbytes; ++i) {
    src_cpu[i] = static_cast<int8_t>(i % 127 + 1);
  }
  auto* rbln_data = static_cast<char*>(c10::rbln::malloc(/*device_index=*/0, nbytes));
  ASSERT_NE(rbln_data, nullptr);
  c10::rbln::memcpy_h2v(rbln_data, src_cpu.data(), nbytes);

  c10::rbln::fill_zeros(rbln_data + offset, length);

  std::vector<int8_t> expected = src_cpu;
  std::fill(expected.begin() + offset, expected.begin() + offset + length, 0);
  std::vector<int8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h(dst_cpu.data(), rbln_data, nbytes);
  EXPECT_EQ(dst_cpu, expected);
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, FillZerosOfNoBytesIsNoop) {
  EXPECT_NO_THROW(c10::rbln::fill_zeros(/*rbln_data=*/nullptr, 0));
}

TEST_F(RBLNFunctionsTest, FillZerosRejectsInvalidRanges) {
  EXPECT_THROW(c10::rbln::fill_zeros(/*rbln_data=*/nullptr, 64), c10::Error);
  int host = 0;
  EXPECT_THROW(c10::rbln::fill_zeros(&host, sizeof(host)), c10::Error);
  // A small block lies in a segment of kSmallSegment bytes, so a longer range runs past it.
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, 64);
  ASSERT_NE(rbln_data, nullptr);
  EXPECT_THROW(c10::rbln::fill_zeros(rbln_data, c10::rbln::caching::kSmallSegment + 1), c10::Error);
  c10::rbln::free(rbln_data);
}

// The key layout of mem_get_info_per_chiplet(), on fixed readings rather than a live one
// another process can move between two queries.
TEST(RBLNFunctionsPerChipletMemoryMap, LaysOutEveryChipletOfEveryNpu) {
  const std::vector<std::vector<c10::rbln::ChipletMemory>> npus = {
      {{500, 400}, {500, 300}},
      {{2000, 2000}},
  };

  const auto out = c10::rbln::per_chiplet_memory_map(npus);

  const std::map<std::string, uint64_t> expected = {
      {"npu.0.total", 1000},
      {"npu.0.used", 300},
      {"npu.0.free", 700},
      {"npu.0.chiplet.0.total", 500},
      {"npu.0.chiplet.0.used", 100},
      {"npu.0.chiplet.0.free", 400},
      {"npu.0.chiplet.1.total", 500},
      {"npu.0.chiplet.1.used", 200},
      {"npu.0.chiplet.1.free", 300},
      {"npu.1.total", 2000},
      {"npu.1.used", 0},
      {"npu.1.free", 2000},
      {"npu.1.chiplet.0.total", 2000},
      {"npu.1.chiplet.0.used", 0},
      {"npu.1.chiplet.0.free", 2000},
  };
  EXPECT_EQ(out, expected);
}

TEST(RBLNFunctionsPerChipletMemoryMap, NoRepliesIsEmpty) {
  EXPECT_TRUE(c10::rbln::per_chiplet_memory_map({}).empty());
}
