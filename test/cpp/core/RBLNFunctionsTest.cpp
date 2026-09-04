#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

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

TEST_F(RBLNFunctionsTest, ToScalarTypeInvertsToRblnDtype) {
  for (const auto scalar_type :
       {c10::kBool,
        c10::kByte,
        c10::kChar,
        c10::kShort,
        c10::kInt,
        c10::kLong,
        c10::kHalf,
        c10::kFloat,
        c10::kDouble,
        c10::kComplexHalf,
        c10::kComplexFloat,
        c10::kComplexDouble,
        c10::kBFloat16,
        c10::kFloat8_e5m2,
        c10::kFloat8_e4m3fn}) {
    EXPECT_EQ(c10::rbln::to_scalar_type(c10::rbln::to_rbln_dtype(scalar_type)), scalar_type);
  }
  EXPECT_FALSE(c10::rbln::to_scalar_type(::rbln::DataType::Undefined).has_value());
  EXPECT_FALSE(c10::rbln::to_scalar_type(::rbln::DataType::CustomFloat16).has_value());
}

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
  // is_eager_malloc() reads the env on every call, so toggling it is observed each time.
  // A process-lifetime cache would latch the first value (the xdist cross-test leak this
  // guards against), so assert set -> unset -> set flips the result live.
  const char* saved = std::getenv("TORCH_RBLN_EAGER_MALLOC");
  const bool had = saved != nullptr;
  const std::string saved_str = had ? saved : "";

  setenv("TORCH_RBLN_EAGER_MALLOC", "1", /*overwrite=*/1);
  EXPECT_TRUE(c10::rbln::is_eager_malloc());
  unsetenv("TORCH_RBLN_EAGER_MALLOC");
  EXPECT_FALSE(c10::rbln::is_eager_malloc());
  setenv("TORCH_RBLN_EAGER_MALLOC", "1", /*overwrite=*/1);
  EXPECT_TRUE(c10::rbln::is_eager_malloc());
  setenv("TORCH_RBLN_EAGER_MALLOC", "0", /*overwrite=*/1); // only exactly "1" is eager
  EXPECT_FALSE(c10::rbln::is_eager_malloc());

  if (had) {
    setenv("TORCH_RBLN_EAGER_MALLOC", saved_str.c_str(), /*overwrite=*/1);
  } else {
    unsetenv("TORCH_RBLN_EAGER_MALLOC");
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

TEST_F(RBLNFunctionsTest, GetTorchDeviceId) {
  const auto device_count = c10::rbln::get_device_count();
  EXPECT_GE(device_count, 1);
  for (c10::DeviceIndex device_index = 0; device_index < device_count; ++device_index) {
    const auto data = c10::rbln::malloc(device_index, size_1gib_);
    EXPECT_TRUE(data != nullptr);

    // The lightweight helper must report the owning device...
    EXPECT_EQ(c10::rbln::get_torch_device_id(data), device_index);
    // ...and agree with the full get_memory_info() round-trip it replaces.
    EXPECT_EQ(
        c10::rbln::get_torch_device_id(data),
        static_cast<c10::DeviceIndex>(c10::rbln::get_memory_info(data).torch_device_id));

    c10::rbln::free(data);
  }
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
// borrow_host_ptr / acquire_host_ptr_for_overwrite / return_borrowed.
// Covers the round-trip happy path, host-write-back semantics, the
// overwrite-acquire variant (no D2H sync), and the input-validation contracts
// (nullptr / zero size / sentinel borrow_id).
// ---------------------------------------------------------------------------

TEST_F(RBLNFunctionsTest, BorrowHostPtrRoundTrip) {
  // Stage host-side bytes into rbln memory, then read them back via borrow.
  const size_t nbytes = 1024;
  std::vector<int8_t> src_cpu(nbytes, 0x5a);
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);
  c10::rbln::memcpy_h2v(rbln_data, src_cpu.data(), nbytes);

  // Borrow returns a host-readable pointer + a non-zero borrow id.
  const auto borrowed = c10::rbln::borrow_host_ptr(rbln_data, nbytes);
  EXPECT_NE(borrowed.host_ptr, uintptr_t{0});
  EXPECT_NE(borrowed.borrow_id, uint64_t{0});

  // Bytes match what we staged.
  const auto* host_view = reinterpret_cast<const int8_t*>(borrowed.host_ptr);
  for (size_t i = 0; i < nbytes; ++i) {
    EXPECT_EQ(host_view[i], static_cast<int8_t>(0x5a)) << "mismatch at byte " << i;
  }

  c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/false);
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, BorrowHostPtrWriteBackVisibleAfterReturn) {
  // Borrow + mutate host buffer + return(updated=true). Subsequent v2h read
  // must observe the host-side mutation (host view becomes the latest source
  // of truth on return with updated=true).
  const size_t nbytes = 64;
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);

  // Initial state: stage zeros so we have a known device-side baseline.
  std::vector<int8_t> zeros(nbytes, 0);
  c10::rbln::memcpy_h2v(rbln_data, zeros.data(), nbytes);

  {
    const auto borrowed = c10::rbln::borrow_host_ptr(rbln_data, nbytes);
    auto* host_writer = reinterpret_cast<int8_t*>(borrowed.host_ptr);
    for (size_t i = 0; i < nbytes; ++i) {
      host_writer[i] = static_cast<int8_t>(i);
    }
    c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
  }

  std::vector<int8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h(dst_cpu.data(), rbln_data, nbytes);
  for (size_t i = 0; i < nbytes; ++i) {
    EXPECT_EQ(dst_cpu[i], static_cast<int8_t>(i));
  }

  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, AcquireHostPtrForOverwriteRoundTrip) {
  // Acquire-for-overwrite skips the device→host sync; caller must overwrite
  // the entire region. Verify (a) the call returns a valid host pointer and
  // (b) writing through it and returning(updated=true) makes the host bytes
  // visible on subsequent v2h.
  const size_t nbytes = 256;
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);

  const auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_data, nbytes);
  EXPECT_NE(borrowed.host_ptr, uintptr_t{0});
  EXPECT_NE(borrowed.borrow_id, uint64_t{0});

  auto* host_writer = reinterpret_cast<uint8_t*>(borrowed.host_ptr);
  std::memset(host_writer, 0xa5, nbytes);
  c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);

  std::vector<uint8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h(dst_cpu.data(), rbln_data, nbytes);
  for (size_t i = 0; i < nbytes; ++i) {
    EXPECT_EQ(dst_cpu[i], 0xa5);
  }

  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, TryAcquireHostPtrForOverwriteRoundTrip) {
  // Non-throwing variant: on success it behaves like acquire_host_ptr_for_overwrite.
  const size_t nbytes = 256;
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);

  auto borrowed = c10::rbln::try_acquire_host_ptr_for_overwrite(rbln_data, nbytes);
  ASSERT_TRUE(borrowed.has_value());
  EXPECT_NE(borrowed->host_ptr, uintptr_t{0});
  EXPECT_NE(borrowed->borrow_id, uint64_t{0});

  std::memset(reinterpret_cast<uint8_t*>(borrowed->host_ptr), 0xa5, nbytes);
  c10::rbln::return_borrowed(borrowed->borrow_id, /*updated=*/true);

  std::vector<uint8_t> dst_cpu(nbytes, 0);
  c10::rbln::memcpy_v2h(dst_cpu.data(), rbln_data, nbytes);
  for (size_t i = 0; i < nbytes; ++i) {
    EXPECT_EQ(dst_cpu[i], 0xa5);
  }
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, TryAcquireHostPtrForOverwriteInvalidArgsReturnNullopt) {
  // The copy-fallback path in cpu_fallback_rbln relies on invalid/rejected
  // acquires returning nullopt rather than throwing.
  EXPECT_FALSE(c10::rbln::try_acquire_host_ptr_for_overwrite(/*rbln_data=*/nullptr, 64).has_value());
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, 64);
  ASSERT_NE(rbln_data, nullptr);
  EXPECT_FALSE(c10::rbln::try_acquire_host_ptr_for_overwrite(rbln_data, /*nbytes=*/0).has_value());
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, BorrowRejectsNullData) {
  EXPECT_THROW(c10::rbln::borrow_host_ptr(/*rbln_data=*/nullptr, 64), c10::Error);
  EXPECT_THROW(c10::rbln::acquire_host_ptr_for_overwrite(/*rbln_data=*/nullptr, 64), c10::Error);
}

TEST_F(RBLNFunctionsTest, BorrowRejectsZeroSize) {
  // Mirrors memcpy_h2v which also rejects nbytes==0. Callers that have a
  // legitimate zero-byte case must short-circuit before reaching the wrapper.
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, 64);
  ASSERT_NE(rbln_data, nullptr);
  EXPECT_THROW(c10::rbln::borrow_host_ptr(rbln_data, /*nbytes=*/0), c10::Error);
  EXPECT_THROW(c10::rbln::acquire_host_ptr_for_overwrite(rbln_data, /*nbytes=*/0), c10::Error);
  c10::rbln::free(rbln_data);
}

TEST_F(RBLNFunctionsTest, ReturnBorrowedZeroIdIsNoop) {
  // borrow_id == 0 is a sentinel meaning "no live borrow". Cleanup paths in
  // RBLNCPUFallback rely on this so they can call return_borrowed
  // unconditionally over a vector that may contain skipped entries.
  EXPECT_NO_THROW(c10::rbln::return_borrowed(/*borrow_id=*/0, /*updated=*/false));
  EXPECT_NO_THROW(c10::rbln::return_borrowed(/*borrow_id=*/0, /*updated=*/true));
}

TEST_F(RBLNFunctionsTest, ReturnBorrowedDoubleReleaseThrows) {
  // The borrow ledger is single-shot; returning the same id twice must
  // surface as an error.
  const size_t nbytes = 64;
  auto rbln_data = c10::rbln::malloc(/*device_index=*/0, nbytes);
  ASSERT_NE(rbln_data, nullptr);

  const auto borrowed = c10::rbln::borrow_host_ptr(rbln_data, nbytes);
  c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/false);
  EXPECT_THROW(c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/false), c10::Error);

  c10::rbln::free(rbln_data);
}
