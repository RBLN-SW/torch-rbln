#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

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
