#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNV2VBatch.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

// V2VBatch unit tests — direct coverage of the buffer/submit abstraction
// without going through the engine. These verify the contract every future
// rebel API backend (flat / batched / strided) must preserve.

class RBLNV2VBatchTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
    ASSERT_GE(c10::rbln::get_device_count(), 1);
  }

  void SetUp() override {
    c10::rbln::set_device_index(device_index_);
    ASSERT_EQ(c10::rbln::get_device_index(), device_index_);
  }

  // Allocate `nbytes` of device memory, h2v-copy a host buffer onto it, and
  // return the device pointer. Caller frees with c10::rbln::free.
  void* AllocAndCopyFromHost(const void* host_data, size_t nbytes) {
    auto* dev = c10::rbln::malloc(device_index_, nbytes);
    EXPECT_NE(dev, nullptr);
    c10::rbln::memcpy_h2v(dev, host_data, nbytes);
    return dev;
  }

  // Pull `nbytes` of device memory into a freshly allocated host vector.
  std::vector<int8_t> CopyToHost(const void* dev, size_t nbytes) {
    std::vector<int8_t> out(nbytes);
    c10::rbln::memcpy_v2h(out.data(), dev, nbytes);
    return out;
  }

  const c10::DeviceIndex device_index_ = 0;
};

// Empty batch: submit() is idempotent and pending_count returns 0.
TEST_F(RBLNV2VBatchTest, EmptyBatchSubmit) {
  c10::rbln::V2VBatch batch;
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
}

// enqueue(0 bytes) is a no-op — runtime would reject nbytes==0 v2v.
TEST_F(RBLNV2VBatchTest, EnqueueZeroBytesNoOp) {
  constexpr size_t n = 16;
  std::vector<int8_t> src_host(n, 7);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::V2VBatch batch;
  batch.enqueue(dst, src, 0);
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();

  auto dst_after = CopyToHost(dst, n);
  // Untouched.
  EXPECT_EQ(dst_after, dst_initial);

  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Single flat enqueue + submit copies a contig slab.
TEST_F(RBLNV2VBatchTest, SingleFlatEnqueue) {
  constexpr size_t n = 1024;
  std::vector<int8_t> src_host(n);
  std::iota(src_host.begin(), src_host.end(), 0);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue(dst, src, n);
    EXPECT_EQ(batch.pending_count(), 1u);
    batch.submit();
    EXPECT_EQ(batch.pending_count(), 0u);
  }

  EXPECT_EQ(CopyToHost(dst, n), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Multiple flat enqueues into adjacent dst slots — verifies submit doesn't
// reorder or merge entries incorrectly.
TEST_F(RBLNV2VBatchTest, MultipleFlatEnqueues) {
  constexpr size_t blk = 32;
  constexpr size_t nblk = 8;
  constexpr size_t total = blk * nblk;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i)
    src_host[i] = static_cast<int8_t>(i % 127);
  std::vector<int8_t> dst_initial(total, 0);

  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), total));
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), total));

  {
    c10::rbln::V2VBatch batch;
    for (size_t i = 0; i < nblk; ++i) {
      batch.enqueue(dst + i * blk, src + i * blk, blk);
    }
    EXPECT_EQ(batch.pending_count(), nblk);
    batch.submit();
  }

  EXPECT_EQ(CopyToHost(dst, total), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// enqueue_strided with empty outer_sizes degenerates to a single flat
// enqueue.
TEST_F(RBLNV2VBatchTest, StridedEmptyOuterIsFlat) {
  constexpr size_t n = 64;
  std::vector<int8_t> src_host(n, 0x42);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::V2VBatch batch;
  batch.enqueue_strided(dst, src, n, {}, {}, {});
  EXPECT_EQ(batch.pending_count(), 1u);
  batch.submit();

  EXPECT_EQ(CopyToHost(dst, n), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// enqueue_strided expands a 2-D outer range correctly: dst is row-stride dst,
// src is row-stride src. With no broadcast each row is copied once.
TEST_F(RBLNV2VBatchTest, StridedTwoDimRowMajor) {
  constexpr int64_t rows = 4;
  constexpr int64_t cols = 8;
  constexpr size_t total = static_cast<size_t>(rows * cols);

  std::vector<int8_t> src_host(total);
  std::iota(src_host.begin(), src_host.end(), 1);
  std::vector<int8_t> dst_initial(total, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), total);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), total);

  std::vector<int64_t> outer_sizes = {rows};
  std::vector<int64_t> src_bs = {cols};
  std::vector<int64_t> dst_bs = {cols};

  c10::rbln::V2VBatch batch;
  batch.enqueue_strided(dst, src, cols, outer_sizes, src_bs, dst_bs);
  EXPECT_EQ(batch.pending_count(), static_cast<size_t>(rows));
  batch.submit();

  EXPECT_EQ(CopyToHost(dst, total), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// stride==0 broadcast: same src slab replicated across `rows` dst slabs.
TEST_F(RBLNV2VBatchTest, StridedBroadcastStrideZero) {
  constexpr int64_t rows = 5;
  constexpr int64_t cols = 4;
  constexpr size_t dst_total = static_cast<size_t>(rows * cols);

  std::vector<int8_t> src_host(cols);
  for (int64_t i = 0; i < cols; ++i)
    src_host[i] = static_cast<int8_t>(10 + i);

  std::vector<int8_t> dst_initial(dst_total, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), cols);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), dst_total);

  std::vector<int64_t> outer_sizes = {rows};
  std::vector<int64_t> src_bs = {0}; // broadcast
  std::vector<int64_t> dst_bs = {cols};

  c10::rbln::V2VBatch batch;
  batch.enqueue_strided(dst, src, cols, outer_sizes, src_bs, dst_bs);
  batch.submit();

  std::vector<int8_t> expected(dst_total);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c)
      expected[r * cols + c] = src_host[c];
  }
  EXPECT_EQ(CopyToHost(dst, dst_total), expected);

  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Dropping a batch without calling submit() does NOT issue backend calls —
// the destructor is leak-prevention only (warns in dev, never throws). The
// dst slab stays at its pre-batch contents.
TEST_F(RBLNV2VBatchTest, DestructorDoesNotFlush) {
  constexpr size_t n = 64;
  std::vector<int8_t> src_host(n);
  std::iota(src_host.begin(), src_host.end(), 1);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue(dst, src, n);
    // Deliberately no submit().
  }

  // dst unchanged: dtor must not have drained the queue.
  EXPECT_EQ(CopyToHost(dst, n), dst_initial);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// The destructor must NOT throw even when entries are pending during stack
// unwind — otherwise a backend rejection layered on an in-flight exception
// would terminate the process.
TEST_F(RBLNV2VBatchTest, DestructorIsNoexceptDuringUnwind) {
  constexpr size_t n = 32;
  std::vector<int8_t> src_host(n, 1);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  bool caught = false;
  try {
    c10::rbln::V2VBatch batch;
    batch.enqueue(dst, src, n);
    throw std::runtime_error("simulated mid-scope error");
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);
  // dst stays untouched — dtor never submitted on the unwind path.
  EXPECT_EQ(CopyToHost(dst, n), dst_initial);

  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Double-submit: second submit is a no-op (queue already drained).
TEST_F(RBLNV2VBatchTest, DoubleSubmit) {
  constexpr size_t n = 64;
  std::vector<int8_t> src_host(n, 9);
  std::vector<int8_t> dst_initial(n, 0);

  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::V2VBatch batch;
  batch.enqueue(dst, src, n);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);

  EXPECT_EQ(CopyToHost(dst, n), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Mismatched strided lengths must throw a c10 error (input validation).
TEST_F(RBLNV2VBatchTest, StridedLengthMismatchThrows) {
  constexpr size_t n = 16;
  std::vector<int8_t> src_host(n, 0);
  std::vector<int8_t> dst_initial(n, 0);
  void* src = AllocAndCopyFromHost(src_host.data(), n);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::V2VBatch batch;
  std::vector<int64_t> outer = {2};
  std::vector<int64_t> two = {1, 2};
  EXPECT_THROW(batch.enqueue_strided(dst, src, 4, outer, two, two), c10::Error);

  c10::rbln::free(src);
  c10::rbln::free(dst);
}
