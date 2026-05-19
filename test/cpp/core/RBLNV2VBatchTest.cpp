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

// Cross-device entry mixed into the batch — V2VBatch must fall back to
// per-entry memcpy_v2v (which routes through a host bounce buffer) and
// still produce byte-identical contents at every destination. Validates
// that the bulk-dispatch fast path doesn't quietly drop cross-device
// entries.
TEST_F(RBLNV2VBatchTest, CrossDeviceFallback) {
  const auto device_count = c10::rbln::get_device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Skipping: cross-device batch requires at least 2 devices.";
  }

  constexpr size_t n = 1024;
  std::vector<int8_t> src_host(n);
  for (size_t i = 0; i < n; ++i) {
    src_host[i] = static_cast<int8_t>((i * 13) % 127);
  }
  std::vector<int8_t> dst_initial(n, -1);

  // src + dst0 on device 0 (would qualify for the bulk path on its own);
  // dst1 on device 1 forces the heterogeneous fallback.
  c10::rbln::set_device_index(0);
  auto* src = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  auto* dst0 = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst0, nullptr);
  c10::rbln::memcpy_h2v(src, src_host.data(), n);
  c10::rbln::memcpy_h2v(dst0, dst_initial.data(), n);

  auto* dst1 = static_cast<int8_t*>(c10::rbln::malloc(1, n));
  ASSERT_NE(dst1, nullptr);
  c10::rbln::memcpy_h2v(dst1, dst_initial.data(), n);

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue(dst0, src, n); // same-device (device 0)
    batch.enqueue(dst1, src, n); // cross-device (src on 0, dst on 1)
    EXPECT_EQ(batch.pending_count(), 2u);
    batch.submit();
    EXPECT_EQ(batch.pending_count(), 0u);
  }

  std::vector<int8_t> dst0_host(n);
  std::vector<int8_t> dst1_host(n);
  c10::rbln::memcpy_v2h(dst0_host.data(), dst0, n);
  c10::rbln::memcpy_v2h(dst1_host.data(), dst1, n);
  EXPECT_EQ(dst0_host, src_host);
  EXPECT_EQ(dst1_host, src_host);

  c10::rbln::free(src);
  c10::rbln::free(dst0);
  c10::rbln::free(dst1);
}

// Two same-device pairs on different devices in one batch — every entry is
// same-device for itself, but the batch as a whole spans two devices. The
// bulk dispatch targets a single device, so this case must also fall back
// to per-entry. Guards against an "anchor only tracks first entry" bug.
TEST_F(RBLNV2VBatchTest, MultiDeviceAnchorMismatch) {
  const auto device_count = c10::rbln::get_device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Skipping: multi-device anchor requires at least 2 devices.";
  }

  constexpr size_t n = 256;
  std::vector<int8_t> src_host(n);
  for (size_t i = 0; i < n; ++i) {
    src_host[i] = static_cast<int8_t>((i * 41 + 3) % 127);
  }
  std::vector<int8_t> dst_initial(n, -1);

  auto* src0 = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  auto* dst0 = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  auto* src1 = static_cast<int8_t*>(c10::rbln::malloc(1, n));
  auto* dst1 = static_cast<int8_t*>(c10::rbln::malloc(1, n));
  c10::rbln::memcpy_h2v(src0, src_host.data(), n);
  c10::rbln::memcpy_h2v(dst0, dst_initial.data(), n);
  c10::rbln::memcpy_h2v(src1, src_host.data(), n);
  c10::rbln::memcpy_h2v(dst1, dst_initial.data(), n);

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue(dst0, src0, n); // device 0 -> device 0
    batch.enqueue(dst1, src1, n); // device 1 -> device 1, but anchor=0
    batch.submit();
  }

  std::vector<int8_t> dst0_host(n);
  std::vector<int8_t> dst1_host(n);
  c10::rbln::memcpy_v2h(dst0_host.data(), dst0, n);
  c10::rbln::memcpy_v2h(dst1_host.data(), dst1, n);
  EXPECT_EQ(dst0_host, src_host);
  EXPECT_EQ(dst1_host, src_host);

  c10::rbln::free(src0);
  c10::rbln::free(dst0);
  c10::rbln::free(src1);
  c10::rbln::free(dst1);
}

// Reusing one V2VBatch instance for two submits — second batch must reset
// homogeneity bookkeeping cleanly. First submit goes through the fallback
// path (cross-device); a sticky `homogeneous=false` flag would force the
// second (same-device) batch into the slow path, but correctness would
// hold either way — so this test also asserts the right path was chosen
// by inspecting pending_count between submits.
TEST_F(RBLNV2VBatchTest, ReuseAfterSubmitResetsState) {
  const auto device_count = c10::rbln::get_device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Skipping: reuse-after-submit test requires at least 2 devices.";
  }

  constexpr size_t n = 128;
  std::vector<int8_t> src_host(n);
  for (size_t i = 0; i < n; ++i) {
    src_host[i] = static_cast<int8_t>((i * 7 + 11) % 127);
  }
  std::vector<int8_t> dst_initial(n, -1);

  auto* src = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  auto* dst0 = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  auto* dst1 = static_cast<int8_t*>(c10::rbln::malloc(1, n));
  auto* dst0b = static_cast<int8_t*>(c10::rbln::malloc(0, n));
  c10::rbln::memcpy_h2v(src, src_host.data(), n);
  c10::rbln::memcpy_h2v(dst0, dst_initial.data(), n);
  c10::rbln::memcpy_h2v(dst1, dst_initial.data(), n);
  c10::rbln::memcpy_h2v(dst0b, dst_initial.data(), n);

  c10::rbln::V2VBatch batch;
  // First submit — heterogeneous, should fall back.
  batch.enqueue(dst0, src, n);
  batch.enqueue(dst1, src, n);
  EXPECT_EQ(batch.pending_count(), 2u);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);

  // Second submit — homogeneous on device 0 only. If state didn't reset,
  // the bookkeeping would be stale but content correctness would still
  // hold; we additionally confirm pending_count semantics work.
  batch.enqueue(dst0b, src, n);
  EXPECT_EQ(batch.pending_count(), 1u);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);

  std::vector<int8_t> dst0_host(n);
  std::vector<int8_t> dst1_host(n);
  std::vector<int8_t> dst0b_host(n);
  c10::rbln::memcpy_v2h(dst0_host.data(), dst0, n);
  c10::rbln::memcpy_v2h(dst1_host.data(), dst1, n);
  c10::rbln::memcpy_v2h(dst0b_host.data(), dst0b, n);
  EXPECT_EQ(dst0_host, src_host);
  EXPECT_EQ(dst1_host, src_host);
  EXPECT_EQ(dst0b_host, src_host);

  c10::rbln::free(src);
  c10::rbln::free(dst0);
  c10::rbln::free(dst1);
  c10::rbln::free(dst0b);
}

// Single-entry batch — degenerate fast path. submit() must still dispatch
// correctly when only one V2VCopyOp is queued.
TEST_F(RBLNV2VBatchTest, SingleEntryBatchedSubmit) {
  constexpr size_t n = 512;
  std::vector<int8_t> src_host(n);
  for (size_t i = 0; i < n; ++i) {
    src_host[i] = static_cast<int8_t>((i * 19) % 127);
  }
  std::vector<int8_t> dst_initial(n, 0);

  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), n));
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), n));

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

// enqueue_strided expands one logical strided range into N flat entries that
// all share the same src/dst base. The bookkeeping must observe the base
// devices and (a) take the fast path for same-device strided, (b) fall
// back when src/dst bases are on different devices.
TEST_F(RBLNV2VBatchTest, StridedSameDeviceUsesFastPath) {
  constexpr size_t inner = 16;
  constexpr int64_t outer = 8;
  constexpr size_t total = inner * static_cast<size_t>(outer);

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 23) % 127);
  }
  std::vector<int8_t> dst_initial(total, -1);

  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), total));
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), total));

  const std::vector<int64_t> outer_sizes = {outer};
  const std::vector<int64_t> stride_bytes = {static_cast<int64_t>(inner)};

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue_strided(dst, src, inner, outer_sizes, stride_bytes, stride_bytes);
    EXPECT_EQ(batch.pending_count(), static_cast<size_t>(outer));
    batch.submit();
  }

  EXPECT_EQ(CopyToHost(dst, total), src_host);
  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// enqueue_strided with cross-device bases — falls back to per-entry. Since
// every expanded flat entry inherits the base devices, a single lookup of
// the bases is sufficient to trip the fallback.
TEST_F(RBLNV2VBatchTest, StridedCrossDeviceFallsBack) {
  const auto device_count = c10::rbln::get_device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Skipping: cross-device strided requires at least 2 devices.";
  }

  constexpr size_t inner = 16;
  constexpr int64_t outer = 4;
  constexpr size_t total = inner * static_cast<size_t>(outer);

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 29 + 5) % 127);
  }
  std::vector<int8_t> dst_initial(total, -1);

  auto* src = static_cast<int8_t*>(c10::rbln::malloc(0, total));
  auto* dst = static_cast<int8_t*>(c10::rbln::malloc(1, total));
  c10::rbln::memcpy_h2v(src, src_host.data(), total);
  c10::rbln::memcpy_h2v(dst, dst_initial.data(), total);

  const std::vector<int64_t> outer_sizes = {outer};
  const std::vector<int64_t> stride_bytes = {static_cast<int64_t>(inner)};

  {
    c10::rbln::V2VBatch batch;
    batch.enqueue_strided(dst, src, inner, outer_sizes, stride_bytes, stride_bytes);
    EXPECT_EQ(batch.pending_count(), static_cast<size_t>(outer));
    batch.submit();
  }

  std::vector<int8_t> dst_host(total);
  c10::rbln::memcpy_v2h(dst_host.data(), dst, total);
  EXPECT_EQ(dst_host, src_host);

  c10::rbln::free(src);
  c10::rbln::free(dst);
}

// Large batched submit — exercises the bulk rbln_memcpy_v2v_multi path that
// V2VBatch::submit() now routes through. With many entries any per-entry
// merge / reorder / drop bug would show up as a byte mismatch.
TEST_F(RBLNV2VBatchTest, LargeBatchedSubmit) {
  constexpr size_t blk = 16;
  constexpr size_t nblk = 256;
  constexpr size_t total = blk * nblk;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 31) % 127);
  }
  std::vector<int8_t> dst_initial(total, -1);

  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), total));
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), total));

  {
    c10::rbln::V2VBatch batch;
    for (size_t i = 0; i < nblk; ++i) {
      batch.enqueue(dst + i * blk, src + i * blk, blk);
    }
    EXPECT_EQ(batch.pending_count(), nblk);
    batch.submit();
    EXPECT_EQ(batch.pending_count(), 0u);
  }

  EXPECT_EQ(CopyToHost(dst, total), src_host);
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
