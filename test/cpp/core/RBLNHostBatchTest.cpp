#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHostBatch.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// H2VBatch / V2HBatch unit tests, without going through ATen.
//
// Axes mirror RBLNV2VBatchTest (empty / zero-byte / flat / destructor / reuse)
// since the buffering contract is shared, plus the one v2v has no equivalent
// for: heterogeneous batches splitting into one bulk submit per device.
//
// Every case reads the destination back, so a dropped or mis-addressed entry
// fails rather than passing on a plausible pending_count().

namespace {

// A heap block whose deleter records that it ran, so a test can assert whether
// the batch is still holding a reference.
struct FreeFlagHolder {
  static std::shared_ptr<void> make(bool* freed) {
    return std::shared_ptr<void>(new char[1], [freed](void* p) {
      *freed = true;
      delete[] static_cast<char*>(p);
    });
  }
};

} // namespace

class RBLNHostBatchTest : public ::testing::Test {
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

  // Allocate device memory on `dev` and seed it from a host buffer.
  void* AllocAndCopyFromHost(const void* host_data, size_t nbytes, c10::DeviceIndex dev = 0) {
    auto* p = c10::rbln::malloc(dev, nbytes);
    EXPECT_NE(p, nullptr);
    c10::rbln::memcpy_h2v(p, host_data, nbytes);
    return p;
  }

  std::vector<int8_t> CopyToHost(const void* dev, size_t nbytes) {
    std::vector<int8_t> out(nbytes);
    c10::rbln::memcpy_v2h(out.data(), dev, nbytes);
    return out;
  }

  static std::vector<int8_t> Ramp(size_t n, int8_t start = 0) {
    std::vector<int8_t> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
  }

  const c10::DeviceIndex device_index_ = 0;
};

// ---------------------------------------------------------------------------
// H2VBatch — shared buffering contract
// ---------------------------------------------------------------------------

TEST_F(RBLNHostBatchTest, H2VEmptyBatchSubmitIsIdempotent) {
  c10::rbln::H2VBatch batch;
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
}

// enqueue(0 bytes) records nothing — the runtime rejects a zero-byte copy, and
// callers with genuinely empty tensors short-circuit earlier.
TEST_F(RBLNHostBatchTest, H2VEnqueueZeroBytesNoOp) {
  constexpr size_t n = 16;
  const auto src_host = std::vector<int8_t>(n, 7);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::H2VBatch batch;
  batch.enqueue(dst, src_host.data(), 0);
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();

  EXPECT_EQ(CopyToHost(dst, n), dst_initial);
  c10::rbln::free(dst);
}

TEST_F(RBLNHostBatchTest, H2VSingleFlatEnqueue) {
  constexpr size_t n = 1024;
  const auto src_host = Ramp(n);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  {
    c10::rbln::H2VBatch batch;
    batch.enqueue(dst, src_host.data(), n);
    EXPECT_EQ(batch.pending_count(), 1u);
    batch.submit();
    EXPECT_EQ(batch.pending_count(), 0u);
  }

  EXPECT_EQ(CopyToHost(dst, n), src_host);
  c10::rbln::free(dst);
}

// Several independent host sources into disjoint device ranges — the shape the
// bulk entrypoint exists for. Destinations must not overlap (runtime contract).
TEST_F(RBLNHostBatchTest, H2VMultipleEntriesOneSubmit) {
  constexpr size_t chunk = 128;
  constexpr size_t n_chunks = 8;
  constexpr size_t total = chunk * n_chunks;

  std::vector<std::vector<int8_t>> srcs;
  srcs.reserve(n_chunks);
  for (size_t i = 0; i < n_chunks; ++i) {
    srcs.push_back(std::vector<int8_t>(chunk, static_cast<int8_t>(i + 1)));
  }
  const auto dst_initial = std::vector<int8_t>(total, 0);
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), total));

  c10::rbln::H2VBatch batch;
  for (size_t i = 0; i < n_chunks; ++i) {
    batch.enqueue(dst + i * chunk, srcs[i].data(), chunk);
  }
  EXPECT_EQ(batch.pending_count(), n_chunks);
  batch.submit();

  std::vector<int8_t> expected;
  expected.reserve(total);
  for (const auto& s : srcs) {
    expected.insert(expected.end(), s.begin(), s.end());
  }
  EXPECT_EQ(CopyToHost(dst, total), expected);
  c10::rbln::free(dst);
}

// No cap on the entry count: the runtime splits across dispatches internally.
// Wide enough to exceed the v2v per-destination limit, which h2v/v2h do not have.
TEST_F(RBLNHostBatchTest, H2VLargeBatchExceedsV2VCap) {
  constexpr size_t entries = 2048; // > kMaxV2VMultiCopies (1024)
  constexpr size_t chunk = 8;
  constexpr size_t total = entries * chunk;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 31) % 127);
  }
  const auto dst_initial = std::vector<int8_t>(total, -1);
  auto* dst = static_cast<int8_t*>(AllocAndCopyFromHost(dst_initial.data(), total));

  c10::rbln::H2VBatch batch;
  for (size_t i = 0; i < entries; ++i) {
    batch.enqueue(dst + i * chunk, src_host.data() + i * chunk, chunk);
  }
  EXPECT_EQ(batch.pending_count(), entries);
  batch.submit();

  EXPECT_EQ(CopyToHost(dst, total), src_host);
  c10::rbln::free(dst);
}

// ---------------------------------------------------------------------------
// H2VBatch — strided expansion (shared with V2VBatch, so cover the same shapes)
// ---------------------------------------------------------------------------

TEST_F(RBLNHostBatchTest, H2VNullPointerThrows) {
  constexpr size_t n = 8;
  const auto src_host = std::vector<int8_t>(n, 1);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::H2VBatch batch;
  EXPECT_THROW(batch.enqueue(nullptr, src_host.data(), n), c10::Error);
  EXPECT_THROW(batch.enqueue(dst, nullptr, n), c10::Error);
  EXPECT_EQ(batch.pending_count(), 0u);

  c10::rbln::free(dst);
}

// ---------------------------------------------------------------------------
// H2VBatch — lifetime / reuse
// ---------------------------------------------------------------------------

// Dropping a batch without submit() must not issue backend calls: a rejection
// during stack unwind would terminate the process. Leak-prevention only.
TEST_F(RBLNHostBatchTest, H2VDestructorDoesNotFlush) {
  constexpr size_t n = 64;
  const auto src_host = Ramp(n, 1);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  {
    c10::rbln::H2VBatch batch;
    batch.enqueue(dst, src_host.data(), n);
    EXPECT_EQ(batch.pending_count(), 1u);
  }

  EXPECT_EQ(CopyToHost(dst, n), dst_initial);
  c10::rbln::free(dst);
}

TEST_F(RBLNHostBatchTest, H2VDestructorIsNoexceptDuringUnwind) {
  constexpr size_t n = 32;
  const auto src_host = std::vector<int8_t>(n, 1);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  bool caught = false;
  try {
    c10::rbln::H2VBatch batch;
    batch.enqueue(dst, src_host.data(), n);
    throw std::runtime_error("simulated mid-scope error");
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(CopyToHost(dst, n), dst_initial);

  c10::rbln::free(dst);
}

// submit() resets the batch, so the same object can be refilled. A second
// submit with nothing queued is a no-op rather than a replay.
TEST_F(RBLNHostBatchTest, H2VReuseAfterSubmitAndDoubleSubmit) {
  constexpr size_t n = 64;
  const auto first = std::vector<int8_t>(n, 3);
  const auto second = std::vector<int8_t>(n, 9);
  const auto dst_initial = std::vector<int8_t>(n, 0);
  void* dst = AllocAndCopyFromHost(dst_initial.data(), n);

  c10::rbln::H2VBatch batch;
  batch.enqueue(dst, first.data(), n);
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
  EXPECT_EQ(CopyToHost(dst, n), first);

  // Second submit with an empty queue must not re-apply the drained entry.
  batch.submit();
  EXPECT_EQ(CopyToHost(dst, n), first);

  batch.enqueue(dst, second.data(), n);
  batch.submit();
  EXPECT_EQ(CopyToHost(dst, n), second);

  c10::rbln::free(dst);
}

// ---------------------------------------------------------------------------
// V2HBatch — mirror coverage
// ---------------------------------------------------------------------------

TEST_F(RBLNHostBatchTest, V2HEmptyBatchSubmitIsIdempotent) {
  c10::rbln::V2HBatch batch;
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();
  batch.submit();
  EXPECT_EQ(batch.pending_count(), 0u);
}

TEST_F(RBLNHostBatchTest, V2HEnqueueZeroBytesNoOp) {
  constexpr size_t n = 16;
  const auto src_host = Ramp(n, 5);
  void* src = AllocAndCopyFromHost(src_host.data(), n);
  std::vector<int8_t> dst(n, 0);
  const auto dst_initial = dst;

  c10::rbln::V2HBatch batch;
  batch.enqueue(dst.data(), src, 0);
  EXPECT_EQ(batch.pending_count(), 0u);
  batch.submit();

  EXPECT_EQ(dst, dst_initial);
  c10::rbln::free(src);
}

TEST_F(RBLNHostBatchTest, V2HSingleFlatEnqueue) {
  constexpr size_t n = 1024;
  const auto src_host = Ramp(n);
  void* src = AllocAndCopyFromHost(src_host.data(), n);
  std::vector<int8_t> dst(n, 0);

  {
    c10::rbln::V2HBatch batch;
    batch.enqueue(dst.data(), src, n);
    EXPECT_EQ(batch.pending_count(), 1u);
    batch.submit();
    EXPECT_EQ(batch.pending_count(), 0u);
  }

  EXPECT_EQ(dst, src_host);
  c10::rbln::free(src);
}

// Gather shape: N device slabs read into disjoint regions of one host buffer.
// This is the direction lmcache's KV offload needs, and the one that currently
// has no batched path at all.
TEST_F(RBLNHostBatchTest, V2HGatherIntoOneHostBuffer) {
  constexpr size_t chunk = 96;
  constexpr size_t n_chunks = 10;
  constexpr size_t total = chunk * n_chunks;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 17) % 127);
  }
  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), total));
  std::vector<int8_t> dst(total, -1);

  c10::rbln::V2HBatch batch;
  for (size_t i = 0; i < n_chunks; ++i) {
    batch.enqueue(dst.data() + i * chunk, src + i * chunk, chunk);
  }
  EXPECT_EQ(batch.pending_count(), n_chunks);
  batch.submit();

  EXPECT_EQ(dst, src_host);
  c10::rbln::free(src);
}

// Several entries reading the SAME device range into different host
// destinations. The runtime documents source ranges as pure reads, so this must
// not be rejected as an overlap.
TEST_F(RBLNHostBatchTest, V2HRepeatedSourceRangeIsAllowed) {
  constexpr size_t n = 64;
  const auto src_host = Ramp(n, 2);
  void* src = AllocAndCopyFromHost(src_host.data(), n);

  std::vector<int8_t> dst_a(n, 0);
  std::vector<int8_t> dst_b(n, 0);

  c10::rbln::V2HBatch batch;
  batch.enqueue(dst_a.data(), src, n);
  batch.enqueue(dst_b.data(), src, n);
  batch.submit();

  EXPECT_EQ(dst_a, src_host);
  EXPECT_EQ(dst_b, src_host);
  c10::rbln::free(src);
}

TEST_F(RBLNHostBatchTest, V2HLargeBatchExceedsV2VCap) {
  constexpr size_t entries = 2048; // > kMaxV2VMultiCopies (1024)
  constexpr size_t chunk = 8;
  constexpr size_t total = entries * chunk;

  std::vector<int8_t> src_host(total);
  for (size_t i = 0; i < total; ++i) {
    src_host[i] = static_cast<int8_t>((i * 29) % 127);
  }
  auto* src = static_cast<int8_t*>(AllocAndCopyFromHost(src_host.data(), total));
  std::vector<int8_t> dst(total, -1);

  c10::rbln::V2HBatch batch;
  for (size_t i = 0; i < entries; ++i) {
    batch.enqueue(dst.data() + i * chunk, src + i * chunk, chunk);
  }
  batch.submit();

  EXPECT_EQ(dst, src_host);
  c10::rbln::free(src);
}

TEST_F(RBLNHostBatchTest, V2HDestructorDoesNotFlush) {
  constexpr size_t n = 64;
  const auto src_host = Ramp(n, 4);
  void* src = AllocAndCopyFromHost(src_host.data(), n);
  std::vector<int8_t> dst(n, 0);
  const auto dst_initial = dst;

  {
    c10::rbln::V2HBatch batch;
    batch.enqueue(dst.data(), src, n);
    EXPECT_EQ(batch.pending_count(), 1u);
  }

  EXPECT_EQ(dst, dst_initial);
  c10::rbln::free(src);
}

// ---------------------------------------------------------------------------
// Heterogeneous batches — split submit, not a host bounce (no v2v equivalent)
// ---------------------------------------------------------------------------

// One h2v batch spanning two devices. Host memory is reachable from both, so the
// batch splits per destination device and every slab must still land. A dropped
// group would leave one destination at its seed value.
TEST_F(RBLNHostBatchTest, H2VCrossDeviceSplitSubmit) {
  if (c10::rbln::get_device_count() < 2) {
    GTEST_SKIP() << "cross-device batch requires at least 2 devices";
  }
  constexpr size_t n = 512;
  const auto src0 = std::vector<int8_t>(n, 21);
  const auto src1 = std::vector<int8_t>(n, 42);
  const auto seed = std::vector<int8_t>(n, -1);

  c10::rbln::set_device_index(0);
  void* dst0 = AllocAndCopyFromHost(seed.data(), n, 0);
  void* dst1 = AllocAndCopyFromHost(seed.data(), n, 1);

  {
    c10::rbln::H2VBatch batch;
    batch.enqueue(dst0, src0.data(), n);
    batch.enqueue(dst1, src1.data(), n);
    EXPECT_EQ(batch.pending_count(), 2u);
    batch.submit();
  }

  EXPECT_EQ(CopyToHost(dst0, n), src0);
  EXPECT_EQ(CopyToHost(dst1, n), src1);

  c10::rbln::free(dst0);
  c10::rbln::free(dst1);
  c10::rbln::set_device_index(0);
}

// Same for the read direction: sources on two devices gathered into one host
// buffer's disjoint halves.
TEST_F(RBLNHostBatchTest, V2HCrossDeviceSplitSubmit) {
  if (c10::rbln::get_device_count() < 2) {
    GTEST_SKIP() << "cross-device batch requires at least 2 devices";
  }
  constexpr size_t n = 512;
  const auto host0 = std::vector<int8_t>(n, 33);
  const auto host1 = std::vector<int8_t>(n, 77);

  c10::rbln::set_device_index(0);
  void* src0 = AllocAndCopyFromHost(host0.data(), n, 0);
  void* src1 = AllocAndCopyFromHost(host1.data(), n, 1);

  std::vector<int8_t> dst(2 * n, -1);
  {
    c10::rbln::V2HBatch batch;
    batch.enqueue(dst.data(), src0, n);
    batch.enqueue(dst.data() + n, src1, n);
    batch.submit();
  }

  std::vector<int8_t> expected;
  expected.insert(expected.end(), host0.begin(), host0.end());
  expected.insert(expected.end(), host1.begin(), host1.end());
  EXPECT_EQ(dst, expected);

  c10::rbln::free(src0);
  c10::rbln::free(src1);
  c10::rbln::set_device_index(0);
}

// ---------------------------------------------------------------------------
// Composite scenario — the shape an ATen consumer produces
// ---------------------------------------------------------------------------

// Round-trip a paged layout: scatter N host chunks into a strided device buffer
// with H2VBatch, gather it back with V2HBatch, compare against an explicit
// expected buffer (not a self-comparison, so a matching offset error in both
// directions still fails). Same geometry the KV offload path uses.
TEST_F(RBLNHostBatchTest, ScatterThenGatherRoundTrip) {
  constexpr int64_t blocks = 5;
  constexpr int64_t layers = 3;
  constexpr int64_t block_bytes = 64;
  constexpr int64_t dev_pitch = 96; // device blocks padded
  constexpr size_t dev_total = static_cast<size_t>(blocks * layers * dev_pitch);
  constexpr size_t host_total = static_cast<size_t>(blocks * layers * block_bytes);

  std::vector<int8_t> host_src(host_total);
  for (size_t i = 0; i < host_total; ++i) {
    host_src[i] = static_cast<int8_t>((i * 23 + 5) % 127);
  }
  const auto seed = std::vector<int8_t>(dev_total, -1);
  auto* dev = static_cast<int8_t*>(AllocAndCopyFromHost(seed.data(), dev_total));

  // Scatter: one entry per (block, layer), host contiguous -> device padded.
  {
    c10::rbln::H2VBatch batch;
    for (int64_t b = 0; b < blocks; ++b) {
      for (int64_t l = 0; l < layers; ++l) {
        const auto idx = b * layers + l;
        batch.enqueue(dev + idx * dev_pitch, host_src.data() + idx * block_bytes, static_cast<size_t>(block_bytes));
      }
    }
    EXPECT_EQ(batch.pending_count(), static_cast<size_t>(blocks * layers));
    batch.submit();
  }

  // Gather back into a fresh host buffer.
  std::vector<int8_t> host_dst(host_total, 0);
  {
    c10::rbln::V2HBatch batch;
    for (int64_t b = 0; b < blocks; ++b) {
      for (int64_t l = 0; l < layers; ++l) {
        const auto idx = b * layers + l;
        batch.enqueue(host_dst.data() + idx * block_bytes, dev + idx * dev_pitch, static_cast<size_t>(block_bytes));
      }
    }
    batch.submit();
  }

  EXPECT_EQ(host_dst, host_src);

  // The padding between device blocks must be untouched by the scatter.
  const auto dev_after = CopyToHost(dev, dev_total);
  for (int64_t idx = 0; idx < blocks * layers; ++idx) {
    for (int64_t off = block_bytes; off < dev_pitch; ++off) {
      EXPECT_EQ(dev_after[static_cast<size_t>(idx * dev_pitch + off)], -1)
          << "scatter wrote past the payload at block-slot " << idx << " offset " << off;
    }
  }

  c10::rbln::free(dev);
}
