#include <c10/core/Device.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHostBatch.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/detail/RBLNCopyBatchImpl.h>

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace c10::rbln {

namespace {

// Work one bulk call may carry. Past it the runtime's job never completes: the
// device reports SYS_KERNEL_TIMEOUT and cancels the context, so the per-entry
// retry below cannot recover either. Measured on lmcache's D2H store: 24
// entries / 12 MiB pass, 32 / 16 MiB start timing out. See the commit message.
constexpr size_t kMaxBulkEntries = 16;
constexpr size_t kMaxBulkBytes = size_t{8} * 1024 * 1024;

/**
 * @brief Submit one group as a single bulk call, per-entry on rejection.
 *
 * Mirrors V2VBatch::submit's retry: the bulk entrypoint enforces stricter
 * inter-copy invariants than the single-copy path, so a geometry the per-entry
 * path handles fine can still be refused in bulk. Replaying per entry is safe
 * because a copy is idempotent — it is a plain write of `src` bytes over `dst`,
 * never a read-modify-write — so an entry that the failed bulk call had already
 * applied simply gets the same bytes written again. That is the only reason a
 * retry is sound under an API that documents no rollback.
 */
template <typename Desc, typename BulkFn, typename OneFn>
void submit_one_call(const std::vector<Desc>& group, const char* who, const BulkFn& bulk, const OneFn& one) {
  try {
    bulk(group);
    return;
  } catch (const c10::Error& e) {
    // PROFILER (cold branch): the batched host<->device entrypoint refused this
    // group; the per-entry loop below has no inter-copy constraint.
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kHostBatchToPerEntry, 0);
    RBLN_LOG_WARN(
        "{}::submit batched path rejected ({} entries) — falling back to per-entry: {}",
        who,
        group.size(),
        e.what_without_backtrace());
  }
  for (const auto& e : group) {
    one(e);
  }
}

/**
 * @brief Flush one homogeneous group, in as few bulk calls as the cap allows.
 */
template <typename Desc, typename BulkFn, typename OneFn>
void flush_group(const std::vector<Desc>& group, const char* who, const BulkFn& bulk, const OneFn& one) {
  if (group.empty()) {
    return;
  }
  size_t total = 0;
  for (const auto& e : group) {
    total += e.nbytes;
  }
  if (group.size() <= kMaxBulkEntries && total <= kMaxBulkBytes) {
    submit_one_call(group, who, bulk, one);
    return;
  }
  RBLN_LOG_DEBUG(
      "{}::submit splitting {} entries / {} bytes to stay under the cap ({} entries, {} bytes)",
      who,
      group.size(),
      total,
      kMaxBulkEntries,
      kMaxBulkBytes);
  for (size_t i = 0; i < group.size();) {
    size_t j = i + 1; // one entry always goes, however large
    size_t bytes = group[i].nbytes;
    while (j < group.size() && j - i < kMaxBulkEntries && bytes + group[j].nbytes <= kMaxBulkBytes) {
      bytes += group[j].nbytes;
      ++j;
    }
    submit_one_call(
        std::vector<Desc>(group.begin() + static_cast<ptrdiff_t>(i), group.begin() + static_cast<ptrdiff_t>(j)),
        who,
        bulk,
        one);
    i = j;
  }
}

/**
 * @brief Submit every entry, one bulk call per anchoring device.
 *
 * The homogeneous case — by far the common one — submits the pending vector as
 * is, with no copy and no per-entry device lookup. Only a genuinely mixed batch
 * pays the grouping pass, and it still gets one bulk call per device rather than
 * the host bounce a cross-device v2v would need.
 *
 * @param device_of Returns the anchoring (device-side) pointer of an entry.
 */
template <typename Desc, typename AnchorFn, typename BulkFn, typename OneFn>
void submit_grouped(
    detail::BatchState<Desc>& st,
    const char* who,
    const AnchorFn& device_of,
    const BulkFn& bulk,
    const OneFn& one) {
  if (st.homogeneous) {
    RBLN_LOG_DEBUG("{}::submit draining {} entries (single device)", who, st.pending.size());
    flush_group(st.pending, who, bulk, one);
    return;
  }
  // Ordered map so the submit order is deterministic by device index; a batch
  // spanning devices is rare enough that the log line is worth more than the
  // constant factor.
  std::map<c10::DeviceIndex, std::vector<Desc>> by_device;
  for (const auto& e : st.pending) {
    by_device[get_torch_device_id(device_of(e))].push_back(e);
  }
  RBLN_LOG_DEBUG(
      "{}::submit draining {} entries across {} devices (split submit)", who, st.pending.size(), by_device.size());
  for (const auto& [dev, group] : by_device) {
    flush_group(group, who, bulk, one);
  }
}

constexpr const char* kH2VWho = "H2VBatch";
constexpr const char* kV2HWho = "V2HBatch";

} // namespace

// ---------------------------------------------------------------------------
// H2VBatch — host source, device destination. Only dst anchors the device.
// ---------------------------------------------------------------------------

struct H2VBatch::Impl {
  detail::BatchState<H2VCopyOp> st;
};

H2VBatch::H2VBatch() : impl_(std::make_unique<Impl>()) {}

H2VBatch::~H2VBatch() {
  if (impl_) {
    detail::warn_if_unsubmitted(impl_->st, kH2VWho);
  }
}

void H2VBatch::enqueue(void* dst, const void* src, size_t nbytes) {
  detail::enqueue_one<H2VCopyOp, detail::DeviceAnchor::kDstOnly>(impl_->st, kH2VWho, dst, src, nbytes);
}

void H2VBatch::enqueue_strided(
    void* dst,
    const void* src,
    size_t inner_block_bytes,
    c10::IntArrayRef outer_sizes,
    c10::IntArrayRef src_byte_strides,
    c10::IntArrayRef dst_byte_strides) {
  detail::enqueue_strided_impl<H2VCopyOp, detail::DeviceAnchor::kDstOnly>(
      impl_->st, kH2VWho, dst, src, inner_block_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
}

void H2VBatch::keep_alive(std::shared_ptr<void> holder) {
  if (holder) {
    impl_->st.keepalive.push_back(std::move(holder));
  }
}

void H2VBatch::submit() {
  if (!impl_ || impl_->st.pending.empty()) {
    // Still drop any holders: a caller that registered one and then enqueued
    // nothing should not keep the buffer alive until the batch dies.
    if (impl_) {
      impl_->st.reset();
    }
    return;
  }
  struct ResetGuard {
    detail::BatchState<H2VCopyOp>* st;
    ~ResetGuard() noexcept {
      st->reset();
    }
  } guard{&impl_->st};

  submit_grouped(
      impl_->st,
      kH2VWho,
      [](const H2VCopyOp& e) { return e.dst; },
      [](const std::vector<H2VCopyOp>& g) { memcpy_h2v_multi(g); },
      [](const H2VCopyOp& e) { memcpy_h2v(e.dst, e.src, e.nbytes); });
}

size_t H2VBatch::pending_count() const {
  return impl_ ? impl_->st.pending.size() : 0;
}

// ---------------------------------------------------------------------------
// V2HBatch — device source, host destination. Only src anchors the device.
// ---------------------------------------------------------------------------

struct V2HBatch::Impl {
  detail::BatchState<V2HCopyOp> st;
};

V2HBatch::V2HBatch() : impl_(std::make_unique<Impl>()) {}

V2HBatch::~V2HBatch() {
  if (impl_) {
    detail::warn_if_unsubmitted(impl_->st, kV2HWho);
  }
}

void V2HBatch::enqueue(void* dst, const void* src, size_t nbytes) {
  detail::enqueue_one<V2HCopyOp, detail::DeviceAnchor::kSrcOnly>(impl_->st, kV2HWho, dst, src, nbytes);
}

void V2HBatch::enqueue_strided(
    void* dst,
    const void* src,
    size_t inner_block_bytes,
    c10::IntArrayRef outer_sizes,
    c10::IntArrayRef src_byte_strides,
    c10::IntArrayRef dst_byte_strides) {
  detail::enqueue_strided_impl<V2HCopyOp, detail::DeviceAnchor::kSrcOnly>(
      impl_->st, kV2HWho, dst, src, inner_block_bytes, outer_sizes, src_byte_strides, dst_byte_strides);
}

void V2HBatch::keep_alive(std::shared_ptr<void> holder) {
  if (holder) {
    impl_->st.keepalive.push_back(std::move(holder));
  }
}

void V2HBatch::submit() {
  if (!impl_ || impl_->st.pending.empty()) {
    if (impl_) {
      impl_->st.reset();
    }
    return;
  }
  struct ResetGuard {
    detail::BatchState<V2HCopyOp>* st;
    ~ResetGuard() noexcept {
      st->reset();
    }
  } guard{&impl_->st};

  submit_grouped(
      impl_->st,
      kV2HWho,
      [](const V2HCopyOp& e) { return e.src; },
      [](const std::vector<V2HCopyOp>& g) { memcpy_v2h_multi(g); },
      [](const V2HCopyOp& e) { memcpy_v2h(e.dst, e.src, e.nbytes); });
}

size_t V2HBatch::pending_count() const {
  return impl_ ? impl_->st.pending.size() : 0;
}

} // namespace c10::rbln
