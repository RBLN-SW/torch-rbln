#include <ATen/native/rbln/RBLNStridedHostCopy.h>

#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace at::native::rbln {

namespace {

/**
 * @brief Byte-level description of one strided tensor pair.
 *
 * `inner_block_bytes` is the largest run contiguous in both tensors; the outer
 * vectors describe the iteration over everything above it. An empty
 * `outer_sizes` means the whole copy is a single slab.
 */
struct StridedPlan {
  size_t inner_block_bytes = 0;
  std::vector<int64_t> outer_sizes;
  std::vector<int64_t> src_byte_strides;
  std::vector<int64_t> dst_byte_strides;
  bool flat = false; // both sides fully contiguous — one slab, no outer walk
};

/**
 * @brief Translate a (dst, src) tensor pair into byte offsets and strides.
 *
 * Deliberately identical to the analysis in strided_v2v_copy: which side lives
 * in host memory does not enter the stride arithmetic, so h2v/v2h/v2v all want
 * the same plan for the same geometry. Keeping it in one helper is also what
 * makes the descriptor counts match across directions, which matters because a
 * consumer switching from one to the other should not silently change its
 * dispatch count.
 */
StridedPlan plan_strided(const at::Tensor& dst, const at::Tensor& src) {
  StridedPlan plan;
  const auto rank = dst.dim();
  const auto elm = static_cast<size_t>(dst.element_size());

  // Fast path: both fully contiguous → single copy of the whole payload.
  if (dst.is_contiguous() && src.is_contiguous()) {
    plan.flat = true;
    plan.inner_block_bytes = static_cast<size_t>(dst.numel()) * elm;
    return plan;
  }

  // 0-D is covered by is_contiguous() above for both sides; defend anyway in
  // case contiguity semantics for 0-D ever change.
  if (rank == 0) {
    plan.flat = true;
    plan.inner_block_bytes = elm;
    return plan;
  }

  const auto sizes = dst.sizes();
  const auto src_strides = src.strides();
  const auto dst_strides = dst.strides();

  const int64_t inner_start = common_inner_start(sizes, src_strides, dst_strides);

  // Inner block = product of dims [inner_start, rank). May be 1 element when no
  // joint contiguous suffix exists — correct, just the slowest shape.
  int64_t inner_block_elems = 1;
  for (int64_t i = inner_start; i < rank; ++i) {
    inner_block_elems *= sizes[i];
  }
  plan.inner_block_bytes = static_cast<size_t>(inner_block_elems) * elm;

  // Outer description over dims [0, inner_start). Byte strides; a stride of 0
  // (broadcast) is preserved verbatim so the same source bytes replicate across
  // writes rather than collapsing into one.
  const int64_t elm_signed = static_cast<int64_t>(elm);
  plan.outer_sizes.assign(sizes.begin(), sizes.begin() + inner_start);
  plan.src_byte_strides.resize(static_cast<size_t>(inner_start));
  plan.dst_byte_strides.resize(static_cast<size_t>(inner_start));
  for (int64_t i = 0; i < inner_start; ++i) {
    plan.src_byte_strides[static_cast<size_t>(i)] = src_strides[i] * elm_signed;
    plan.dst_byte_strides[static_cast<size_t>(i)] = dst_strides[i] * elm_signed;
  }
  return plan;
}

/** @brief Shape / dtype / numel preconditions common to both directions. */
void check_pair(const at::Tensor& dst, const at::Tensor& src, const char* who) {
  RBLN_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "{}: dtype mismatch (dst={} src={}) — this primitive does not convert",
      who,
      c10::str(dst.scalar_type()),
      c10::str(src.scalar_type()));
  RBLN_CHECK(
      dst.sizes() == src.sizes(),
      "{}: shape mismatch (dst={} src={})",
      who,
      c10::str(dst.sizes()),
      c10::str(src.sizes()));
  RBLN_CHECK(dst.numel() > 0, "{}: numel must be > 0 (caller should short-circuit)", who);
}

/**
 * @brief Route an h2v/v2h backend rejection to a CPU fallback.
 *
 * Shared by both submit_or_fallback overloads. A rejection surfaces as a
 * c10::Error whose message carries the failing runtime entrypoint; anything else
 * is a caller-side validation error and must propagate.
 */
bool is_host_copy_backend_failure(std::string_view msg) {
  return msg.find("rbln_memcpy_h2v_multi failed") != std::string_view::npos ||
      msg.find("rbln_memcpy_v2h_multi failed") != std::string_view::npos ||
      msg.find("rbln_memcpy_h2v failed") != std::string_view::npos ||
      msg.find("rbln_memcpy_v2h failed") != std::string_view::npos;
}

template <typename Batch>
void submit_or_fallback_impl(Batch& batch, const char* op_name, const std::function<void()>& cpu_fallback) {
  try {
    batch.submit();
  } catch (const c10::Error& e) {
    // TODO: Replace substring match with a typed exception when the wrapper API
    // allows — mirrors the v2v path's TODO.
    if (!is_host_copy_backend_failure(e.what())) {
      throw; // validation error — propagate
    }
    if (c10::rbln::is_fallback_disabled("strided_copy_error")) {
      throw;
    }
    RBLN_LOG_WARN(
        "{}: batched strided host copy failed — falling back to CPU op. Underlying error: {}", op_name, e.what());
    // PROFILER (cold branch): a strided h2v/v2h was rejected and fell back to a
    // host CPU op. Shares the site with the v2v strided fallback: both mean "a
    // strided copy we expected on-device ran on the host instead", which is the
    // signal a reader acts on.
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kStridedV2VFallback, 0);
    cpu_fallback();
  }
}

} // namespace

void strided_h2v_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::H2VBatch& batch) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      dst.device().is_privateuseone(),
      "strided_h2v_copy: dst must be on an RBLN (PrivateUse1) device, got {}",
      c10::str(dst.device()));
  RBLN_CHECK(src.device().is_cpu(), "strided_h2v_copy: src must be on CPU, got {}", c10::str(src.device()));
  check_pair(dst, src, "strided_h2v_copy");

  const auto plan = plan_strided(dst, src);
  if (plan.flat) {
    batch.enqueue(dst.data_ptr(), src.data_ptr(), plan.inner_block_bytes);
    return;
  }
  RBLN_LOG_DEBUG(
      "strided_h2v_copy: sizes={} src_strides={} dst_strides={} inner_block_bytes={} outer={}",
      c10::str(dst.sizes()),
      c10::str(src.strides()),
      c10::str(dst.strides()),
      plan.inner_block_bytes,
      c10::str(plan.outer_sizes));
  batch.enqueue_strided(
      dst.data_ptr(),
      src.data_ptr(),
      plan.inner_block_bytes,
      c10::IntArrayRef(plan.outer_sizes),
      c10::IntArrayRef(plan.src_byte_strides),
      c10::IntArrayRef(plan.dst_byte_strides));
}

void strided_v2h_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::V2HBatch& batch) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(dst.device().is_cpu(), "strided_v2h_copy: dst must be on CPU, got {}", c10::str(dst.device()));
  RBLN_CHECK(
      src.device().is_privateuseone(),
      "strided_v2h_copy: src must be on an RBLN (PrivateUse1) device, got {}",
      c10::str(src.device()));
  check_pair(dst, src, "strided_v2h_copy");

  const auto plan = plan_strided(dst, src);
  if (plan.flat) {
    batch.enqueue(dst.data_ptr(), src.data_ptr(), plan.inner_block_bytes);
    return;
  }
  RBLN_LOG_DEBUG(
      "strided_v2h_copy: sizes={} src_strides={} dst_strides={} inner_block_bytes={} outer={}",
      c10::str(dst.sizes()),
      c10::str(src.strides()),
      c10::str(dst.strides()),
      plan.inner_block_bytes,
      c10::str(plan.outer_sizes));
  batch.enqueue_strided(
      dst.data_ptr(),
      src.data_ptr(),
      plan.inner_block_bytes,
      c10::IntArrayRef(plan.outer_sizes),
      c10::IntArrayRef(plan.src_byte_strides),
      c10::IntArrayRef(plan.dst_byte_strides));
}

void submit_or_fallback(c10::rbln::H2VBatch& batch, const char* op_name, std::function<void()> cpu_fallback) {
  submit_or_fallback_impl(batch, op_name, cpu_fallback);
}

void submit_or_fallback(c10::rbln::V2HBatch& batch, const char* op_name, std::function<void()> cpu_fallback) {
  submit_or_fallback_impl(batch, op_name, cpu_fallback);
}

} // namespace at::native::rbln
