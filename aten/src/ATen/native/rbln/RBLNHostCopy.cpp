#include <ATen/native/rbln/RBLNHostCopy.h>

#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <string_view>

namespace at::native::rbln {

namespace {

/** @brief Shape / dtype / layout / numel preconditions common to both directions. */
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
  RBLN_CHECK(
      dst.is_contiguous() && src.is_contiguous(),
      "{}: both tensors must be contiguous (dst={} src={}) — the caller owns the eligibility check",
      who,
      dst.is_contiguous(),
      src.is_contiguous());
  RBLN_CHECK(dst.numel() > 0, "{}: numel must be > 0 (caller should short-circuit)", who);
}

/** @brief Total payload of a contiguous pair, in bytes. */
size_t payload_bytes(const at::Tensor& dst) {
  return static_cast<size_t>(dst.numel()) * static_cast<size_t>(dst.element_size());
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
    RBLN_LOG_WARN("{}: batched host copy failed — falling back to CPU op. Underlying error: {}", op_name, e.what());
    // PROFILER (cold branch): an h2v/v2h batch was rejected and fell back to a
    // host CPU op. Shares the site with the v2v fallback: both mean "a copy we
    // expected the device to do ran on the host instead", which is the signal a
    // reader acts on.
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kStridedV2VFallback, 0);
    cpu_fallback();
  }
}

} // namespace

void h2v_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::H2VBatch& batch) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      dst.device().is_privateuseone(),
      "h2v_copy: dst must be on an RBLN (PrivateUse1) device, got {}",
      c10::str(dst.device()));
  RBLN_CHECK(src.device().is_cpu(), "h2v_copy: src must be on CPU, got {}", c10::str(src.device()));
  check_pair(dst, src, "h2v_copy");

  batch.enqueue(dst.data_ptr(), src.data_ptr(), payload_bytes(dst));
}

void v2h_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::V2HBatch& batch) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(dst.device().is_cpu(), "v2h_copy: dst must be on CPU, got {}", c10::str(dst.device()));
  RBLN_CHECK(
      src.device().is_privateuseone(),
      "v2h_copy: src must be on an RBLN (PrivateUse1) device, got {}",
      c10::str(src.device()));
  check_pair(dst, src, "v2h_copy");

  batch.enqueue(dst.data_ptr(), src.data_ptr(), payload_bytes(dst));
}

void submit_or_fallback(c10::rbln::H2VBatch& batch, const char* op_name, std::function<void()> cpu_fallback) {
  submit_or_fallback_impl(batch, op_name, cpu_fallback);
}

void submit_or_fallback(c10::rbln::V2HBatch& batch, const char* op_name, std::function<void()> cpu_fallback) {
  submit_or_fallback_impl(batch, op_name, cpu_fallback);
}

} // namespace at::native::rbln
