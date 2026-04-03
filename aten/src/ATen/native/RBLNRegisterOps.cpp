#include <ATen/core/Tensor.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNResize.h>
#include <ATen/native/rbln/RBLNTensorFactories.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace at::native {

namespace {

/**
 * @brief Fallback handler that executes unsupported operations on RBLN devices.
 *
 * This function handles operations that are not currently supported on the RBLN backend by falling back to CPU
 * execution. If `TORCH_RBLN_LOG_LEVEL` environment variable is set to `INFO` or lower, the operator name and a full
 * Python stack trace will be logged for each fallback.
 *
 * @param op    The operator handle for the operation being executed.
 * @param stack The stack containing the input and output tensors.
 */
void fallback_rbln(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  c10::rbln::log_cpu_fallback(op.schema().name());
  at::native::rbln::cpu_fallback_rbln(op, stack);
}

/**
 * @brief Error fallback handler that throws an error for unsupported operations.
 *
 * This function throws an error when an unsupported operation on the RBLN device is executed.
 *
 * @param op    The operator handle for the operation being executed.
 * @param stack The stack containing the input and output tensors.
 */
void error_fallback_rbln(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  RBLN_CHECK(false,
      "The operator `{}` is not implemented for RBLN devices. "
      "To request this op, please contact client_support@rebellions.ai. "
      "To enable fallback to CPU operation, remove 'unsupported_op' from `TORCH_RBLN_DISABLE_FALLBACK`.",
      op.schema().name());
}

/**
 * @brief Fallback handler for autograd operations.
 *
 * PyTorch has separate builds, some of which don't include autograd.
 * This function handles both cases: when autograd is not included, it
 * redispatches to the next available dispatch key; when autograd is
 * included, it uses the VariableHooksInterface layer.
 *
 * @param op            The operator handle for the operation.
 * @param dispatch_keys The set of dispatch keys available.
 * @param stack         The stack containing the input and output tensors.
 *
 * @see aten/src/ATen/core/VariableHooksInterface.h
 */
void autograd_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  if (!at::impl::HasVariableHooks()) {
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
    return;
  }
  at::impl::GetVariableHooks()->basic_autograd_not_implemented_fallback(op, dispatch_keys, stack);
}

/**
 * @brief Selects the SDP (Scaled Dot Product Attention) backend for PrivateUse1 operations.
 *
 * The overrideable backend requires 4D tensors (batch, heads, seq_len, head_dim).
 * For non-4D tensors, this function falls back to the math backend to avoid meta
 * function errors.
 *
 * @param query      The query tensor.
 * @param key        The key tensor.
 * @param value      The value tensor.
 * @param attn_mask  Optional attention mask tensor.
 * @param dropout_p  Dropout probability.
 * @param is_causal  Whether to apply causal masking.
 * @param scale      Optional scale factor.
 * @param enable_gqa Whether to enable grouped query attention.
 *
 * @return The selected SDP backend as an int64_t value.
 */
int64_t _fused_sdp_choice_privateuse1(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return static_cast<int64_t>(sdp::SDPBackend::math);
  }
  return static_cast<int64_t>(sdp::SDPBackend::overrideable);
}

// Register the SDP backend selection function for PrivateUse1 dispatch
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_privateuse1);

} // namespace

/**
 * @brief Registers fallback handlers for PrivateUse1 operations.
 *
 * By default, all unsupported ops fall back to CPU execution.
 * When `TORCH_RBLN_DISABLE_FALLBACK` contains 'unsupported_op' or 'all', an error is raised instead of falling back.
 */
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  if (c10::rbln::is_fallback_disabled("unsupported_op")) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&error_fallback_rbln>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  }
}

/**
 * @brief Registers autograd fallback handler for AutogradPrivateUse1 operations.
 */
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&autograd_fallback>());
}

// ATen operations registration for the RBLN backend (PrivateUse1)
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Operations that use the device runtime API
  m.impl("_copy_from", TORCH_FN(at::native::rbln::_copy_from_rbln));
  m.impl("_copy_from_and_resize", TORCH_FN(at::native::rbln::_copy_from_and_resize_rbln));
  m.impl("empty.memory_format", TORCH_FN(at::native::rbln::empty_rbln));
  m.impl("empty_strided", TORCH_FN(at::native::rbln::empty_strided_rbln));
  m.impl("resize_", TORCH_FN(at::native::rbln::resize_rbln_));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(at::native::rbln::set_storage_rbln_));

  // View operations (metadata-only, no device computation)
  // These operations only manipulate tensor metadata and do not require
  // device computation, so they don't move tensors to CPU in the fallback
  // sequence but simply manipulate the metadata.
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("set_.source_Tensor", TORCH_FN(at::native::set_tensor_));
  m.impl("set_.source_Storage", TORCH_FN(at::native::set_));
  m.impl("unfold", TORCH_FN(at::native::unfold));
  m.impl("_unsafe_view", TORCH_FN(at::native::_unsafe_view));
  m.impl("alias", TORCH_FN(at::native::alias));
  // Note: view_as_real falls back to CPU because conjugate is not supported
  m.impl("view_as_real", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Operations that compile on the RBLN device
  // TODO: These are temporarily implemented as CPU fallbacks. They are needed
  // for the print op, not because they are used in a specific model.
  m.impl("max.dim_max", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("min.dim_min", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("flip", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("cumsum.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("arange.start_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Operations used in pytest (temporary CPU fallbacks)
  // These are registered to manage unsupported operations encountered during pytest runs.
  m.impl("sum.IntList_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("clamp.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Operations not supported on the RBLN device (CPU fallback)
  // Scalar and tensor manipulation
  m.impl("trunc.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("_local_scalar_dense", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("fill_.Scalar", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("fill_.Tensor", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("equal", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("_efficientzerotensor", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("native_dropout", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Random number generation
  m.impl("normal_", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("normal.Tensor_Tensor", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("normal.Tensor_float", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("uniform_", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("random_.from", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("randperm.generator_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Indexing and selection operations
  m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("masked_select.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("masked_scatter_", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("index.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("_index_put_impl_", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  // Note: index_select doesn't work on device if arg(index) is a tensor
  m.impl("index_select", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("index_select.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("index_add.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("index_copy.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("index_fill_.int_Scalar", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("gather.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("scatter.src_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("put_", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("nonzero", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("cat.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Bitwise operations
  m.impl("bitwise_and.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_not.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_or.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_xor.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_and.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_not.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_or.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("bitwise_xor.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Logical operations
  m.impl("logical_and.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("logical_or.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("logical_xor.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Reduction and comparison operations
  m.impl("all.all_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("all.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("any.all_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("any.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("argmax.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("topk.values", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("count_nonzero.dim_IntList", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Mathematical functions
  m.impl("cos.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("sin.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("exp.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("fmod.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("sgn.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("sign.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("signbit.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("pow.Tensor_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("pow.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("linalg_vector_norm.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("_linalg_svd.U", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("addcmul.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Special value checks
  m.impl("isnan", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("isposinf.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("isneginf.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("nan_to_num.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("isin.Tensor_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Matrix operations
  m.impl("tril.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("triu.out", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Activation and normalization functions
  m.impl("relu", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("native_batch_norm", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // Backward operations
  m.impl("logit_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("sigmoid_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("tanh_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("unfold_backward", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());
  m.impl("native_dropout_backward", torch::CppFunction::makeFromBoxedFunction<&fallback_rbln>());

  // SDPA overrideable operations
  m.impl("_fused_sdp_choice", &_fused_sdp_choice_privateuse1);
}

} // namespace at::native
