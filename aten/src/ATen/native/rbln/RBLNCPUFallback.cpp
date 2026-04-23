// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/core/DispatchKey.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

namespace at::native::rbln {

namespace {

bool is_borrow_aliasable_dtype(c10::ScalarType dtype) {
  // Only integer/boolean dtypes where raw host bytes round-trip cleanly; rebel
  // never reinterprets these the way it may for float16. fp dtypes stay on the
  // existing device-backed path and continue to flow through NPU compile.
  switch (dtype) {
    case c10::kBool:
    case c10::kByte:
    case c10::kChar:
    case c10::kShort:
    case c10::kInt:
    case c10::kLong:
      return true;
    default:
      return false;
  }
}

bool can_borrow_alias(const at::Tensor& t) {
  if (!t.defined()) return false;
  if (!t.device().is_privateuseone()) return false;
  if (!is_borrow_aliasable_dtype(t.scalar_type())) return false;
  // Contiguous required so at::from_blob sees a linear memory region. Non-
  // contiguous views have interior strides that rebel's borrow cannot expose
  // as a single host-backed range; let those fall back through memcpy_v2h.
  if (!t.is_contiguous()) return false;
  if (t.numel() == 0) return false;
  // View-safe: we borrow the whole storage and offset into it. Sanity-check
  // that the view lies inside the storage (it always should).
  const auto element_nbytes = static_cast<size_t>(t.numel()) * t.element_size();
  const auto offset_nbytes = static_cast<size_t>(t.storage_offset()) * t.element_size();
  if (offset_nbytes + element_nbytes > t.storage().nbytes()) return false;
  return true;
}

std::optional<c10::OperatorHandle> find_out_variant(const c10::OperatorHandle& op) {
  const auto& name = op.schema().operator_name();
  auto& dispatcher = c10::Dispatcher::singleton();
  // Try the common out-variant overload suffixes the PyTorch codegen emits.
  // The order matters slightly (prefer plain "out" before the typed variants).
  static constexpr const char* kCandidateOverloads[] = {
      "out",
      "Tensor_out",
      "Scalar_out",
      "Tensor_Tensor_out",
      "Tensor_Scalar_out",
      "Scalar_Tensor_out",
      "Scalar_Scalar_out",
      "dim_out",
      "dimname_out",
      "DimnameList_out",
      "IntList_out",
      "names_out",
      "dtype_out",
      "int_out",
      "self_out",
      "src_out",
      "value_out",
      "unary_out",
  };
  for (const char* overload : kCandidateOverloads) {
    auto candidate = c10::OperatorName(name.name, overload);
    if (auto handle = dispatcher.findSchema(candidate); handle.has_value()) {
      return handle;
    }
  }
  return std::nullopt;
}

// Borrow the backing host pointer of an rbln tensor and wrap it as a CPU
// tensor via at::from_blob. We borrow the full storage (not just the tensor
// slice) so views with storage_offset > 0 or sub-storage extent still work:
// rebel only knows about storage-level v-memory entries, and re-borrowing the
// same storage twice in the same fallback call is safe because each borrow
// gets its own borrow_id. ``writable`` threads the schema's isWrite() bit
// through to return_borrowed so rebel marks the host side as latest only for
// mutating ops.
// Forward declaration (defined below). Needed by try_output_alias_dispatch.
at::Tensor borrow_alias_as_cpu(const at::Tensor& rbln_t, bool writable);

// Run the op through the Meta dispatch key to infer its output shape/dtype
// without touching real storage. Used by the D2 non-out-variant path to
// size the pre-allocated rbln output before dispatching on CPU. Returns
// nullopt for ops whose inputs contain tensorlists, or whose meta kernel
// is missing / throws. (Input-side tensorlist inference is not yet plumbed
// through, so those ops stay on the regular fallback path.)
std::optional<at::Tensor> infer_single_tensor_output_via_meta(
    const c10::OperatorHandle& op, const torch::jit::Stack& orig_stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  if (orig_stack.size() < num_arguments) return std::nullopt;
  const auto arguments_begin = orig_stack.size() - num_arguments;

  torch::jit::Stack meta_stack;
  meta_stack.reserve(num_arguments);
  for (size_t i = 0; i < num_arguments; ++i) {
    const auto& ivalue = orig_stack[arguments_begin + i];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (!t.defined()) {
        meta_stack.emplace_back(at::Tensor{});
        continue;
      }
      auto meta_t = at::empty_strided(t.sizes(), t.strides(),
                                       t.options().device(c10::kMeta));
      meta_stack.emplace_back(std::move(meta_t));
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      // Not handled — signal fallback.
      return std::nullopt;
    } else if (ivalue.isDevice()) {
      meta_stack.emplace_back(c10::IValue(c10::Device(c10::kMeta)));
    } else {
      meta_stack.emplace_back(ivalue);
    }
  }

  try {
    c10::Dispatcher::singleton().callBoxedForDispatchKey(
        op, c10::DispatchKey::Meta, &meta_stack);
  } catch (...) {
    return std::nullopt;
  }

  if (schema.returns().size() != 1) return std::nullopt;
  if (meta_stack.empty()) return std::nullopt;
  const auto& ret_ivalue = meta_stack.back();
  if (!ret_ivalue.isTensor()) return std::nullopt;
  const auto& meta_out = ret_ivalue.toTensor();
  if (!meta_out.defined()) return std::nullopt;
  return meta_out;
}

// Core output-alias entry point. CPU fallback sees the op already lowered to
// its out-variant by PyTorch's CompositeImplicitAutograd (e.g. for a user call
// ``torch.arange(5, 15, device='rbln')`` the op reaching this function is
// ``aten::arange.start_out`` with an rbln ``out`` tensor already on the stack).
// We turn that existing out into a CPU at::from_blob alias over the rebel
// borrowed host pointer, dispatch the out-variant on CPU so it writes
// directly into the host-backed storage, and return the original rbln tensor
// from the stack — skipping the output H2D copy the regular Step 4 would do.
//
// Returns true iff the op was dispatched via the aliased path; caller then
// skips the default Step 2 / Step 4 processing for this op.
bool try_output_alias_dispatch(
    const c10::OperatorHandle& op, torch::jit::Stack* stack,
    c10::DispatchKey cpu_dispatch_key,
    const std::vector<at::Tensor>& tensor_args,
    const std::vector<int>& tensor_args_indices) {
  const auto& schema = op.schema();
  const auto& returns = schema.returns();
  if (returns.size() != 1) return false;
  if (returns[0].type()->kind() != c10::TypeKind::TensorType) return false;

  // Only handle the case where the caller has already produced an
  // out-variant schema — that is, one of the positional args is named
  // ``out`` and marked as a write-alias.
  const auto& args = schema.arguments();
  int out_idx = -1;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].name() == "out") {
      const auto* ai = args[i].alias_info();
      if (ai != nullptr && ai->isWrite()) {
        out_idx = static_cast<int>(i);
        break;
      }
    }
  }
  if (out_idx < 0) return false;

  // Find the original rbln ``out`` tensor from tensor_args. Stack currently
  // holds the CPU-converted version (Step 1 ran at::_to_cpu on fallback
  // inputs), which is not what we want to borrow from.
  int tensor_args_pos = -1;
  for (size_t i = 0; i < tensor_args_indices.size(); ++i) {
    if (tensor_args_indices[i] == out_idx) {
      tensor_args_pos = static_cast<int>(i);
      break;
    }
  }
  if (tensor_args_pos < 0) return false;
  const auto out_tensor = tensor_args[tensor_args_pos];
  if (!out_tensor.defined()) return false;
  if (!can_borrow_alias(out_tensor)) return false;
  const auto num_arguments = args.size();
  if (stack->size() < num_arguments) return false;
  const auto args_begin = stack->size() - num_arguments;
  auto& out_ivalue = (*stack)[args_begin + out_idx];

  // Tell rebel "contents are irrelevant, next write will overwrite everything"
  // so the upcoming borrow doesn't waste a d->h sync of the stale backing.
  // rbln_v_mark_zeros is the documented hint: "On the next device write, the
  // transfer is skipped entirely (PHYSICAL_VIEW_IS_LATEST)". Wrapped in a
  // try/catch so a mark-zeros failure just loses the optimisation instead of
  // breaking the op.
  try {
    c10::rbln::mark_zeros(out_tensor.storage().data_ptr().get());
  } catch (...) {
    // Not fatal — proceed with borrow anyway.
  }

  // Wrap the rbln ``out`` as a CPU alias; CPU kernel will write through it.
  at::Tensor cpu_alias_out;
  try {
    cpu_alias_out = borrow_alias_as_cpu(out_tensor, /*writable=*/true);
  } catch (...) {
    return false;
  }
  // Substitute the alias into the out slot of the stack.
  out_ivalue = c10::IValue(cpu_alias_out);

  // Redispatch the op on CPU with the aliased out in place.
  try {
    op.callBoxedForDispatchKey(cpu_dispatch_key, *stack);
  } catch (...) {
    // Propagate; the stack state is inconsistent after a kernel throw.
    throw;
  }

  // The kernel pushed one tensor return (== the alias, per out-variant
  // semantics). Replace it with the original rbln tensor so downstream code
  // sees the device-side result and Step 4 becomes a no-op.
  if (!stack->empty() && stack->back().isTensor()) {
    stack->pop_back();
  }
  stack->emplace_back(out_tensor);
  return true;
}

// D2: output alias for non-out-variant ops. When PyTorch's CompositeImplicit
// lowering doesn't rewrite the op to its ``.out`` overload before reaching
// cpu_fallback_rbln (some niche ops only), but the Dispatcher still has an
// out-variant registered, we:
//   1. meta-dispatch the original op to infer output shape/dtype
//   2. pre-allocate the rbln destination
//   3. mark_zeros + borrow it as a CPU at::from_blob alias
//   4. push that alias as an extra ``out`` slot on the stack
//   5. callBoxedForDispatchKey on the out-variant (CPU)
//   6. replace the alias with the rbln tensor as the return
//
// Returns true iff the op was handled. Conservative: rejects tensor-list
// inputs (meta inference doesn't cover those here) and any output dtype that
// isn't alias-safe. Caller must skip the normal Step 2 when true.
bool try_output_alias_non_out_variant(
    const c10::OperatorHandle& op, torch::jit::Stack* stack,
    c10::DispatchKey cpu_dispatch_key,
    const std::vector<at::Tensor>& tensor_args,
    const std::vector<int>& /*tensor_args_indices*/) {
  const auto& schema = op.schema();
  const auto& returns = schema.returns();
  if (returns.size() != 1) return false;
  if (returns[0].type()->kind() != c10::TypeKind::TensorType) return false;
  if (returns[0].alias_info() != nullptr) return false;

  const auto& orig_args = schema.arguments();
  // Only handle ops that are *not* already an out-variant (the existing path
  // try_output_alias_dispatch covers those).
  for (const auto& a : orig_args) {
    if (a.name() == "out") return false;
  }

  auto out_op_opt = find_out_variant(op);
  if (!out_op_opt.has_value()) return false;
  const auto& out_op = *out_op_opt;
  const auto& out_schema = out_op.schema();
  const auto& out_args = out_schema.arguments();
  if (out_args.size() != orig_args.size() + 1) return false;
  if (out_args.back().name() != "out") return false;
  for (size_t i = 0; i < orig_args.size(); ++i) {
    if (orig_args[i].name() != out_args[i].name()) return false;
  }

  auto meta_out_opt = infer_single_tensor_output_via_meta(op, *stack);
  if (!meta_out_opt.has_value()) return false;
  const auto& meta_out = *meta_out_opt;
  if (!is_borrow_aliasable_dtype(meta_out.scalar_type())) return false;

  // Output device inherits from the first rbln tensor input.
  c10::Device out_device(c10::kCPU);
  bool found_device = false;
  for (const auto& t : tensor_args) {
    if (t.defined() && t.device().is_privateuseone()) {
      out_device = t.device();
      found_device = true;
      break;
    }
  }
  if (!found_device) return false;

  auto rbln_out = at::empty(
      meta_out.sizes(),
      at::TensorOptions().dtype(meta_out.scalar_type()).device(out_device));
  if (!can_borrow_alias(rbln_out)) return false;

  try {
    c10::rbln::mark_zeros(rbln_out.storage().data_ptr().get());
  } catch (...) {
    // not fatal
  }

  at::Tensor cpu_alias;
  try {
    cpu_alias = borrow_alias_as_cpu(rbln_out, /*writable=*/true);
  } catch (...) {
    return false;
  }

  stack->emplace_back(cpu_alias);
  try {
    out_op.callBoxedForDispatchKey(cpu_dispatch_key, *stack);
  } catch (...) {
    throw;
  }

  if (!stack->empty() && stack->back().isTensor()) {
    stack->pop_back();
  }
  stack->emplace_back(rbln_out);
  return true;
}

at::Tensor borrow_alias_as_cpu(const at::Tensor& rbln_t, bool writable) {
  uint64_t borrow_id = 0;
  const auto storage_nbytes = rbln_t.storage().nbytes();
  const auto element_size = rbln_t.element_size();
  const auto offset_nbytes = static_cast<size_t>(rbln_t.storage_offset()) * element_size;
  const uintptr_t storage_host_ptr = c10::rbln::borrow_host_ptr(
      rbln_t.storage().data_ptr().get(), storage_nbytes, borrow_id);
  const uintptr_t tensor_host_ptr = storage_host_ptr + offset_nbytes;
  auto deleter = [borrow_id, writable](void*) {
    c10::rbln::return_borrowed(borrow_id, /*updated=*/writable);
  };
  return at::from_blob(
      reinterpret_cast<void*>(tensor_host_ptr),
      rbln_t.sizes(),
      rbln_t.strides(),
      std::move(deleter),
      at::TensorOptions().dtype(rbln_t.scalar_type()).device(at::kCPU));
}

// Per-element alias path for (optional) TensorList args. Tries to alias each
// element that satisfies can_borrow_alias; the rest are materialised in one
// batch through at::_to_cpu. Returns a parallel aliased_mask so Step 3 can
// skip _copy_from_and_resize on the aliased elements.
template <typename T>
std::pair<std::vector<T>, std::vector<bool>> alias_list_elements(
    const std::vector<T>& elems, bool writable) {
  std::vector<T> out(elems.size());
  std::vector<bool> aliased(elems.size(), false);
  std::vector<at::Tensor> fallback;
  std::vector<size_t> fallback_pos;
  fallback.reserve(elems.size());
  fallback_pos.reserve(elems.size());
  for (size_t i = 0; i < elems.size(); ++i) {
    const at::Tensor* tref = nullptr;
    if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
      if (elems[i].has_value() && elems[i].value().defined()) {
        tref = &elems[i].value();
      }
    } else {
      if (elems[i].defined()) tref = &elems[i];
    }
    if (tref == nullptr) {
      out[i] = elems[i];
      continue;
    }
    if (can_borrow_alias(*tref)) {
      at::Tensor aliased_cpu;
      try {
        aliased_cpu = borrow_alias_as_cpu(*tref, writable);
      } catch (...) {
        fallback_pos.push_back(i);
        fallback.push_back(*tref);
        continue;
      }
      if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
        out[i] = std::optional<at::Tensor>(aliased_cpu);
      } else {
        out[i] = aliased_cpu;
      }
      aliased[i] = true;
    } else {
      fallback_pos.push_back(i);
      fallback.push_back(*tref);
    }
  }
  if (!fallback.empty()) {
    auto fb_cpu = at::_to_cpu(fallback);  // fallback entries are all defined
    for (size_t j = 0; j < fallback_pos.size(); ++j) {
      const auto pos = fallback_pos[j];
      if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
        out[pos] = std::optional<at::Tensor>(std::move(fb_cpu[j]));
      } else {
        out[pos] = std::move(fb_cpu[j]);
      }
    }
  }
  return {std::move(out), std::move(aliased)};
}

// convenience helper for converting tensors to cpu
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Tensor> || std::is_same_v<T, std::optional<at::Tensor>>, int> = 1>
std::vector<T> to_cpu(const std::vector<T>& tensors) {
  // We can't just call at::to_cpu() on the entire list of Tensors
  // Because it will break on undefined tensors. Separate out undefined tensors first.
  const int num = tensors.size();
  std::vector<T> cpu_tensors(num);
  std::vector<at::Tensor> valid_tensors;
  std::vector<bool> to_translate(num);
  for (const auto i : c10::irange(num)) {
    to_translate[i] = false;
    // Explicitly handling undefined tensors here instead of letting `at::_to_cpu` handle it.
    // Otherwise, we'd need to require all backends with their own implementation of _to_cpu
    // to properly handle undefined tensors.
    if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
      if (tensors[i].has_value() && tensors[i].value().defined()) {
        const at::Tensor& tensor_ref = tensors[i].value();
        to_translate[i] = true;
        valid_tensors.push_back(tensors[i].value());
      } else {
        cpu_tensors[i] = tensors[i];
      }
    } else {
      if (tensors[i].defined()) {
        const at::Tensor& tensor_ref = tensors[i];
        to_translate[i] = true;
        valid_tensors.push_back(tensors[i]);
      } else {
        cpu_tensors[i] = tensors[i];
      }
    }
  }

  // copy device to cpu
  auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
  for (int i = 0, defined_pos = 0; i < num; ++i) {
    if (to_translate[i]) {
      cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
    }
  }
  return cpu_tensors;
}

std::optional<c10::Device> compute_target_device(
    std::vector<at::Tensor>& t_args,
    const std::vector<c10::List<at::Tensor>>& tlist_args) {
  // Decide what device to move the output tensor(s) to.
  // The current convention is that we use the first tensor arg to pick the device
  // Barring that, we take the first tensor from a TensorList arg.
  if (!t_args.empty()) {
    return t_args[0].device();
  } else {
    // We need to loop through all of the (potentially multiple) TensorList arguments
    // In case, e.g. the first one is empty but the second is not.
    for (auto& tens_list : tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        return tens_list.get(i).device();
      }
    }
  }
  return std::nullopt;
}

bool validate_tensor_list(const c10::List<at::Tensor>& tensorlist) {
  bool flag = false;

  for (const auto& i : c10::irange(tensorlist.size())) {
    if (tensorlist[i].defined())
      flag = true;
  }

  return flag;
}

} // namespace

void cpu_fallback_rbln(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool error_on_views,
    c10::DispatchKey cpu_dispatch_key) {
  TORCH_CHECK(
      c10::BackendComponent::CPUBit == c10::toBackendComponent(cpu_dispatch_key),
      "Expected CPU backend DispatchKey but got ",
      c10::toString(cpu_dispatch_key));

  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  std::vector<c10::List<at::Tensor>> tensorlist_args;
  std::vector<int> tensorlist_args_indices;

  std::vector<c10::List<std::optional<at::Tensor>>> optional_tensorlist_args;
  std::vector<int> optional_tensorlist_args_indices;

  std::optional<c10::Device> tgt_device = std::nullopt;
  // save converted cpu tensor for TensorList and optional TensorList
  std::vector<c10::IValue> tensorlist_cpu_args;
  std::vector<c10::IValue> optional_tensorlist_cpu_args;
  // Per-element alias mask so Step 3 can skip _copy_from_and_resize on the
  // list entries that were aliased (their deleter already handles the write).
  std::vector<std::vector<bool>> tensorlist_aliased_mask;
  std::vector<std::vector<bool>> optional_tensorlist_aliased_mask;

  // Step 1: Convert all non-CPU tensor inputs into CPU tensors
  // and put them on the stack at the correct indices.
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // Per-element alias (D1): individual rbln tensors in the list go through
      // the zero-copy borrow path, the rest fall back in a single at::_to_cpu.
      tensorlist_args.push_back(ivalue.toTensorList());
      tensorlist_args_indices.push_back(idx);
      const AliasInfo* list_alias_info = schema_args[idx].alias_info();
      const bool list_is_write = (list_alias_info != nullptr && list_alias_info->isWrite());
      auto [aliased_vec, mask] = alias_list_elements<at::Tensor>(
          ivalue.toTensorVector(), list_is_write);
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(aliased_vec));
      tensorlist_cpu_args.push_back(cpu_ivalue);
      tensorlist_aliased_mask.push_back(std::move(mask));
      (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    } else if (ivalue.isOptionalTensorList()) {
      optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
      optional_tensorlist_args_indices.push_back(idx);
      const AliasInfo* list_alias_info = schema_args[idx].alias_info();
      const bool list_is_write = (list_alias_info != nullptr && list_alias_info->isWrite());
      auto [aliased_vec, mask] = alias_list_elements<std::optional<at::Tensor>>(
          ivalue.toOptionalTensorVector(), list_is_write);
      auto cpu_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(aliased_vec));
      optional_tensorlist_cpu_args.push_back(cpu_ivalue);
      optional_tensorlist_aliased_mask.push_back(std::move(mask));
      (*stack)[arguments_begin + idx] = c10::IValue(cpu_ivalue);
    } else if (ivalue.isDevice()) {
      tgt_device = ivalue.toDevice();
      (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(kCPU));
    }
  }
  // Build the CPU tensor view for each tensor argument. Integer rbln tensors
  // (read-only AND write mutable) are exposed as at::from_blob wrappers over
  // the rebel-borrowed host pointer (zero copy); everything else still flows
  // through the batched at::_to_cpu path.
  //
  // aliased_mask[i] records whether cpu_tensors[i] is a borrow alias; used in
  // Step 3 to skip the redundant _copy_from_and_resize on write inputs (the
  // alias deleter already marks the host side as latest via updated=true).
  std::vector<at::Tensor> cpu_tensors(tensor_args.size());
  std::vector<bool> aliased_mask(tensor_args.size(), false);
  std::vector<at::Tensor> fallback_inputs;
  std::vector<size_t> fallback_positions;
  fallback_inputs.reserve(tensor_args.size());
  fallback_positions.reserve(tensor_args.size());
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    const auto idx = tensor_args_indices[i];
    const AliasInfo* alias_info = schema_args[idx].alias_info();
    const bool schema_is_write = (alias_info != nullptr && alias_info->isWrite());
    if (can_borrow_alias(tensor_args[i])) {
      cpu_tensors[i] = borrow_alias_as_cpu(tensor_args[i], /*writable=*/schema_is_write);
      aliased_mask[i] = true;
    } else {
      fallback_positions.push_back(i);
      fallback_inputs.push_back(tensor_args[i]);
    }
  }
  if (!fallback_inputs.empty()) {
    auto fallback_cpu = to_cpu(fallback_inputs);
    for (size_t j = 0; j < fallback_positions.size(); ++j) {
      cpu_tensors[fallback_positions[j]] = std::move(fallback_cpu[j]);
    }
  }

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  // Step 2a (optional C-alpha): try to dispatch the op via its out-variant with
  // a pre-allocated rbln output wrapped as a CPU alias. On success the stack
  // already holds the rbln tensor return, so we skip the regular Step 2 /
  // mutable-alias handling / Step 4 device copy for this op.
  bool output_alias_applied =
      try_output_alias_dispatch(op, stack, cpu_dispatch_key, tensor_args, tensor_args_indices);
  if (!output_alias_applied) {
    // D2: op wasn't already in out-variant form — try to reroute to its
    // registered out-variant (meta-dispatch for shape, pre-alloc, redispatch).
    output_alias_applied = try_output_alias_non_out_variant(
        op, stack, cpu_dispatch_key, tensor_args, tensor_args_indices);
  }

  if (!output_alias_applied) {
    // Step 2: Call the underlying CPU implementation of the operator
    op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);
  }

  // Step 3: We need to take special care to handle mutable aliases properly:
  // If any input tensors are mutable aliases, we need to
  // directly copy the updated data on the CPU tensors back to the original inputs.
  // When the output-alias path fired, the CPU kernel already wrote through the
  // rebel-backed host pointer, so we must skip the h2v copy-back entirely —
  // otherwise we re-copy the stale CPU input buffer over the freshly-written
  // rbln storage.
  if (!output_alias_applied) {
    for (const auto i : c10::irange(tensor_args_indices.size())) {
      auto tensor_idx = tensor_args_indices[i];
      const AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
      if (alias_info != nullptr && alias_info->isWrite()) {
        if (!tensor_args[i].defined())
          continue;
        // Alias path already writes directly into the rbln-backed host pointer;
        // the deleter tagged the borrow with updated=true so rebel will sync on
        // the next device read. No explicit device-side copy needed.
        if (aliased_mask[i])
          continue;
        at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
      }
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of tensors. Skip elements that went through the borrow-alias path — their
  // deleter will mark the host as latest so the h2v copy is unnecessary.
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = tensorlist_cpu_args[i].toTensorVector();
      const auto& mask = tensorlist_aliased_mask[i];
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        if (!cpu_tensors[idx].defined())
          continue;
        if (idx < mask.size() && mask[idx]) continue;
        at::_copy_from_and_resize(cpu_tensors[idx], tensorlist_args[i][idx]);
      }
    }
  }

  // Same for optional TensorLists.
  for (const auto i : c10::irange(optional_tensorlist_args_indices.size())) {
    auto tensorlist_idx = optional_tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = optional_tensorlist_cpu_args[i].toOptionalTensorList();
      const auto& mask = optional_tensorlist_aliased_mask[i];
      for (const auto idx : c10::irange(optional_tensorlist_args[i].size())) {
        if (cpu_tensors[idx].has_value() && cpu_tensors[idx].value().defined()) {
          if (idx < mask.size() && mask[idx]) continue;
          const std::optional<at::Tensor>& optional_tensor = optional_tensorlist_args[i][idx];
          at::_copy_from_and_resize(cpu_tensors[idx].value(), optional_tensor.value());
        }
      }
    }
  }

  // Step 4: Convert any CPU output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary CPU output tensor that we created.
  //
  // Note [CPU Fallback Does Not Handle View Operators]
  // Also note that we are incapable of handling immutable aliases properly.
  // Why?
  // Schemas with an immutable alias'd tensor outputs correspond to view operators.
  // For example, the `view_as` schema from native_functions.yaml:
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // We can't handle these ops properly, because view ops are supposed to return
  // a NEW tensor that shares the SAME storage as the original tensor.
  // However, the new tensor that we created cannot share the same storage,
  // since it lives on CPU and the original tensor lives on a different device.
  // Because of that, we warn if someone attempts to call the
  // CPU fallback on a view operator (this is to maintain BC for view ops for XLA
  // that fall back to CPU).
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  if (tgt_device == std::nullopt) {
    tgt_device = compute_target_device(tensor_args, tensorlist_args);
  }

  for (const auto idx : c10::irange(returns.size())) {
    const AliasInfo* alias_info = schema_returns[idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      // Case (1): mutable alias case.
      // Move the input ivalue directly onto the stack in place of
      // the existing cpu output tensor.
      bool found_alias = false;
      if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
        // We could store some extra metadata on the function schema to avoid
        // the loop here if we need to improve perf.
        for (const auto i : c10::irange(tensor_args_indices.size())) {
          auto input_tensor_idx = tensor_args_indices[i];
          const auto& input_tensor = cpu_tensors[i];
          const AliasInfo* input_alias_info = schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (input_tensor.defined() &&
              (alias_info == input_alias_info || (input_alias_info != nullptr && *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
            found_alias = true;
            break;
          }
        }
      } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
        for (const auto i : c10::irange(tensorlist_args_indices.size())) {
          auto input_tensor_idx = tensorlist_args_indices[i];
          const AliasInfo* input_alias_info = schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (validate_tensor_list(tensorlist_args[i]) &&
              (alias_info == input_alias_info || (input_alias_info != nullptr && *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensorlist_args[i]);
            found_alias = true;
            break;
          }
        }
      }
      TORCH_CHECK(
          found_alias,
          "The operator ",
          op.schema().operator_name(),
          " appears to have invalid alias information. ",
          "Found a return tensor argument with a mismatched mutable alias: ",
          schema_returns[idx]);
    } else {
      if (alias_info != nullptr && !alias_info->isWrite()) {
        // Case (3): immutable alias (view) case.
        // Warn here, since we're copying and not creating a view.
        // If this operator is needed, the backend should provide a kernel for
        // it. See Note [CPU Fallback Does Not Handle View Operators]
        std::stringstream dev_str;
        if (tgt_device) {
          dev_str << *tgt_device;
        } else {
          dev_str << "<none>";
        }
        if (error_on_views) {
          TORCH_CHECK(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support ",
              "since the tensor's storage cannot be shared across devices.");
        } else {
          TORCH_WARN(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support falling back to run on the CPU, ",
              "since the tensor's storage cannot be shared across devices.");
        }
      }
      // Case (2): copy case.
      // Copy the cpu output tensor to the original device.

      // We technically  might not have a target device, e.g. if you call
      // torch.cat() with an empty list In that case, we shouldn't have any
      // tensors to schlep across devices anyway.
      if (tgt_device) {
        if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
          (*stack)[returns_begin + idx] = c10::IValue(returns[idx].toTensor().to(*tgt_device));
        } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
          const auto& cpu_tensors = returns[idx].toTensorList().vec();
          std::vector<at::Tensor> tensors;
          tensors.reserve(cpu_tensors.size());

          for (const auto& tensor : cpu_tensors) {
            tensors.push_back(tensor.to(*tgt_device));
          }
          (*stack)[returns_begin + idx] = c10::IValue(c10::List<at::Tensor>(tensors));
        }
      }
    }
  }
}

} // namespace at::native::rbln
