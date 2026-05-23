// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>

#include <c10/rbln/RBLNFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::native::rbln {

namespace {
// Per-op schema cache: mirrors the DispatchShim SchemaCache idea. Same op handle
// reuses the same FunctionSchema pointer for the process lifetime, so caching
// the per-arg static flags avoids walking schema_args/alias_info every dispatch.
//
// LLaMA-1B eager hits cpu_fallback_rbln 10066 times across <10 distinct ops, so
// the cache populates at startup and stays read-only. Keep the populate step
// behind a unique_lock; readers take a shared_lock and are otherwise lock-free.
// OptionalTensor covers schema args typed `Tensor?` (e.g. linear's `bias`).
// At runtime such an IValue is either a Tensor or None — old code recognized
// this via the IValue's `isTensor()` cascade. After moving to schema-typed
// arg_kind dispatch we must classify Tensor? explicitly, otherwise the arg
// stays on the stack as an RBLN tensor when CPU dispatch runs (mixed-device
// kernel inputs blow up at the next dispatch hop).
enum class CpuFbArgKind : uint8_t {
  Other = 0,
  Tensor,
  OptionalTensor,
  TensorList,
  OptionalTensorList,
  Device,
};

struct CpuFbSchemaInfo {
  std::vector<CpuFbArgKind> arg_kind; // per-positional arg
  std::vector<bool> is_write_alias; // alias_info != null && isWrite
  std::vector<bool> is_pure_out; // kwarg_only && name=="out" && is_write_alias
  bool populated = false;
};

struct CpuFbSchemaCache {
  std::shared_mutex mu;
  std::unordered_map<const c10::FunctionSchema*, CpuFbSchemaInfo> by_schema;
};

CpuFbSchemaCache& schema_cache() {
  static auto* c = new CpuFbSchemaCache();
  return *c;
}

const CpuFbSchemaInfo& get_or_populate_schema_info(const c10::FunctionSchema& schema) {
  auto& cache = schema_cache();
  const auto* key = &schema;
  {
    std::shared_lock<std::shared_mutex> rd(cache.mu);
    auto it = cache.by_schema.find(key);
    if (it != cache.by_schema.end() && it->second.populated) {
      return it->second;
    }
  }
  std::unique_lock<std::shared_mutex> wr(cache.mu);
  auto& info = cache.by_schema[key];
  if (info.populated) {
    return info;
  }
  const auto& args = schema.arguments();
  const auto n = args.size();
  info.arg_kind.assign(n, CpuFbArgKind::Other);
  info.is_write_alias.assign(n, false);
  info.is_pure_out.assign(n, false);
  for (size_t i = 0; i < n; ++i) {
    const auto& a = args[i];
    const auto& type = a.type();
    using namespace c10;
    if (type->isSubtypeOf(*TensorType::get())) {
      info.arg_kind[i] = CpuFbArgKind::Tensor;
    } else if (type->isSubtypeOf(*ListType::ofTensors())) {
      info.arg_kind[i] = CpuFbArgKind::TensorList;
    } else if (type->isSubtypeOf(*ListType::ofOptionalTensors())) {
      info.arg_kind[i] = CpuFbArgKind::OptionalTensorList;
    } else if (auto opt = type->cast<OptionalType>(); opt && opt->getElementType()->isSubtypeOf(*TensorType::get())) {
      info.arg_kind[i] = CpuFbArgKind::OptionalTensor;
    } else if (type->kind() == TypeKind::DeviceObjType) {
      info.arg_kind[i] = CpuFbArgKind::Device;
    }
    const auto* alias = a.alias_info();
    const bool is_w = (alias != nullptr && alias->isWrite());
    info.is_write_alias[i] = is_w;
    info.is_pure_out[i] = is_w && a.kwarg_only() && a.name() == "out";
  }
  info.populated = true;
  return info;
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

// Borrow a host pointer from the rbln virtual memory backing `t` and wrap it
// as a CPU tensor with the same sizes/strides. Writes the resulting borrow id
// into `borrow_id_out` (0 means "nothing to return"). Returns an undefined
// tensor if `t` isn't an rbln tensor — the caller falls back to the existing
// copy-based path for that slot.
at::Tensor borrow_rbln_as_cpu(const at::Tensor& t, uint64_t& borrow_id_out) {
  borrow_id_out = 0;
  if (!t.defined()) {
    return {};
  }
  if (t.device().type() != c10::DeviceType::PrivateUse1) {
    // CPU / other-device tensor: nothing to borrow.
    return {};
  }
  if (!t.is_contiguous()) {
    // Non-contiguous strided views may reach beyond `t.nbytes()` via stride
    // arithmetic; borrowing only `numel*itemsize` would alias an undersized
    // host region and corrupt reads. Caller falls back to the copy path.
    return {};
  }

  const uint64_t nbytes = t.nbytes();
  if (nbytes == 0) {
    // Zero-element / empty tensor: no host region to borrow. Use an empty CPU
    // tensor with matching dtype/shape instead.
    return at::empty(t.sizes(), t.options().device(at::kCPU));
  }

  auto borrowed = c10::rbln::borrow_host_ptr(t.data_ptr(), nbytes);
  borrow_id_out = borrowed.borrow_id;

  auto options = at::TensorOptions().dtype(t.dtype()).device(at::kCPU);
  return at::from_blob(reinterpret_cast<void*>(borrowed.host_ptr), t.sizes(), t.strides(), options);
}

// TensorList variant of borrow_rbln_as_cpu. Walks each element; rbln entries
// are borrowed (no D2H copy when host-latest), others fall through to the
// legacy batched at::_to_cpu. Borrow ids are appended to `borrow_ids_out` so
// the caller can release them after the op runs.
std::vector<at::Tensor> borrow_rbln_list_as_cpu(
    const std::vector<at::Tensor>& tensors,
    std::vector<uint64_t>& borrow_ids_out) {
  std::vector<at::Tensor> cpu_tensors(tensors.size());
  std::vector<at::Tensor> leftover; // non-rbln / empty — to_cpu later
  std::vector<size_t> leftover_indices;
  for (size_t i = 0; i < tensors.size(); ++i) {
    uint64_t bid = 0;
    cpu_tensors[i] = borrow_rbln_as_cpu(tensors[i], bid);
    borrow_ids_out.push_back(bid);
    if (bid == 0 && !cpu_tensors[i].defined()) {
      leftover.push_back(tensors[i]);
      leftover_indices.push_back(i);
    }
  }
  if (!leftover.empty()) {
    auto filled = to_cpu(leftover);
    for (size_t k = 0; k < filled.size(); ++k) {
      cpu_tensors[leftover_indices[k]] = std::move(filled[k]);
    }
  }
  return cpu_tensors;
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
  // Per-TensorList-arg borrow-id vectors. Entries with non-zero ids are
  // wrapped via the v-mem borrow path; zeros indicate the legacy `to_cpu`
  // fallback (used for non-rbln tensors, contiguity-guard skips, etc.).
  std::vector<std::vector<uint64_t>> tensorlist_borrow_ids;

  // Step 1: Convert all non-CPU tensor inputs into CPU tensors and put them
  // on the stack at the correct indices.
  //
  // Per-op schema info (arg-position kinds + alias-write flags) is cached: same
  // operator handle reuses the same FunctionSchema pointer for the process
  // lifetime, so a switch on the cached kind replaces the per-call IValue type
  // cascade. ~10 distinct ops in LLaMA-1B eager → first call populates, every
  // call after is a hashmap shared-lock.
  const auto& schema_info = get_or_populate_schema_info(op.schema());
  const auto n_args = arguments.size();
  for (size_t idx = 0; idx < n_args; ++idx) {
    const auto kind = schema_info.arg_kind[idx];
    if (kind == CpuFbArgKind::Other) {
      continue;
    }
    const auto& ivalue = arguments[idx];
    switch (kind) {
      case CpuFbArgKind::Tensor:
        tensor_args.push_back(ivalue.toTensor());
        tensor_args_indices.push_back(idx);
        break;
      case CpuFbArgKind::OptionalTensor:
        // `Tensor?` runtime IValue is Tensor when the optional has a value,
        // else None. Only stage the present case onto tensor_args; absent
        // (None) needs no transformation — the CPU kernel sees None too.
        if (ivalue.isTensor()) {
          tensor_args.push_back(ivalue.toTensor());
          tensor_args_indices.push_back(idx);
        }
        break;
      case CpuFbArgKind::TensorList: {
        tensorlist_args.push_back(ivalue.toTensorList());
        tensorlist_args_indices.push_back(idx);
        auto rbln_list = ivalue.toTensorVector();
        std::vector<uint64_t> bids;
        auto cpu_list = borrow_rbln_list_as_cpu(rbln_list, bids);
        tensorlist_borrow_ids.push_back(std::move(bids));
        auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(cpu_list));
        tensorlist_cpu_args.push_back(cpu_ivalue);
        (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
        break;
      }
      case CpuFbArgKind::OptionalTensorList: {
        optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
        optional_tensorlist_args_indices.push_back(idx);
        auto cpu_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(to_cpu(ivalue.toOptionalTensorVector())));
        optional_tensorlist_cpu_args.push_back(cpu_ivalue);
        (*stack)[arguments_begin + idx] = c10::IValue(cpu_ivalue);
        break;
      }
      case CpuFbArgKind::Device:
        tgt_device = ivalue.toDevice();
        (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(kCPU));
        break;
      case CpuFbArgKind::Other:
        break; // unreachable — gated above
    }
  }
  // Stage tensor args onto the stack as CPU views. We obtain a host pointer
  // into the existing vmem region (no D2H copy) and wrap it as a CPU tensor
  // via at::from_blob; borrow ids are tracked so we can release them after
  // the op runs. Slots that can't be borrowed (write-alias outputs, undefined
  // tensors, non-rbln tensors, contiguity-guard skips) fall through to the
  // batched `at::_to_cpu` copy path.
  std::vector<uint64_t> borrow_ids(tensor_args.size(), 0);
  std::vector<at::Tensor> cpu_tensors(tensor_args.size());

  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Skip the borrow fast path for write-alias outputs (`out=` tensors).
    // Borrowed tensors are wrapped via at::from_blob and have a fixed-size
    // host-mapped storage, so the CPU op's resize path (which fires for
    // wrong-shape `out=`, broadcasting binary ops, etc.) silently no-ops
    // and the op then writes broadcasted values into the unresized buffer.
    // Routing write-alias slots through the legacy `to_cpu` path gives
    // them fresh CPU storage that PyTorch core can resize freely.
    if (schema_info.is_write_alias[tensor_args_indices[i]]) {
      continue;
    }
    cpu_tensors[i] = borrow_rbln_as_cpu(tensor_args[i], borrow_ids[i]);
  }
  // Fill any slots that weren't borrowed (write-alias, undefined, non-rbln,
  // or contiguity guard) by routing them through the legacy batched copy.
  // Pure `out=` slots — kwarg-only, write-alias, named exactly "out" — get
  // fresh empty CPU storage instead of going through the batched D2H copy:
  // the CPU kernel will overwrite the buffer immediately, so D2H'ing the
  // device contents we're about to discard is wasted bandwidth.
  //
  // We deliberately stay narrow here (vs. all write-alias args). In-place
  // ops like `add_(self, other)` have a write-alias `self` that is also an
  // *input* the CPU kernel reads from; replacing it with empty storage
  // would feed garbage to the kernel. Kwarg-only `out=` is the schema shape
  // that's guaranteed pure output (the CPU op writes; never reads). Other
  // output kwarg names (e.g. `max.dim_max`'s `max`/`max_values`) need
  // per-schema audit before extending.
  //
  // Profiling on LLaMA-1B eager: this single change freed ~22% of
  // cpu_fallback_rbln host time — mean.out / rsqrt.out / pow.out's `out=`
  // was hitting the slow path 1122 times each.
  std::vector<at::Tensor> non_borrowed;
  std::vector<size_t> non_borrowed_indices;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    if (borrow_ids[i] != 0 || cpu_tensors[i].defined()) {
      continue;
    }
    const bool is_pure_out = schema_info.is_pure_out[tensor_args_indices[i]];
    if (is_pure_out && tensor_args[i].defined()) {
      // Direct output borrow: wrap the rbln out= tensor's host backing as a
      // CPU view. The CPU kernel writes straight into the rbln vmem region,
      // skipping the writeback memcpy entirely. release_borrowed(updated=true)
      // is issued automatically by the existing borrow-write path in Step 3.
      //
      // Safety conditions: contiguous + non-zero bytes + RBLN device. PyTorch
      // out= callers pass a tensor whose shape matches the result; if the
      // CPU kernel ever tries to resize_(), `from_blob`'s fixed-size mapping
      // silently no-ops and we'd corrupt the next slot, so we keep the
      // size/contig guards strict and fall back to fresh empty otherwise.
      auto& rbln_out = tensor_args[i];
      const uint64_t nbytes = rbln_out.nbytes();
      if (rbln_out.device().type() == c10::DeviceType::PrivateUse1 && rbln_out.is_contiguous() && nbytes > 0) {
        auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_out.data_ptr(), nbytes);
        borrow_ids[i] = borrowed.borrow_id;
        auto opts = at::TensorOptions().dtype(rbln_out.dtype()).device(at::kCPU);
        cpu_tensors[i] =
            at::from_blob(reinterpret_cast<void*>(borrowed.host_ptr), rbln_out.sizes(), rbln_out.strides(), opts);
      } else {
        // Resize-safe fallback: fresh CPU empty.
        cpu_tensors[i] = at::empty(rbln_out.sizes(), rbln_out.options().device(at::kCPU));
      }
    } else {
      non_borrowed.push_back(tensor_args[i]);
      non_borrowed_indices.push_back(i);
    }
  }
  if (!non_borrowed.empty()) {
    auto filled = to_cpu(non_borrowed);
    for (size_t k = 0; k < filled.size(); ++k) {
      cpu_tensors[non_borrowed_indices[k]] = std::move(filled[k]);
    }
  }

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  // Step 2: Call the underlying CPU implementation of the operator.
  //
  // Before redispatching, consult :class:`CPUFastPathRegistry` — handlers
  // registered under aten/src/ATen/native/rbln/fast_paths/*.cpp self-register
  // a specialized host micro-kernel for a single op (e.g. rsqrt.out,
  // pow.Tensor_Scalar_out with exp==2, mean.out reducing last dim). The
  // handler runs its own dtype/contig/shape guard; on match it writes the
  // result directly into the borrowed out-tensor buffer and replaces
  // ``stack`` with the single return. On no-match it returns ``false`` and
  // we fall through to the generic boxed dispatcher.
  // If anything in Step 2 (dispatch) or Step 3 (write-back) throws — e.g.
  // PyTorch raises "result type X can't be cast to Y" for a dtype-mismatched
  // out=, or _copy_from_and_resize trips when the caller passes a wrong-device
  // out= that survived to step 3 — the borrows acquired above must still be
  // released. Otherwise the rbln vmem manager keeps the input/out vaddrs in
  // the borrowed state, and the next free on that vaddr fails. Since
  // c10::rbln::free throws from a c10::DataPtr deleter (called from
  // ~TensorImpl, a noexcept context), that failure escalates to std::terminate.
  // RAII guard releases every still-live borrow as `updated=false` on
  // destruction; the normal step-4 release loop clears borrow_ids[*] to 0
  // before this guard goes out of scope on the happy path, so the guard is a
  // no-op when nothing threw.
  struct BorrowReleaseGuard {
    std::vector<uint64_t>& borrow_ids;
    std::vector<std::vector<uint64_t>>& tensorlist_borrow_ids;
    ~BorrowReleaseGuard() {
      for (auto& bid : borrow_ids) {
        if (bid != 0) {
          try {
            c10::rbln::return_borrowed(bid, /*updated=*/false);
          } catch (...) {
            // Best-effort: dtor is noexcept by default. Swallow runtime
            // rejections so we don't escalate one borrow-release failure into
            // std::terminate; the original exception that triggered cleanup
            // is more useful for diagnosis.
          }
          bid = 0;
        }
      }
      for (auto& bids : tensorlist_borrow_ids) {
        for (auto& bid : bids) {
          if (bid != 0) {
            try {
              c10::rbln::return_borrowed(bid, /*updated=*/false);
            } catch (...) {
            }
            bid = 0;
          }
        }
      }
    }
  } _borrow_guard{borrow_ids, tensorlist_borrow_ids};

  auto fast_path_fn = CPUFastPathRegistry::instance().try_get(op.schema());
  const bool fast_path_taken = fast_path_fn != nullptr && fast_path_fn(cpu_tensors, stack, arguments_begin);
  if (!fast_path_taken) {
    op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);
  }

  // Step 3: Mutable alias write-back.
  // - Legacy path: the CPU op wrote into the fresh CPU tensor at cpu_tensors[i];
  //   we copy that result back into the original rbln tensor.
  // - Borrow path, in-place case (borrow_ids[i] != 0): the CPU op wrote
  //   directly into the borrowed host pointer, which already aliases the rbln
  //   tensor's host-backed vmem; we only need to mark the borrow as updated so
  //   the next device consumer lazily syncs.
  // - Borrow path, resized-empty case: the composite's functional wrapper
  //   allocates an empty out (numel=0) and lets the CPU kernel resize+fill it.
  //   borrow_rbln_as_cpu couldn't borrow a zero-size vaddr, so borrow_id is 0
  //   and cpu_tensors[i] holds fresh CPU storage. We resize the original rbln
  //   tensor to match, borrow its now-sized vmem, memcpy the CPU content, and
  //   return the borrow as updated — replacing the eager H2D that
  //   at::_copy_from_and_resize would do.
  std::vector<bool> borrow_write(tensor_args.size(), false);
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    if (!schema_info.is_write_alias[tensor_args_indices[i]]) {
      continue;
    }
    if (!tensor_args[i].defined()) {
      continue;
    }
    if (borrow_ids[i] != 0) {
      // The CPU kernel may have shrink-resized cpu_tensors[i] — a from_blob
      // view wrapping the rbln_out vmem — when the caller passed an out= with
      // wrong shape (PyTorch's "resize with warning" path). The shrink writes
      // the new-shape data into the borrow host_ptr, but rbln_out's tensor
      // metadata still reports the pre-resize shape. Sync sizes so downstream
      // consumers (and the test's shape assertion) see the resized output.
      if (cpu_tensors[i].sizes() != tensor_args[i].sizes()) {
        tensor_args[i].resize_(cpu_tensors[i].sizes());
      }
      borrow_write[i] = true;
      continue;
    }

    // Borrow-based write-back is only safe when BOTH the cpu_out and the rbln
    // out are contiguous. The cpu->host memcpy below copies `nbytes` of
    // contiguous bytes; if `rbln_out` were strided/noncontiguous, the borrow
    // would alias only the [vaddr, vaddr+nbytes) span — leaving the rest of
    // the strided storage stale. Caller's noncontiguous out= must go through
    // the legacy `_copy_from_and_resize` path which honors strides.
    const bool borrow_resize_case = tensor_args[i].device().type() == c10::DeviceType::PrivateUse1 &&
        cpu_tensors[i].defined() && cpu_tensors[i].is_contiguous() && cpu_tensors[i].nbytes() > 0 &&
        tensor_args[i].is_contiguous();
    if (borrow_resize_case) {
      auto& rbln_out = tensor_args[i];
      if (rbln_out.sizes() != cpu_tensors[i].sizes()) {
        rbln_out.resize_(cpu_tensors[i].sizes());
      }
      const uint64_t nbytes = rbln_out.nbytes();
      if (nbytes > 0) {
        // Acquire-for-overwrite: we immediately memcpy over the whole region, so any
        // D2H from a stale PHYSICAL_VIEW_IS_LATEST state would be thrown away.
        auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_out.data_ptr(), nbytes);
        std::memcpy(reinterpret_cast<void*>(borrowed.host_ptr), cpu_tensors[i].data_ptr(), nbytes);
        c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
      }
    } else {
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of tensors. On the borrow path any element with a non-zero borrow id will
  // be committed via `rbln_v_return_borrowed(updated=true)` at the release
  // step; for those we just mark instead of doing the eager H2D copy.
  std::vector<std::vector<bool>> tensorlist_borrow_write(tensorlist_args_indices.size());
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    const auto& cpu_list = tensorlist_cpu_args[i].toTensorVector();
    tensorlist_borrow_write[i].assign(tensorlist_args[i].size(), false);
    if (alias_info != nullptr && alias_info->isWrite()) {
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        if (!cpu_list[idx].defined())
          continue;
        const bool is_borrowed = i < tensorlist_borrow_ids.size() && idx < tensorlist_borrow_ids[i].size() &&
            tensorlist_borrow_ids[i][idx] != 0;
        if (is_borrowed) {
          tensorlist_borrow_write[i][idx] = true;
        } else {
          at::_copy_from_and_resize(cpu_list[idx], tensorlist_args[i][idx]);
        }
      }
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of optional tensors
  for (const auto i : c10::irange(optional_tensorlist_args_indices.size())) {
    auto tensorlist_idx = optional_tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = optional_tensorlist_cpu_args[i].toOptionalTensorList();
      for (const auto idx : c10::irange(optional_tensorlist_args[i].size())) {
        if (cpu_tensors[idx].has_value() && cpu_tensors[idx].value().defined()) {
          const std::optional<at::Tensor>& optional_tensor = optional_tensorlist_args[i][idx];
          at::_copy_from_and_resize(cpu_tensors[idx].value(), optional_tensor.value());
        }
      }
    }
  }

  // Release any vmem borrows issued on the input path. Write-alias inputs use
  // `updated=true` so the rbln tensor's host view becomes the latest source of
  // truth and the next device consumer triggers a lazy host→device sync.
  // Zero out each id after release so the BorrowReleaseGuard above does not
  // double-release on its way out.
  for (size_t i = 0; i < borrow_ids.size(); ++i) {
    c10::rbln::return_borrowed(borrow_ids[i], borrow_write[i]);
    borrow_ids[i] = 0;
  }
  for (size_t i = 0; i < tensorlist_borrow_ids.size(); ++i) {
    for (size_t k = 0; k < tensorlist_borrow_ids[i].size(); ++k) {
      const bool upd = (i < tensorlist_borrow_write.size() && k < tensorlist_borrow_write[i].size())
          ? tensorlist_borrow_write[i][k]
          : false;
      c10::rbln::return_borrowed(tensorlist_borrow_ids[i][k], upd);
      tensorlist_borrow_ids[i][k] = 0;
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
      // Copy the cpu output tensor to the original device. On the v-mem
      // borrow path we allocate a fresh rbln tensor, borrow its host-backed
      // view, memcpy the CPU result into it, and return the borrow with
      // updated=true — that way the next device consumer triggers a lazy
      // host→device sync instead of an eager H2D copy here.

      // We technically  might not have a target device, e.g. if you call
      // torch.cat() with an empty list In that case, we shouldn't have any
      // tensors to schlep across devices anyway.
      if (tgt_device) {
        const bool use_borrow_out = tgt_device->type() == c10::DeviceType::PrivateUse1;
        if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
          const auto& cpu_out = returns[idx].toTensor();
          if (use_borrow_out && cpu_out.is_contiguous()) {
            auto rbln_out = at::empty(cpu_out.sizes(), cpu_out.options().device(*tgt_device));
            const uint64_t nbytes = rbln_out.nbytes();
            if (nbytes > 0) {
              // Acquire-for-overwrite: the tensor is a freshly-allocated at::empty;
              // any pre-existing data (in practice there is none, but if the
              // allocator ever caches a warm buffer the old device data is
              // irrelevant) will be overwritten by the memcpy below.
              auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_out.data_ptr(), nbytes);
              std::memcpy(reinterpret_cast<void*>(borrowed.host_ptr), cpu_out.data_ptr(), nbytes);
              c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
            }
            (*stack)[returns_begin + idx] = c10::IValue(rbln_out);
          } else {
            (*stack)[returns_begin + idx] = c10::IValue(cpu_out.to(*tgt_device));
          }
        } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
          // TensorList output: keep the legacy .to(device) path for now; the
          // borrow-based write-back can be extended here if profiling shows
          // meaningful cost.
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
