#include <torch_rbln/csrc/rbln/DispatchShim.h>
#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/library.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace torch_rbln::shim {

namespace {

using warmcache::CacheEntry;
using warmcache::CacheKey;
using warmcache::OutputProfile;
using warmcache::ScalarValue;
using warmcache::TensorProfile;
using warmcache::WarmCache;

// Cached per-op schema summary so we don't re-walk FunctionSchema::arguments()
// on every dispatch. Populated on the first invocation of a given op and
// looked up with the same registry key thereafter.
struct SchemaCache {
  std::vector<bool> is_kwarg_only; // parallel to schema args
  std::vector<std::string> arg_names; // only populated for kwarg_only slots
  std::vector<bool> is_write_alias; // alias_info != nullptr && isWrite()
  int out_positional_idx = -1; // -1 if no arg is named "out"
  size_t num_args = 0;
  size_t num_positional = 0;
  std::vector<c10::TypePtr> return_types; // parallel to schema returns
  bool populated = false;
};

struct ShimEntry {
  pybind11::object py_fn;
  std::vector<size_t> skip_dtype_args;
  SchemaCache schema_cache; // lazily filled
  const char* op_name_intern = nullptr; // stable pointer for WarmCache keys
};

// Leaky singletons: these hold pybind11::object (registry) and torch::Library
// (installed_libs), both of which keep Python state alive. A regular
// `static T x;` runs its destructor *after* Py_Finalize() during process
// teardown, which decrefs Python objects on a finalized interpreter and
// aborts inside libpython. Allocate with `new` so the storage outlives
// Python finalize; the OS reclaims it at exit.
std::unordered_map<std::string, ShimEntry>& registry() {
  static auto* r = new std::unordered_map<std::string, ShimEntry>();
  return *r;
}

std::vector<std::unique_ptr<torch::Library>>& installed_libs() {
  static auto* v = new std::vector<std::unique_ptr<torch::Library>>();
  return *v;
}

// Guards registry-level mutations (register_cpp_shim). Per-entry schema_cache
// is populated on first dispatch and read unlocked thereafter — populated is
// written last, so readers that see populated=true observe a consistent cache.
std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

ShimEntry* find_shim_entry(const std::string& op_name) {
  auto& r = registry();
  auto it = r.find(op_name);
  return it != r.end() ? &it->second : nullptr;
}

void populate_schema_cache(SchemaCache& cache, const c10::FunctionSchema& schema) {
  const auto& args = schema.arguments();
  const auto& returns = schema.returns();
  cache.num_args = args.size();
  cache.is_kwarg_only.resize(args.size());
  cache.arg_names.resize(args.size());
  cache.is_write_alias.resize(args.size());
  size_t n_pos = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    cache.is_kwarg_only[i] = args[i].kwarg_only();
    cache.arg_names[i] = args[i].name();
    const auto* alias_info = args[i].alias_info();
    cache.is_write_alias[i] = (alias_info != nullptr && alias_info->isWrite());
    if (!cache.is_kwarg_only[i]) {
      ++n_pos;
    }
    if (cache.arg_names[i] == "out") {
      cache.out_positional_idx = static_cast<int>(i);
    }
  }
  cache.num_positional = n_pos;
  cache.return_types.reserve(returns.size());
  for (const auto& r : returns) {
    cache.return_types.push_back(r.type());
  }
  cache.populated = true;
}

bool is_skipped_arg(const std::vector<size_t>& skip_list, size_t i) {
  for (auto idx : skip_list) {
    if (idx == i) {
      return true;
    }
  }
  return false;
}

// Cheap C++-side pre-check mirroring the cheap branches of
// torch_rbln._internal.ops_utils.is_cpu_fallback_cases():
//   2. dtype != float16 on any input tensor
//   3. all input tensors are scalar (ndim == 0)
//   4. any input tensor is_contiguous() with storage_offset != 0
//
// Inputs means args NOT schema-marked as write aliases (out-tensor skipped).
// `skip_dtype_args` indexes positional args whose dtype check is ignored (e.g.
// where.self_out's cond, which is bool).
//
// **Wrapped 0-dim Tensors are skipped** from the dtype check. PyTorch's Python
// frontend wraps Python scalars (`1.0` in `tensor + 1.0`) as 0-dim tensors with
// the `is_wrapped_number` flag set; on the way to the Python shim's `add_rbln`
// wrapper, `torch::jit::toPyObject` unwraps such tensors back into Python
// scalars (via `.item()`) so the Python wrapper sees only the real tensor and
// avoids the dtype-mismatch fallback — `chunk + 1.0` runs on the RBLN compile
// path, not CPU. If we counted wrapped 0-dim against the shortcut here, we
// would force the shortcut for the most common binary-op-with-python-scalar
// case and bypass the compile-path that the test suite expects.
bool quick_fallback_check(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    const std::vector<size_t>& skip_dtype_args) {
  auto args = torch::jit::last(stack, cache.num_args);
  bool has_input_tensor = false;
  bool all_input_scalar = true;
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = args[i];
    if (!iv.isTensor()) {
      continue;
    }
    const auto& t = iv.toTensor();
    if (!t.defined()) {
      continue;
    }
    if (cache.is_write_alias[i]) {
      continue;
    }
    // NOTE: storage_offset != 0 contiguous inputs are NOT short-circuited to
    // cpu_fallback_rbln here. The Python wrapper's cpu_fallback_path takes a
    // different host-copy route (tensor.cpu()) than at::_to_cpu via
    // op.redispatchBoxed(CPU); for the storage_offset>0 case the latter
    // produces partially-corrupted reads on some rbln runtime builds. Let the
    // shim fall through to the Python wrapper which dispatches to
    // cpu_fallback_path.
    if (is_skipped_arg(skip_dtype_args, i)) {
      continue;
    }
    // Wrapped 0-dim numbers behave like Python scalars and are unwrapped by
    // the pybind boundary. Skip them from the dtype check so the shortcut
    // doesn't fire for `tensor + 1.0` etc.
    if (t.dim() == 0 && t.unsafeGetTensorImpl()->is_wrapped_number()) {
      continue;
    }
    has_input_tensor = true;
    if (t.scalar_type() != c10::kHalf) {
      return true;
    }
    if (t.dim() != 0) {
      all_input_scalar = false;
    }
  }
  return has_input_tensor && all_input_scalar;
}

// ---------------------------------------------------------------------------
// Warm-cache integration
// ---------------------------------------------------------------------------

// Extract a ScalarValue from an IValue for cache keying. Returns Missing for
// anything that isn't a plain scalar (tensors, None, lists, etc.) since those
// don't contribute to the warm-cache key: tensor profiles are already captured
// as TensorProfile; None/list args mean the schema uses an uncommon overload
// shape that we don't currently warm-cache.
ScalarValue ival_to_scalar(const c10::IValue& iv) {
  if (iv.isInt())
    return ScalarValue::fromInt(iv.toInt());
  if (iv.isDouble())
    return ScalarValue::fromFloat(iv.toDouble());
  if (iv.isBool())
    return ScalarValue::fromBool(iv.toBool());
  if (iv.isScalar()) {
    const auto& s = iv.toScalar();
    if (s.isIntegral(false))
      return ScalarValue::fromInt(s.toLong());
    if (s.isFloatingPoint())
      return ScalarValue::fromFloat(s.toDouble());
    if (s.isBoolean())
      return ScalarValue::fromBool(s.toBool());
  }
  return ScalarValue::missing();
}

// Build a WarmCache::CacheKey from the current stack's last num_args IValues.
// Tensor args (non-write-alias, defined) become TensorProfiles in their
// positional order. Scalar args become ScalarValues. None/Tensor-list args
// are silently treated as a signal that we cannot warm-cache this call
// (return false; caller skips warm cache and falls through to pybind).
bool build_cache_key(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    const char* op_name_intern,
    CacheKey& out_key) {
  out_key.schema_name_intern = op_name_intern;
  out_key.inputs.clear();
  out_key.scalars.clear();

  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = arguments[i];
    if (iv.isTensor()) {
      const at::Tensor& t = iv.toTensor();
      if (!t.defined())
        continue;
      if (cache.is_write_alias[i])
        continue; // out tensor, not part of key
      TensorProfile tp;
      tp.dtype = t.scalar_type();
      tp.shape.assign(t.sizes().begin(), t.sizes().end());
      tp.device_index = static_cast<int8_t>(t.device().index());
      out_key.inputs.emplace_back(std::move(tp));
    } else if (iv.isNone()) {
      // Treat `None` slot as a Missing scalar — keeps positional structure
      // without requiring us to distinguish "optional scalar absent" from
      // "optional tensor absent"; both just miss if later calls differ.
      out_key.scalars.push_back(ScalarValue::missing());
    } else if (iv.isTensorList() || iv.isList()) {
      // Lists are not handled by the warm-cache path yet (no shim op uses
      // them). Bail out: caller falls through to pybind.
      return false;
    } else {
      out_key.scalars.push_back(ival_to_scalar(iv));
    }
  }
  return true;
}

// Thread-local context that ties a just-computed CacheKey (built before the
// pybind miss-path call) to the later pybind-exposed install_pending hook
// called from the Python wrapper after it compiles. This avoids re-walking
// the args from Python to reconstruct the key.
struct PendingInstall {
  bool valid = false;
  const char* op_name_intern = nullptr;
  CacheKey key;
};

thread_local PendingInstall t_pending;

// Take ownership of the pending context (single reader); installer clears it.
PendingInstall take_pending() {
  PendingInstall p = std::move(t_pending);
  t_pending.valid = false;
  t_pending.op_name_intern = nullptr;
  return p;
}

// Hot path: look up the warm-cache entry for `key` and, on hit, drive rebel's
// PyRblnSyncRuntime directly from C++ — no pybind, no Python wrapper. Returns
// true iff the hit path was taken and the stack has been left with the proper
// return value.
//
// Currently supports the shape:
//   - single output (schema.returns().size() == 1)
//   - output is either: (a) a write-alias out= arg the caller passed in, or
//                        (b) a freshly allocated tensor per the cached profile
// Extended support (TensorLists, multi-output) can be added with parallel
// codepaths — they're not on any shim op today.
bool try_warmcache_hit(torch::jit::Stack* stack, const SchemaCache& cache, const CacheKey& key) {
  auto& wc = WarmCache::instance();
  if (!wc.is_enabled() || WarmCache::is_building_entry())
    return false;
  if (cache.return_types.size() != 1)
    return false;

  const CacheEntry* entry = wc.find(key);
  if (entry == nullptr)
    return false;

  // Build input-ptr map in the order tensor inputs appear on the stack.
  std::map<uint32_t, uint64_t> dev_in;
  std::map<uint32_t, uintptr_t> cpu_in;

  auto arguments = torch::jit::last(stack, cache.num_args);
  at::Tensor out_tensor;
  uint32_t in_idx = 0;
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = arguments[i];
    if (!iv.isTensor())
      continue;
    const at::Tensor& t = iv.toTensor();
    if (!t.defined())
      continue;
    if (cache.is_write_alias[i]) {
      out_tensor = t;
      continue;
    }
    // Safety: a tensor with data_ptr() == 0 has no backing v-memory yet
    // (e.g. an alias produced by a previous op whose materialization is
    // pending). Passing 0 to PrepareInputs trips the rebel runtime's
    // `Invalid key_vaddr=0` guard. Fall back to the pybind path so the
    // Python wrapper can force materialization (via to_cpu/contig/etc.)
    // and still produce a correct result.
    void* ptr = t.data_ptr();
    if (ptr == nullptr) {
      return false;
    }
    dev_in.emplace(in_idx++, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
  }

  // For non-out ops (e.g. max.unary, min.unary with no overload), allocate a
  // fresh output tensor per the cached profile.
  if (!out_tensor.defined()) {
    if (entry->out_profiles.empty())
      return false;
    const OutputProfile& op0 = entry->out_profiles[0];
    if (!op0.is_rbln_device)
      return false; // CPU output unsupported on hit path
    int8_t dev_idx = 0;
    if (!key.inputs.empty())
      dev_idx = key.inputs.front().device_index;
    auto device = c10::Device(c10::DeviceType::PrivateUse1, dev_idx);
    out_tensor = at::empty(op0.shape, at::TensorOptions().dtype(op0.dtype).device(device));
  }

  std::map<uint32_t, uint64_t> dev_out;
  std::map<uint32_t, uintptr_t> cpu_out;
  void* out_ptr = out_tensor.data_ptr();
  if (out_ptr == nullptr) {
    // Same materialization concern as inputs (see dev_in loop above).
    return false;
  }
  dev_out.emplace(0u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out_ptr)));

  // rebel's PyRblnSyncRuntime methods are wrapped via pybind; they may set a
  // Python exception on failure rather than throw a C++ exception. Acquire
  // the GIL so we can both call them safely and inspect ``PyErr_Occurred``
  // after each call to detect the soft failure path. On failure we clear
  // the Python error and return false so the caller falls through to the
  // pybind miss path (which routes through DynamoRuntime and performs the
  // v-memory bookkeeping that lets the same tensor inputs succeed).
  pybind11::gil_scoped_acquire wc_gil;
  bool runtime_failed = false;
  auto clear_and_fail = [&]() {
    if (PyErr_Occurred())
      PyErr_Clear();
    runtime_failed = true;
  };
  try {
    entry->runtime->PrepareInputs(dev_in, cpu_in);
    if (PyErr_Occurred()) {
      clear_and_fail();
    }
    if (!runtime_failed) {
      entry->runtime->PrepareOutputs(dev_out, cpu_out);
      if (PyErr_Occurred()) {
        clear_and_fail();
      }
    }
    if (!runtime_failed) {
      entry->runtime->Run();
      if (PyErr_Occurred()) {
        clear_and_fail();
      }
    }
  } catch (const pybind11::error_already_set&) {
    clear_and_fail();
  } catch (const std::exception&) {
    clear_and_fail();
  } catch (...) {
    clear_and_fail();
  }
  if (runtime_failed) {
    // The runtime that we cached cannot serve this profile after all (e.g.
    // input v-memory was created by an allocation path the runtime can't
    // resolve). Drop the entry so subsequent dispatches with this key go
    // through the pybind miss path and rebuild via DynamoRuntime, which
    // handles the edge case correctly.
    WarmCache::instance().erase(key);
    return false;
  }

  // Pop args, push single return.
  torch::jit::drop(stack, cache.num_args);
  torch::jit::push(stack, out_tensor);
  return true;
}

// The boxed kernel that Library::impl points at for every shimmed op.
void generic_shim_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // Build the fully-qualified key as "<namespace>::<name>[.overload]" so it
  // matches what register_cpp_shim stored (e.g. "aten::add.out").
  std::string op_name = op.schema().name();
  const auto& overload = op.schema().overload_name();
  if (!overload.empty()) {
    op_name += "." + overload;
  }

  ShimEntry* entry = nullptr;
  {
    std::lock_guard<std::mutex> lk(registry_mutex());
    entry = find_shim_entry(op_name);
    TORCH_CHECK(entry != nullptr, "No Python impl registered for shim op: ", op_name);
    if (!entry->schema_cache.populated) {
      populate_schema_cache(entry->schema_cache, op.schema());
    }
  }

  const SchemaCache& cache = entry->schema_cache;
  const auto& skip_dtype_args = entry->skip_dtype_args;
  const char* op_name_intern = entry->op_name_intern;

  // The C++ precheck identifies cheap "must fallback" cases (real non-fp16
  // input or all-0-dim) and short-circuits straight into cpu_fallback_rbln,
  // bypassing the pybind hop into the Python wrapper. The wrapped-0-dim
  // case (e.g. `tensor + 1.0` where PyTorch wraps `1.0` as a 0-dim CPU
  // tensor with `is_wrapped_number`) is intentionally excluded from the
  // precheck — see quick_fallback_check — because the pybind boundary
  // unwraps such tensors back to Python scalars, so the Python wrapper sees
  // a single tensor arg and routes through the RBLN compile path; if we
  // shortcut those calls into cpu_fallback_rbln we'd skip that compile
  // path and get bit-different fp16 rounding than the surrounding
  // RBLN-compiled ops produce.
  //
  // Earlier bugs that motivated disabling this shortcut have been fixed at
  // the borrow site: write-alias args are skipped from the borrow loop and
  // the borrow_resize_case is gated on contiguity (see RBLNCPUFallback.cpp).
  const bool would_fallback = quick_fallback_check(stack, cache, skip_dtype_args);
  if (would_fallback) {
    ::at::native::rbln::cpu_fallback_rbln(op, stack);
    return;
  }

  // Warm-cache hot path: if we've previously compiled this op for an identical
  // input profile and have the rebel runtime cached, drive the runtime from
  // C++ directly.
  CacheKey key;
  const bool key_ok = build_cache_key(stack, cache, op_name_intern, key);
  if (key_ok && try_warmcache_hit(stack, cache, key)) {
    return;
  }

  // MISS path: set up thread-local pending install so the Python wrapper can
  // call `_warmcache_install_pending(runtime, out_profiles)` once it finishes
  // compile + first run. The pending context is discarded unconditionally at
  // the end of this function (even on failure / exception) to avoid leaking
  // into subsequent unrelated ops on the same thread.
  if (key_ok) {
    t_pending.valid = true;
    t_pending.op_name_intern = op_name_intern;
    t_pending.key = std::move(key);
  } else {
    t_pending.valid = false;
  }

  pybind11::gil_scoped_acquire gil;

  // Build args in a single pass into a pre-sized py::tuple (skip the list →
  // tuple copy) and a kwargs dict. Holds borrowed refs to the py_fn so the
  // registry mutex isn't needed during the Python call.
  pybind11::object py_fn_copy = entry->py_fn;

  pybind11::tuple pos_tup(cache.num_positional);
  pybind11::dict kwargs;
  pybind11::object out_obj = pybind11::none();
  size_t pos_idx = 0;

  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    pybind11::object val = torch::jit::toPyObject(arguments[i]);
    if (cache.is_kwarg_only[i]) {
      kwargs[cache.arg_names[i].c_str()] = val;
      if (static_cast<int>(i) == cache.out_positional_idx) {
        out_obj = val;
      }
    } else {
      pos_tup[pos_idx++] = val;
    }
  }

  pybind11::object result;
  try {
    result = py_fn_copy(*pos_tup, **kwargs);
  } catch (...) {
    t_pending.valid = false; // scrub stale context on exception
    throw;
  }

  // Drop pending regardless of what Python did (install_pending, if called,
  // already cleared t_pending via take_pending()).
  t_pending.valid = false;

  torch::jit::drop(stack, cache.num_args);

  if (cache.return_types.empty()) {
    return;
  }
  if (cache.return_types.size() == 1) {
    if (result.is_none() && !out_obj.is_none()) {
      // Out-variant where the Python impl mutates `out` in place and returns
      // None; the schema return is `Tensor(a!)` and we push the out arg.
      auto iv = torch::jit::toIValue(out_obj, cache.return_types[0]);
      torch::jit::push(stack, iv);
    } else {
      auto iv = torch::jit::toIValue(result, cache.return_types[0]);
      torch::jit::push(stack, iv);
    }
    return;
  }
  pybind11::tuple tup = result.cast<pybind11::tuple>();
  TORCH_CHECK(
      tup.size() == cache.return_types.size(),
      "Python impl returned ",
      tup.size(),
      " values but schema expects ",
      cache.return_types.size());
  for (size_t i = 0; i < cache.return_types.size(); ++i) {
    pybind11::object v = tup[i];
    auto iv = torch::jit::toIValue(v, cache.return_types[i]);
    torch::jit::push(stack, iv);
  }
}

// Extract the overload-qualified name `foo.bar` from a fully-qualified
// operator name `ns::foo.bar`.
std::string strip_namespace(const std::string& op_name) {
  const auto pos = op_name.find("::");
  if (pos == std::string::npos) {
    return op_name;
  }
  return op_name.substr(pos + 2);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void register_cpp_shim(const std::string& op_name, pybind11::object py_fn, const std::vector<size_t>& skip_dtype_args) {
  std::lock_guard<std::mutex> lk(registry_mutex());

  const char* interned = warmcache::intern_op_name(op_name);

  const bool first_time = registry().find(op_name) == registry().end();
  registry()[op_name] = ShimEntry{std::move(py_fn), skip_dtype_args, SchemaCache{}, interned};
  if (!first_time) {
    // Same op re-registered (e.g. codegen re-run during tests): reuse existing
    // Library entry, just refresh the stored Python callable above.
    return;
  }

  auto lib = std::make_unique<torch::Library>(
      torch::Library::IMPL,
      "aten",
      std::optional<c10::DispatchKey>(c10::DispatchKey::PrivateUse1),
      __FILE__,
      static_cast<uint32_t>(__LINE__));
  const std::string overload = strip_namespace(op_name);
  lib->impl(overload.c_str(), torch::CppFunction::makeFromBoxedFunction<&generic_shim_boxed>());
  installed_libs().push_back(std::move(lib));
}

// ---------------------------------------------------------------------------
// Warm-cache install hook, called from Python after a successful miss-path
// compile. `out_profiles` is a list of (shape, dtype_str, is_rbln) tuples
// computed by the Python wrapper from the post-compile output tensors.
// ---------------------------------------------------------------------------

bool install_warmcache_from_pending(
    pybind11::object dyn_runtime,
    pybind11::int_ runtime_raw_ptr,
    uint32_t num_inputs,
    uint32_t num_outputs,
    const std::vector<std::tuple<std::vector<int64_t>, std::string, bool>>& out_profiles) {
  PendingInstall p = take_pending();
  if (!p.valid)
    return false;

  CacheEntry entry;
  entry.py_dyn_runtime = std::move(dyn_runtime);
  void* runtime_void_ptr = PyLong_AsVoidPtr(runtime_raw_ptr.ptr());
  if (runtime_void_ptr == nullptr) {
    if (PyErr_Occurred()) {
      PyErr_Clear();
    }
    return false;
  }
  entry.runtime = static_cast<::rbln::PyRblnSyncRuntime*>(runtime_void_ptr);
  entry.num_inputs = num_inputs;
  entry.num_outputs = num_outputs;
  entry.out_profiles.reserve(out_profiles.size());
  for (const auto& tup : out_profiles) {
    OutputProfile op;
    op.shape.assign(std::get<0>(tup).begin(), std::get<0>(tup).end());
    const std::string& dtype_s = std::get<1>(tup);
    // Same table as the reference's dtype_from_rbln_string, kept local to
    // avoid a dependency across files.
    if (dtype_s == "float16" || dtype_s == "torch.float16")
      op.dtype = at::kHalf;
    else if (dtype_s == "float32" || dtype_s == "torch.float32")
      op.dtype = at::kFloat;
    else if (dtype_s == "bfloat16" || dtype_s == "torch.bfloat16")
      op.dtype = at::kBFloat16;
    else if (dtype_s == "int64" || dtype_s == "torch.int64")
      op.dtype = at::kLong;
    else if (dtype_s == "int32" || dtype_s == "torch.int32")
      op.dtype = at::kInt;
    else if (dtype_s == "int16" || dtype_s == "torch.int16")
      op.dtype = at::kShort;
    else if (dtype_s == "int8" || dtype_s == "torch.int8")
      op.dtype = at::kChar;
    else if (dtype_s == "uint8" || dtype_s == "torch.uint8")
      op.dtype = at::kByte;
    else if (dtype_s == "bool" || dtype_s == "torch.bool")
      op.dtype = at::kBool;
    else
      return false; // unknown dtype: don't install
    op.is_rbln_device = std::get<2>(tup);
    entry.out_profiles.emplace_back(std::move(op));
  }

  WarmCache::instance().install(std::move(p.key), entry);
  return true;
}

} // namespace torch_rbln::shim
