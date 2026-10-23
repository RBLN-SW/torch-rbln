#include <torch_rbln/csrc/rbln/DispatchShim.h>
#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <rebel/v2/runtime/flags.h>
#include <rebel/v2/runtime/precision.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch_rbln::shim {

// ---------------------------------------------------------------------------
// DIAG counters: dispatch path classification. Populated on every call into
// the boxed shim handler. Externally readable via diag_dump_dispatch_paths().
// ---------------------------------------------------------------------------
namespace {
std::atomic<uint64_t> g_diag_n_total{0}; // total generic_shim_boxed invocations
std::atomic<uint64_t> g_diag_n_fallback{0}; // would_fallback=true → cpu_fallback_rbln
std::atomic<uint64_t> g_diag_n_warm_hit{0}; // warm-cache fast path hit
std::atomic<uint64_t> g_diag_n_miss{0}; // Python compile (miss) path
std::atomic<uint64_t> g_diag_ns_warm_hit{0}; // total ns inside warm-cache hit path
std::atomic<uint64_t> g_diag_ns_miss{0}; // total ns inside miss path
std::atomic<uint64_t> g_diag_ns_fallback{0}; // total ns inside cpu_fallback_rbln (the COST, not just count)
// cpu_fallback reason histogram. index = reason code from quick_fallback_check
// (1=dtype-not-fp16, 2=nan/inf input, 3=all-scalar). Bumped on the fallback
// branch only (the reason is already computed there) -> ON==OFF preserved.
std::array<std::atomic<uint64_t>, 4> g_fallback_reason{};

// Warm-cache hit path per-segment timers. Accumulated only on successful hits
// so per-segment averages = ns_X / n_hits give the steady-state breakdown.
std::atomic<uint64_t> g_diag_warm_n_hits{0};
std::atomic<uint64_t> g_diag_warm_ns_lookup{0};
std::atomic<uint64_t> g_diag_warm_ns_run{0};
std::atomic<uint64_t> g_diag_warm_ns_finalize{0};
} // namespace

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_dispatch_paths() {
  return std::make_tuple(
      g_diag_n_total.load(std::memory_order_relaxed),
      g_diag_n_fallback.load(std::memory_order_relaxed),
      g_diag_n_warm_hit.load(std::memory_order_relaxed),
      g_diag_n_miss.load(std::memory_order_relaxed),
      g_diag_ns_warm_hit.load(std::memory_order_relaxed),
      g_diag_ns_miss.load(std::memory_order_relaxed),
      g_diag_ns_fallback.load(std::memory_order_relaxed));
}

void diag_reset_dispatch_paths() {
  g_diag_n_total.store(0, std::memory_order_relaxed);
  g_diag_n_fallback.store(0, std::memory_order_relaxed);
  g_diag_n_warm_hit.store(0, std::memory_order_relaxed);
  g_diag_n_miss.store(0, std::memory_order_relaxed);
  g_diag_ns_warm_hit.store(0, std::memory_order_relaxed);
  g_diag_ns_miss.store(0, std::memory_order_relaxed);
  g_diag_ns_fallback.store(0, std::memory_order_relaxed);
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_warm_segments() {
  return std::make_tuple(
      g_diag_warm_n_hits.load(std::memory_order_relaxed),
      g_diag_warm_ns_lookup.load(std::memory_order_relaxed),
      g_diag_warm_ns_run.load(std::memory_order_relaxed),
      g_diag_warm_ns_finalize.load(std::memory_order_relaxed));
}

void diag_reset_warm_segments() {
  g_diag_warm_n_hits.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_lookup.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_run.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_finalize.store(0, std::memory_order_relaxed);
}

namespace {

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

using warmcache::CacheEntry;
using warmcache::CacheEntryPtr;
using warmcache::CacheKey;
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
  bool elementwise = false;
  bool populated = false;
};

const rt::Flag<bool> kEagerBuckets(
    "TORCH_RBLN_EAGER_BUCKETS",
    "Whether an eager elementwise op whose result has two or more rows along its leading axis runs a "
    "function compiled for the bucket of that extent (1, 2, 3, 4, 6, 8, 12, 16, ...) over its rows, "
    "rather than one compiled for the extent itself.",
    false,
    rt::parseBool,
    rt::FlagMutability::kMutable);

struct ShimEntry {
  pybind11::object py_fn;
  std::vector<size_t> skip_dtype_args;
  SchemaCache schema_cache; // lazily filled
  const char* op_name_intern = nullptr; // stable pointer for WarmCache keys
  // Cached pointer into fallback_by_op() for this op; bumped lock-free on the
  // already-slow fallback branch. Raw ptr keeps ShimEntry an aggregate / movable
  // (the registry stores it by value via move-assign).
  std::atomic<uint64_t>* fallback_ctr = nullptr;
  // Same idea for the warm-cache miss (recompile) path -> recompile_by_op().
  std::atomic<uint64_t>* recompile_ctr = nullptr;
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

// Per-op CPU-fallback counts, keyed by the interned op-name pointer (stable +
// deduplicated by intern_op_name). Heap-allocated atomics so each ShimEntry can
// cache a raw pointer (fallback_ctr) and bump it lock-free on the fallback
// branch. Populated/looked-up under registry_mutex at register time (import,
// before any dispatch); survives op re-registration. Leaky singleton to match
// registry() teardown semantics.
std::unordered_map<const char*, std::unique_ptr<std::atomic<uint64_t>>>& fallback_by_op() {
  static auto* m = new std::unordered_map<const char*, std::unique_ptr<std::atomic<uint64_t>>>();
  return *m;
}

// Per-op warm-cache MISS (recompile) counts. Same scheme as fallback_by_op():
// the bump lives on the already-slow miss branch (Python compile), so it does
// not touch the warm-cache hit fast path.
std::unordered_map<const char*, std::unique_ptr<std::atomic<uint64_t>>>& recompile_by_op() {
  static auto* m = new std::unordered_map<const char*, std::unique_ptr<std::atomic<uint64_t>>>();
  return *m;
}

// (A) WHERE: opt-in Python call-site capture. OFF by default -> default explain()
// is byte-identical overhead. When on, capture the user's call-site for an op
// ONCE (deduped per op), and ONLY on the already-slow fallback / miss branch.
std::atomic<bool> g_trace_enabled{false};
std::mutex& trace_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, std::string>& trace_by_op() {
  static auto* m = new std::unordered_map<std::string, std::string>();
  return *m;
}

void capture_site(const std::string& op_name) {
  {
    std::lock_guard<std::mutex> lk(trace_mutex());
    if (trace_by_op().find(op_name) != trace_by_op().end()) {
      return; // already captured -> no GIL / no Python
    }
  }
  if (!Py_IsInitialized()) {
    return;
  }
  // The dispatcher may have released the GIL before this boxed branch; acquire
  // it before ANY Python access (no-op if this thread already holds it).
  pybind11::gil_scoped_acquire gil;
  std::string site;
  try {
    pybind11::list stack = pybind11::module_::import("traceback").attr("extract_stack")();
    const auto n = static_cast<pybind11::ssize_t>(pybind11::len(stack));
    int shown = 0;
    for (pybind11::ssize_t i = n - 1; i >= 0 && shown < 2; --i) {
      pybind11::object fr = stack[static_cast<size_t>(i)];
      const std::string fname = pybind11::str(fr.attr("filename"));
      const long lineno = fr.attr("lineno").cast<long>();
      const std::string func = pybind11::str(fr.attr("name"));
      const auto slash = fname.rfind('/');
      const std::string base = (slash == std::string::npos) ? fname : fname.substr(slash + 1);
      if (!site.empty()) {
        site += " <- ";
      }
      site += base;
      site += ":";
      site += std::to_string(lineno);
      site += "(";
      site += func;
      site += ")";
      ++shown;
    }
  } catch (const pybind11::error_already_set&) {
    PyErr_Clear();
    return;
  }
  std::lock_guard<std::mutex> lk(trace_mutex());
  trace_by_op().emplace(op_name, site);
}

ShimEntry* find_shim_entry(const std::string& op_name) {
  auto& r = registry();
  auto it = r.find(op_name);
  return it != r.end() ? &it->second : nullptr;
}

bool elementwise(const c10::OperatorHandle& op) {
  return op.hasTag(at::Tag::pointwise) && !op.hasTag(at::Tag::nondeterministic_seeded);
}

void populate_schema_cache(SchemaCache& cache, const c10::OperatorHandle& op) {
  const auto& schema = op.schema();
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
  cache.elementwise = elementwise(op);
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

// ---------------------------------------------------------------------------
// Deploy / nan_inf-disable gates
// ---------------------------------------------------------------------------
//
// Contract: read live from the environment on every call (NOT process-cached), so both flags
// are runtime-dynamic. The per-call getenv is negligible against op-dispatch cost. Reading
// live also keeps per-test env toggles from latching a value into a long-lived worker process
// (see docs/CONFIGURATION.md).
bool is_deploy_mode() {
  const char* env = std::getenv("TORCH_RBLN_DEPLOY");
  return env != nullptr && std::strcmp(env, "ON") == 0;
}

bool is_nan_inf_check_disabled() {
  const char* env = std::getenv("TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK");
  if (env == nullptr)
    return false;
  std::string s = env;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(',', start);
    if (end == std::string::npos)
      end = s.size();
    std::string token = s.substr(start, end - start);
    const auto l = token.find_first_not_of(" \t");
    const auto r = token.find_last_not_of(" \t");
    if (l != std::string::npos) {
      token = token.substr(l, r - l + 1);
    } else {
      token.clear();
    }
    if (token == "all" || token == "nan_inf")
      return true;
    start = end + 1;
  }
  return false;
}

// NaN/Inf bit-pattern check.
//
// Every float format encodes NaN/Inf as "exponent field all-ones"; only the
// field differs (fp16: 5 bits at offset 10, mask ``0x7C00``; bf16: 8 bits at
// offset 7, mask ``0x7F80``; fp32: 8 bits at offset 23, mask ``0x7F800000``).
// NaN distinguishes itself by a non-zero mantissa, Inf has mantissa==0 — we
// don't care about the distinction for fallback routing, either value means
// "the rebel.v2 runtime cannot handle this".
template <typename Word, Word kExponent>
bool has_nan_or_inf(const void* data, size_t n) noexcept {
  const auto* words = static_cast<const Word*>(data);
  for (size_t i = 0; i < n; ++i) {
    if ((words[i] & kExponent) == kExponent) {
      return true;
    }
  }
  return false;
}

using ScannerFn = bool (*)(const void*, size_t);
inline ScannerFn scanner_for(c10::ScalarType scalar_type) {
  switch (scalar_type) {
    case c10::kHalf:
      return has_nan_or_inf<uint16_t, 0x7C00>;
    case c10::kBFloat16:
      return has_nan_or_inf<uint16_t, 0x7F80>;
    case c10::kFloat:
      return has_nan_or_inf<uint32_t, 0x7F800000>;
    default:
      TORCH_INTERNAL_ASSERT(false, "missing scanner for ScalarType");
  }
}

// Scan a single tensor for NaN/Inf. An rbln tensor is copied to the host
// first (the price of catching NaN/Inf in just-computed device data — matches
// the Python ``to_cpu(args)`` cost).
//
// Returns false for: undefined / empty / dtype the device does not
// dispatch / non-contiguous tensors. Those dtypes are short-circuited
// earlier in ``quick_fallback_check``; ``skip_dtype_args`` slots are
// typically bool/int (eq/ne ``cond`` etc.) which cannot carry NaN/Inf.
// Non-contiguous tensors are skipped here because the warm-cache key
// requires contig + offset=0 anyway — the non-contig case will miss and
// fall through to the Python wrapper which performs the full
// ``has_invalid_tensor(to_cpu(args))`` scan.
bool tensor_has_nan_or_inf(const at::Tensor& t) {
  if (!t.defined())
    return false;
  const auto numel = t.numel();
  if (numel == 0)
    return false;
  if (!c10::rbln::dispatches(t.scalar_type())) {
    return false;
  }
  const auto scanner = scanner_for(t.scalar_type());
  if (!t.is_contiguous())
    return false;

  const size_t n = static_cast<size_t>(numel);
  const auto dev_type = t.device().type();
  if (dev_type != c10::DeviceType::PrivateUse1) {
    // CPU tensor (e.g. wrapped 0-dim scalar that didn't get unwrapped):
    // scan in place.
    const void* data = t.data_ptr();
    if (data == nullptr)
      return false;
    return scanner(data, n);
  }

  if (t.data_ptr() == nullptr)
    return false;
  const at::Tensor cpu_copy = t.cpu();
  return scanner(cpu_copy.const_data_ptr(), n);
}

// A Python number that PyTorch wrapped into a 0-dim CPU tensor (``tensor + 1``).
// The pybind boundary unwraps it back into a Python scalar and the Python
// wrapper compiles it as a graph constant, so the runtime never sees it as an
// input. The warm cache keys it by value -- ``a + 1`` and ``a + 2`` are
// different programs -- and leaves it out of the device inputs on the hit path.
inline bool is_wrapped_scalar(const at::Tensor& t) {
  return t.dim() == 0 && t.unsafeGetTensorImpl()->is_wrapped_number();
}

// nullopt for a value ScalarValue cannot hold (complex): the key must miss
// rather than collide.
inline std::optional<ScalarValue> wrapped_scalar_value(const at::Tensor& t) {
  const c10::Scalar s = t.item();
  if (s.isBoolean())
    return ScalarValue::fromBool(s.toBool());
  if (s.isIntegral(/*includeBool=*/false))
    return ScalarValue::fromInt(s.toLong());
  if (s.isFloatingPoint())
    return ScalarValue::fromFloat(s.toDouble());
  return std::nullopt;
}

// Cheap C++-side pre-check mirroring the cheap branches of
// torch_rbln._internal.ops_utils.is_cpu_fallback_cases():
//   2. a dtype the device does not dispatch on any input tensor
//   3. all input tensors are scalar (ndim == 0)
//   4. NaN/Inf in any input tensor  (non-deploy mode only; mirrors the
//      ``not is_rbln_deploy() and has_invalid_tensor(to_cpu(args))`` branch
//      that AS-IS ran on every Python wrapper entry — the warm-cache hot
//      path otherwise bypasses Python entirely, losing the safety net).
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
// Returns 0 = no fallback, else the reason code (1=dtype-not-fp16, 2=nan/inf
// input, 3=all-scalar). The reason is already decided here; returning it instead
// of a bool lets the caller attribute WHY at zero extra cost.
int quick_fallback_check(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    const std::vector<size_t>& skip_dtype_args) {
  auto args = torch::jit::last(stack, cache.num_args);
  const bool nan_inf_scan_enabled = !is_deploy_mode() && !is_nan_inf_check_disabled();
  bool has_input_tensor = false;
  bool all_input_scalar = true;
  bool nan_inf_found = false;
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

    // NaN/Inf scan: applies BEFORE the dtype / skip_dtype / wrapped-0-dim
    // gates that skip args from the shortcut counter. We want to catch
    // NaN/Inf in any defined non-write-alias input — including wrapped
    // 0-dim values such as ``tensor + math.nan``. tensor_has_nan_or_inf
    // internally filters out dtypes the device does not dispatch (the ones
    // it dispatches are the floats that can encode NaN/Inf on the shim path).
    if (nan_inf_scan_enabled && !nan_inf_found && tensor_has_nan_or_inf(t)) {
      nan_inf_found = true;
    }

    if (is_skipped_arg(skip_dtype_args, i)) {
      continue;
    }
    // Wrapped 0-dim numbers behave like Python scalars and are unwrapped by
    // the pybind boundary. Skip them from the dtype check so the shortcut
    // doesn't fire for `tensor + 1.0` etc.
    if (is_wrapped_scalar(t)) {
      continue;
    }
    has_input_tensor = true;
    if (!c10::rbln::dispatches(t.scalar_type())) {
      return 1; // dtype-not-fp16 (dtype outside the dispatch policy)
    }
    if (t.dim() != 0) {
      all_input_scalar = false;
    }
  }
  if (nan_inf_found) {
    return 2; // nan/inf in input (non-deploy debug scan)
  }
  return (has_input_tensor && all_input_scalar) ? 3 : 0; // 3 = all-scalar inputs
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

// The tensor arguments of a call that are not written to, in stack order: the
// ones a warm-cache key profiles and an entry's inputs are positions among.
using CallTensors = c10::SmallVector<const at::Tensor*, 4>;

// Build a WarmCache::CacheKey from the current stack's last num_args IValues,
// and collect the call's tensors. Tensor args (non-write-alias, defined) become
// TensorProfiles in their positional order. Scalar args become ScalarValues.
// Tensor-list, list and string args mean the call cannot be warm-cached
// (return false; caller falls through to pybind).
//
// TensorProfile shapes are always the RAW input shapes: two calls that
// broadcast to the same result shape (``(4,8,16) + ()`` and
// ``(4,8,16) + (4,8,1)``) are different programs.
bool build_cache_key(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    const char* op_name_intern,
    CacheKey& out_key,
    CallTensors& tensors) {
  out_key.schema_name_intern = op_name_intern;
  out_key.inputs.clear();
  out_key.scalars.clear();
  out_key.float32_precision = ::rebel::v2::runtime::float32Precision();
  tensors.clear();

  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = arguments[i];
    if (iv.isTensor()) {
      const at::Tensor& t = iv.toTensor();
      if (!t.defined())
        continue;
      if (cache.is_write_alias[i])
        continue; // out tensor, not part of key
      if (is_wrapped_scalar(t)) {
        const auto sv = wrapped_scalar_value(t);
        if (!sv)
          return false;
        out_key.scalars.push_back(*sv);
        continue;
      }
      TensorProfile tp;
      tp.dtype = t.scalar_type();
      tp.shape.assign(t.sizes().begin(), t.sizes().end());
      tp.strides.assign(t.strides().begin(), t.strides().end());
      tp.storage_offset = t.storage_offset();
      tp.device_index = static_cast<int8_t>(t.device().index());
      out_key.inputs.emplace_back(std::move(tp));
      tensors.push_back(&t);
    } else if (iv.isNone()) {
      // Treat `None` slot as a Missing scalar — keeps positional structure
      // without requiring us to distinguish "optional scalar absent" from
      // "optional tensor absent"; both just miss if later calls differ.
      out_key.scalars.push_back(ScalarValue::missing());
    } else if (iv.isTensorList() || iv.isList() || iv.isString()) {
      // Lists are not handled by the warm-cache path, and strings (e.g.
      // ``div.out_mode``'s ``rounding_mode``) are not representable in
      // ``ScalarValue``: floor's function must not be hit by a trunc call.
      return false;
    } else {
      out_key.scalars.push_back(ival_to_scalar(iv));
    }
  }
  return true;
}

// The least bucket extent that holds rows: 1, 2, 3, 4, 6, 8, 12, 16, 24, ... A run computes fewer
// than 1.5 times the rows it returns, and a growing extent compiles twice per doubling.
int64_t bucket_extent(int64_t rows) {
  const auto power = static_cast<int64_t>(std::bit_floor(static_cast<uint64_t>(rows)));
  if (rows == power) {
    return rows;
  }
  return rows <= power + power / 2 ? power + power / 2 : 2 * power;
}

// Rows bucket only when they hold a multiple of this many elements: the compiler computes an elementwise op on
// the host or on the device by whether its sizes are multiples of it, and a bucket must not move a call across.
constexpr int64_t kRowElements = 64;

// A call's leading extent, the extent of its bucket, and the rank of its result.
struct Bucket {
  int64_t rows = 0;
  int64_t extent = 0;
  int64_t rank = 0;
};

// The bucket of a call over tensors, or none when it runs the function of its own extent. A call is
// bucketed when TORCH_RBLN_EAGER_BUCKETS is on, every tensor is contiguous (a view recipe cannot take
// a bucket), every tensor along the leading axis is on the device with rows of a multiple of
// kRowElements elements, and the result has two or more rows.
std::optional<Bucket> bucket_of(const CallTensors& tensors) {
  if (tensors.empty() || !kEagerBuckets.value()) {
    return std::nullopt;
  }
  int64_t rank = 0;
  for (const auto* tensor : tensors) {
    if (!tensor->is_contiguous()) {
      return std::nullopt;
    }
    rank = std::max(rank, tensor->dim());
  }
  int64_t rows = 0;
  for (const auto* tensor : tensors) {
    if (tensor->dim() == rank && rank > 0) {
      rows = std::max(rows, tensor->size(0));
    }
  }
  if (rows < 2) {
    return std::nullopt;
  }
  for (const auto* tensor : tensors) {
    if (tensor->dim() == rank && tensor->size(0) == rows &&
        (!tensor->device().is_privateuseone() || tensor->numel() / rows % kRowElements != 0)) {
      return std::nullopt;
    }
  }
  return Bucket{.rows = rows, .extent = bucket_extent(rows), .rank = rank};
}

// Whether the input `profile` describes stands at the extent of `bucket` in a key bucket_key made.
bool at_bucket(const TensorProfile& profile, const Bucket& bucket) {
  return static_cast<int64_t>(profile.shape.size()) == bucket.rank && profile.shape.front() == bucket.extent;
}

// `key`, which bucket_key made for `bucket`, as the key of the call's own extent.
void unbucket(CacheKey& key, const Bucket& bucket) {
  for (auto& input : key.inputs) {
    if (at_bucket(input, bucket)) {
      input.shape.front() = bucket.rows;
    }
  }
  key.bucketed = false;
}

// The bucket of the call, as bucket_of gives it, with key set to the bucket's function: each tensor
// of the result's rank and leading extent stands at the bucket's extent. An out that holds no
// memory, or is the call's own (own_out), takes the result's memory; one that is not contiguous or
// overlaps an input, as an in-place op's does, takes a copy. Any other out leaves the call unbucketed.
std::optional<Bucket> bucket_key(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    const CallTensors& tensors,
    bool own_out,
    CacheKey& key) {
  if (!cache.elementwise || cache.return_types.size() != 1) {
    return std::nullopt;
  }
  const auto bucket = bucket_of(tensors);
  if (!bucket) {
    return std::nullopt;
  }
  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    if (!cache.is_write_alias[i] || !arguments[i].isTensor() || !arguments[i].toTensor().defined()) {
      continue;
    }
    const auto& out = arguments[i].toTensor();
    const bool copied = !out.is_contiguous() || std::ranges::any_of(tensors, [&](const at::Tensor* t) {
      return at::get_overlap_status(out, *t) != at::MemOverlapStatus::No;
    });
    if (!own_out && out.storage().nbytes() != 0 && !copied) {
      return std::nullopt;
    }
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->dim() == bucket->rank && tensors[i]->size(0) == bucket->rows) {
      key.inputs[i].shape.front() = bucket->extent;
    }
  }
  key.bucketed = true;
  return bucket;
}

// Thread-local context that ties a just-computed CacheKey (built before the
// pybind miss-path call) to the later pybind-exposed install hook called from
// the Python wrapper after it compiles and runs the op, with the call's
// tensors, which the wrapper's inputs are matched against, its op's name and
// its bucket.
struct PendingInstall {
  bool valid = false;
  CacheKey key;
  c10::SmallVector<const c10::TensorImpl*, 4> tensors;
  const std::string* op = nullptr;
  std::optional<Bucket> bucket;
};

thread_local PendingInstall t_pending;

// Gives a shim call's Python wrapper a t_pending of its own: the scope keeps the
// outer call's context, if any, and gives it back when the call returns, so that
// a shim call the wrapper makes in turn does not take it.
class PendingScope {
 public:
  PendingScope() {
    if (t_pending.valid) {
      outer_ = std::move(t_pending);
      t_pending.valid = false;
    }
  }
  ~PendingScope() {
    if (outer_) {
      t_pending = std::move(*outer_);
    } else {
      t_pending.valid = false;
    }
  }
  PendingScope(const PendingScope&) = delete;
  PendingScope& operator=(const PendingScope&) = delete;

 private:
  std::optional<PendingInstall> outer_;
};

// Take ownership of the pending context (single reader); installer clears it.
PendingInstall take_pending() {
  PendingInstall p = std::move(t_pending);
  t_pending.valid = false;
  return p;
}

// The position among the pending call's tensors of each of `tensors`, when the
// pending call is one of `op` and every one of `tensors` is one of its tensors.
// A tensor passed twice takes its positions in order, as ``add(a, a)`` and
// ``add(a, b)`` share a key.
std::optional<c10::SmallVector<uint32_t, 4>> pending_positions(
    const PendingInstall& p,
    const std::string& op,
    c10::ArrayRef<at::Tensor> tensors) {
  if (!p.valid || p.op == nullptr || *p.op != op) {
    return std::nullopt;
  }
  c10::SmallVector<uint32_t, 4> positions;
  c10::SmallVector<bool, 4> taken(p.tensors.size(), false);
  for (const auto& tensor : tensors) {
    const auto* impl = tensor.unsafeGetTensorImpl();
    std::optional<size_t> found;
    for (size_t j = 0; j < p.tensors.size(); ++j) {
      if (p.tensors[j] == impl && (!found || (taken[*found] && !taken[j]))) {
        found = j;
      }
    }
    if (!found) {
      return std::nullopt;
    }
    taken[*found] = true;
    positions.push_back(static_cast<uint32_t>(*found));
  }
  return positions;
}

// The out= overload whose next shim call on this thread has as its out the
// result torch's structured kernel just allocated, which nothing else holds
// (StructuredFunctional).
thread_local const c10::FunctionSchema* t_own_out = nullptr;

// Hot path: look up the warm-cache entry for `key` and, on hit, run its
// OpFunction over the call's tensors from C++ — no pybind, no Python wrapper.
// Returns true iff the hit path was taken and the stack has been left with the
// return value. A call whose tensors the function cannot take as they are (a
// view, an ``out`` of another shape) falls through to the Python wrapper; the
// entry stays for later calls. A bucket whose entry has no function runs its
// calls by the function of their own extent: `key` and `bucket` become the
// call's own, for the miss path as well.
bool try_warmcache_hit(
    torch::jit::Stack* stack,
    const SchemaCache& cache,
    CacheKey& key,
    const CallTensors& tensors,
    std::optional<Bucket>& bucket,
    bool own_out) {
  auto& wc = WarmCache::instance();
  if (!wc.is_enabled() || cache.return_types.size() != 1)
    return false;

  const uint64_t _seg_t0 = now_ns();
  CacheEntryPtr entry = wc.find(key);
  if (entry && !entry->function) {
    unbucket(key, *bucket);
    bucket.reset();
    entry = wc.find(key);
  }
  const uint64_t _seg_t_lookup = now_ns();
  if (!entry)
    return false;

  c10::SmallVector<at::Tensor, 4> inputs;
  inputs.reserve(entry->inputs.size());
  for (const auto position : entry->inputs) {
    inputs.push_back(*tensors[position]);
  }
  // The tensor the schema writes to is the one the function's result goes to.
  std::optional<at::Tensor> out;
  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    if (cache.is_write_alias[i] && arguments[i].isTensor() && arguments[i].toTensor().defined()) {
      out = arguments[i].toTensor();
      break;
    }
  }
  // An out that holds no memory yet, as a composite kernel hands over, takes the memory
  // of a result made for it, as does a bucketed call's own out.
  const auto& output = entry->function->outputs().front();
  auto shape = output.shape;
  if (bucket) {
    shape.front() = bucket->rows;
  }
  const bool fresh = out && (out->storage().nbytes() == 0 || (own_out && bucket)) && out->scalar_type() == output.dtype;
  const bool fits = out && out->scalar_type() == output.dtype && out->sizes() == c10::IntArrayRef(shape);
  // A bucketed run makes a result of the bucket's rows, which the call's out cannot hold: an out of
  // the call's shape takes a copy, and one of another shape goes to the wrapper.
  if (out && !fresh && bucket && !fits) {
    return false;
  }
  const bool bind_out = out && !fresh && !bucket;
  auto results = entry->function->run(
      inputs, bind_out ? c10::ArrayRef<std::optional<at::Tensor>>(*out) : c10::ArrayRef<std::optional<at::Tensor>>());
  bool copied = out && !fresh && bucket;
  // An unbucketed out the function cannot write where it is, as one an in-place op shares with an
  // input, takes a copy of a result made for it.
  if (bind_out && !results && fits) {
    results = entry->function->run(inputs);
    copied = true;
  }
  const uint64_t _seg_t_run = now_ns();
  if (!results) {
    return false;
  }
  if (fresh) {
    out->set_(results->front());
  } else if (copied) {
    out->copy_(results->front());
  }

  torch::jit::drop(stack, cache.num_args);
  torch::jit::push(stack, fresh || copied ? *out : std::move(results->front()));
  const uint64_t _seg_t_finalize = now_ns();

  g_diag_warm_n_hits.fetch_add(1, std::memory_order_relaxed);
  g_diag_warm_ns_lookup.fetch_add(_seg_t_lookup - _seg_t0, std::memory_order_relaxed);
  g_diag_warm_ns_run.fetch_add(_seg_t_run - _seg_t_lookup, std::memory_order_relaxed);
  g_diag_warm_ns_finalize.fetch_add(_seg_t_finalize - _seg_t_run, std::memory_order_relaxed);
  return true;
}

// The boxed kernel that Library::impl points at for every shimmed op.
void generic_shim_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  g_diag_n_total.fetch_add(1, std::memory_order_relaxed);
  const bool own_out = t_own_out == &op.schema();
  if (own_out) {
    t_own_out = nullptr;
  }
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
      populate_schema_cache(entry->schema_cache, op);
    }
  }

  const SchemaCache& cache = entry->schema_cache;
  const auto& skip_dtype_args = entry->skip_dtype_args;
  const char* op_name_intern = entry->op_name_intern;

  // The C++ precheck identifies cheap "must fallback" cases (input dtype
  // outside the dispatch catalog or all-0-dim) and short-circuits straight
  // into cpu_fallback_rbln,
  // bypassing the pybind hop into the Python wrapper. The wrapped-0-dim
  // case (e.g. `tensor + 1.0` where PyTorch wraps `1.0` as a 0-dim CPU
  // tensor with `is_wrapped_number`) is intentionally excluded from the
  // precheck — see quick_fallback_check — because the pybind boundary
  // unwraps such tensors back to Python scalars, so the Python wrapper sees
  // a single tensor arg and routes through the RBLN compile path; if we
  // shortcut those calls into cpu_fallback_rbln we'd skip that compile
  // path and get bit-different fp16 rounding than the surrounding
  // RBLN-compiled ops produce.
  const int fb_reason = quick_fallback_check(stack, cache, skip_dtype_args);
  if (fb_reason != 0) {
    g_diag_n_fallback.fetch_add(1, std::memory_order_relaxed);
    g_fallback_reason[fb_reason].fetch_add(1, std::memory_order_relaxed); // WHY (same slow branch)
    if (entry->fallback_ctr != nullptr) {
      entry->fallback_ctr->fetch_add(1, std::memory_order_relaxed); // per-op attribution (same slow branch)
    }
    if (g_trace_enabled.load(std::memory_order_relaxed)) {
      capture_site(op_name); // (A) WHERE: opt-in, deduped, GIL-safe; off by default
    }
    // Same log the ``fallback_rbln`` handler emits for an unsupported op. The
    // shortcut below decides a fallback the Python wrapper would otherwise have
    // decided and logged, so without this the ops it covers -- add, mul, matmul,
    // the reductions -- are the only ones whose CPU fallback leaves no trace at
    // any log level.
    c10::rbln::log_cpu_fallback(op.schema().name());
    const uint64_t _fb_t0 = now_ns();
    ::at::native::rbln::cpu_fallback_rbln(op, stack);
    // COST of the fallback (wall ns), so the report can tell "many cheap fallbacks
    // (path overhead)" from "few expensive ones (hidden transfer)". Same slow branch.
    g_diag_ns_fallback.fetch_add(now_ns() - _fb_t0, std::memory_order_relaxed);
    return;
  }

  // Warm-cache hot path: if we've previously compiled this op for an identical
  // input profile, run its function from C++ directly.
  CacheKey key;
  CallTensors tensors;
  const bool key_ok = build_cache_key(stack, cache, op_name_intern, key, tensors);
  std::optional<Bucket> bucket;
  if (key_ok) {
    bucket = bucket_key(stack, cache, tensors, own_out, key);
    const uint64_t _diag_warm_t0 = now_ns();
    const bool hit = try_warmcache_hit(stack, cache, key, tensors, bucket, own_out);
    if (hit) {
      g_diag_n_warm_hit.fetch_add(1, std::memory_order_relaxed);
      g_diag_ns_warm_hit.fetch_add(now_ns() - _diag_warm_t0, std::memory_order_relaxed);
      return;
    }
  }

  // MISS path: set up thread-local pending install so the Python wrapper can
  // call `_warmcache_install_pending(op, function, inputs)` once it finishes
  // compile + first run. The pending context is discarded at the end of this
  // function (even on failure / exception) to avoid leaking into subsequent
  // unrelated ops on the same thread.
  g_diag_n_miss.fetch_add(1, std::memory_order_relaxed);
  if (entry->recompile_ctr != nullptr) {
    entry->recompile_ctr->fetch_add(1, std::memory_order_relaxed); // per-op attribution (same slow miss branch)
  }
  if (g_trace_enabled.load(std::memory_order_relaxed)) {
    capture_site(op_name); // (A) WHERE: opt-in, deduped, GIL-safe; off by default
  }
  const uint64_t _diag_miss_t0 = now_ns();
  struct MissScopeTimer {
    uint64_t t0;
    ~MissScopeTimer() {
      g_diag_ns_miss.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    }
  } _diag_miss_guard{_diag_miss_t0};
  const PendingScope pending;
  if (key_ok) {
    t_pending.valid = true;
    t_pending.key = std::move(key);
    t_pending.tensors.clear();
    for (const auto* t : tensors) {
      t_pending.tensors.push_back(t->unsafeGetTensorImpl());
    }
    t_pending.op = &op.schema().name();
    t_pending.bucket = bucket;
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

  // The wrapper gets a stand-in with no memory for a bucketed call's own out. The
  // stand-in takes the result's memory, and the out then takes the stand-in's.
  std::optional<at::Tensor> own;
  at::Tensor stand_in;
  auto arguments = torch::jit::last(stack, cache.num_args);
  for (size_t i = 0; i < cache.num_args; ++i) {
    pybind11::object val;
    if (own_out && bucket && cache.is_write_alias[i] && arguments[i].isTensor() && arguments[i].toTensor().defined()) {
      own = arguments[i].toTensor();
      stand_in = at::empty({0}, own->options());
      val = torch::jit::toPyObject(c10::IValue(stand_in));
    } else {
      val = torch::jit::toPyObject(arguments[i]);
    }
    if (cache.is_kwarg_only[i]) {
      kwargs[cache.arg_names[i].c_str()] = val;
      if (static_cast<int>(i) == cache.out_positional_idx) {
        out_obj = val;
      }
    } else {
      pos_tup[pos_idx++] = val;
    }
  }

  pybind11::object result = py_fn_copy(*pos_tup, **kwargs);
  if (own) {
    own->set_(stand_in);
    out_obj = torch::jit::toPyObject(c10::IValue(*own));
  }

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

// Runs torch's structured kernel of a shimmed elementwise op's functional
// overload, which checks the call, allocates the result and calls the out=
// overload. The result is marked as the call's own, so that a bucketed out= call
// gives it the bucket's storage instead of copying into it (see bucket_key).
class StructuredFunctional final : public c10::OperatorKernel {
 public:
  StructuredFunctional(c10::SafeKernelFunction kernel, const c10::FunctionSchema& out)
      : kernel_(std::move(kernel)), out_(&out) {}

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet keys, torch::jit::Stack* stack) const {
    struct Own {
      explicit Own(const c10::FunctionSchema* out) : outer(std::exchange(t_own_out, out)) {}
      ~Own() {
        t_own_out = outer;
      }
      const c10::FunctionSchema* outer;
    } own(out_);
    kernel_.callBoxed(op, keys, stack);
  }

 private:
  c10::SafeKernelFunction kernel_;
  const c10::FunctionSchema* out_;
};

// The functional overload of `out_op`, an out= overload of an elementwise op,
// that torch runs through its structured kernel: the overload of the same name
// with a CompositeExplicitAutogradNonFunctional kernel, which torchgen gives a
// structured op, and the same arguments but the out.
std::optional<c10::OperatorHandle> structured_functional_of(const c10::OperatorHandle& out_op) {
  const auto& schema = out_op.schema();
  const auto& args = schema.arguments();
  if (!elementwise(out_op) || args.empty() || args.back().alias_info() == nullptr ||
      !args.back().alias_info()->isWrite() || schema.returns().size() != 1) {
    return std::nullopt;
  }
  auto& dispatcher = c10::Dispatcher::singleton();
  for (const auto& name :
       dispatcher.getAllOpNamesForDispatchKey(c10::DispatchKey::CompositeExplicitAutogradNonFunctional)) {
    if (name.name != schema.name() || name.overload_name == schema.overload_name()) {
      continue;
    }
    auto handle = dispatcher.findOp(name);
    if (!handle) {
      continue;
    }
    const auto& f = handle->schema().arguments();
    const bool same = f.size() + 1 == args.size() && handle->schema().returns().size() == 1 &&
        std::equal(f.begin(), f.end(), args.begin(), [](const c10::Argument& a, const c10::Argument& b) {
                        return a.name() == b.name() && *a.type() == *b.type() && a.kwarg_only() == b.kwarg_only() &&
                            a.alias_info() == nullptr;
                      });
    if (same) {
      return handle;
    }
  }
  return std::nullopt;
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
  registry()[op_name] = ShimEntry{
      .py_fn = std::move(py_fn), .skip_dtype_args = skip_dtype_args, .schema_cache = {}, .op_name_intern = interned};
  // Wire up the per-op fallback counter (heap atomic, keyed by interned name so
  // it survives re-registration). The move-assign above reset fallback_ctr to
  // null, so re-point it here.
  {
    auto& slot = fallback_by_op()[interned];
    if (!slot) {
      slot = std::make_unique<std::atomic<uint64_t>>(0);
    }
    registry()[op_name].fallback_ctr = slot.get();
    auto& rslot = recompile_by_op()[interned];
    if (!rslot) {
      rslot = std::make_unique<std::atomic<uint64_t>>(0);
    }
    registry()[op_name].recompile_ctr = rslot.get();
  }
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
  const auto dot = overload.find('.');
  const auto out_op = c10::Dispatcher::singleton().findOp(
      {"aten::" + overload.substr(0, dot), dot == std::string::npos ? "" : overload.substr(dot + 1)});
  if (const auto functional = out_op ? structured_functional_of(*out_op) : std::nullopt) {
    const auto name = c10::toString(functional->schema().operator_name());
    TORCH_INTERNAL_ASSERT(
        !functional->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1),
        name,
        " has a PrivateUse1 kernel of its own, which StructuredFunctional would replace");
    lib->impl(
        strip_namespace(name).c_str(),
        torch::CppFunction::makeFromBoxedFunctor(std::make_unique<StructuredFunctional>(
            functional->getComputedKernelForDispatchKey(c10::DispatchKey::PrivateUse1), out_op->schema())));
  }
  installed_libs().push_back(std::move(lib));
}

std::vector<std::pair<std::string, uint64_t>> diag_dump_fallback_by_op() {
  std::vector<std::pair<std::string, uint64_t>> out;
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (const auto& kv : registry()) {
    const ShimEntry& e = kv.second;
    if (e.fallback_ctr != nullptr) {
      const uint64_t c = e.fallback_ctr->load(std::memory_order_relaxed);
      if (c != 0) {
        out.emplace_back(kv.first, c);
      }
    }
  }
  return out;
}

void diag_reset_fallback_by_op() {
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (auto& kv : fallback_by_op()) {
    if (kv.second) {
      kv.second->store(0, std::memory_order_relaxed);
    }
  }
}

std::vector<std::pair<std::string, uint64_t>> diag_dump_recompile_by_op() {
  std::vector<std::pair<std::string, uint64_t>> out;
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (const auto& kv : registry()) {
    const ShimEntry& e = kv.second;
    if (e.recompile_ctr != nullptr) {
      const uint64_t c = e.recompile_ctr->load(std::memory_order_relaxed);
      if (c != 0) {
        out.emplace_back(kv.first, c);
      }
    }
  }
  return out;
}

void diag_reset_recompile_by_op() {
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (auto& kv : recompile_by_op()) {
    if (kv.second) {
      kv.second->store(0, std::memory_order_relaxed);
    }
  }
}

std::vector<uint64_t> diag_dump_fallback_reasons() {
  // counts for reason codes 1..3: [dtype-not-fp16, nan/inf input, all-scalar].
  return {
      g_fallback_reason[1].load(std::memory_order_relaxed),
      g_fallback_reason[2].load(std::memory_order_relaxed),
      g_fallback_reason[3].load(std::memory_order_relaxed),
  };
}

void diag_reset_fallback_reasons() {
  for (auto& r : g_fallback_reason) {
    r.store(0, std::memory_order_relaxed);
  }
}

// (A) WHERE for bounces: c10::record_bounce calls this (when installed) with the
// BounceSite ordinal. Map it to the report label and reuse capture_site, so a
// bounced copy_ gets its Python call-site keyed by the site name in trace_by_op
// (the report shows "at ..." under the bounce row). noexcept: it is invoked from
// c10's noexcept record_bounce, so it must never let an exception escape. The
// names + order mirror BounceSite and profiler.py's _BOUNCE_SITES.
static void bounce_site_capture(uint8_t site) noexcept {
  if (!g_trace_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  static constexpr std::array<const char*, 7> kNames = {
      "copy_d2d_host_bounce",
      "copy_h2d_staging",
      "copy_h2d_noncontig_dst",
      "strided_v2v_cpu_fallback",
      "host_batch_to_per_entry",
      "op_arg_through_host",
      "typed_through_host"};
  static_assert(kNames.size() == c10::rbln::prof::kNumBounceSites, "bounce site names must match the BounceSite enum");
  if (site >= kNames.size()) {
    return;
  }
  // This hook runs inside c10's noexcept record_bounce, so it must never throw.
  // capture_site can (mutex / map alloc); swallow — a diagnostic hook failing is
  // not worth aborting the run.
  try {
    capture_site(kNames[site]);
  } catch (...) {
    return;
  }
}

// (A) WHERE: opt-in call-site capture. enable() flips the gate the slow branches
// read; dump returns (op_name -> "file:line(func) <- ...") for the ops that fired
// while enabled; reset clears between regions. Also (un)installs the bounce hook
// in c10 so bounced copies capture their call-site too (ON==OFF: null when off).
void diag_set_trace_enabled(bool on) {
  g_trace_enabled.store(on, std::memory_order_relaxed);
  c10::rbln::prof::set_bounce_capture_fn(on ? &bounce_site_capture : nullptr);
}

std::vector<std::pair<std::string, std::string>> diag_dump_trace_by_op() {
  std::vector<std::pair<std::string, std::string>> out;
  std::lock_guard<std::mutex> lk(trace_mutex());
  out.reserve(trace_by_op().size());
  for (const auto& kv : trace_by_op()) {
    out.emplace_back(kv.first, kv.second);
  }
  return out;
}

void diag_reset_trace_by_op() {
  std::lock_guard<std::mutex> lk(trace_mutex());
  trace_by_op().clear();
}

// ---------------------------------------------------------------------------
// Warm-cache install hook, called from Python after a successful miss-path
// compile and run of `function` over `inputs`.
// ---------------------------------------------------------------------------

std::optional<std::pair<int64_t, std::vector<size_t>>> pending_bucket(
    const std::string& op,
    const std::vector<at::Tensor>& tensors) {
  const auto& p = t_pending;
  const auto positions = pending_positions(p, op, tensors);
  if (!positions || !p.bucket) {
    return std::nullopt;
  }
  std::vector<size_t> rows;
  for (size_t i = 0; i < positions->size(); ++i) {
    if (at_bucket(p.key.inputs[(*positions)[i]], *p.bucket)) {
      rows.push_back(i);
    }
  }
  if (rows.empty()) {
    return std::nullopt;
  }
  return std::pair{p.bucket->extent, std::move(rows)};
}

bool install_warmcache_from_pending(
    const std::string& op,
    std::shared_ptr<OpFunction> function,
    const std::vector<at::Tensor>& inputs) {
  PendingInstall p = take_pending();
  if (function->outputs().size() != 1 || inputs.size() != function->num_inputs())
    return false;
  auto positions = pending_positions(p, op, inputs);
  if (!positions)
    return false;

  CacheEntry entry{.function = std::move(function), .inputs = std::move(*positions)};
  TORCH_INTERNAL_ASSERT(
      p.bucket || !entry.function->row_wise(), "a function computed by rows ran over a call that is not bucketed");
  auto& wc = WarmCache::instance();
  // A bucketed call the wrapper ran by the function of its own extent leaves its bucket
  // an entry with no function, so that later calls of the bucket run by theirs.
  if (p.bucket && !entry.function->row_wise()) {
    auto own = p.key;
    unbucket(own, *p.bucket);
    wc.install(std::move(p.key), CacheEntry{});
    wc.install(std::move(own), entry);
    return true;
  }
  wc.install(std::move(p.key), entry);
  return true;
}

} // namespace torch_rbln::shim
