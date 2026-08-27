#include <torch_rbln/csrc/rbln/DispatchShim.h>
#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/library.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
std::atomic<uint64_t> g_diag_n_align_fastpath{0}; // align fast-path hits → cpu_fallback

// Warm-cache hit path per-segment timers. Accumulated only on successful hits
// so per-segment averages = ns_X / n_hits give the steady-state breakdown.
std::atomic<uint64_t> g_diag_warm_n_hits{0};
std::atomic<uint64_t> g_diag_warm_ns_lookup{0};
std::atomic<uint64_t> g_diag_warm_ns_io_build{0};
std::atomic<uint64_t> g_diag_warm_ns_gil{0};
std::atomic<uint64_t> g_diag_warm_ns_prep_in{0};
std::atomic<uint64_t> g_diag_warm_ns_prep_out{0};
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

uint64_t diag_dump_align_fastpath_count() {
  return g_diag_n_align_fastpath.load(std::memory_order_relaxed);
}

void diag_reset_dispatch_paths() {
  g_diag_n_total.store(0, std::memory_order_relaxed);
  g_diag_n_fallback.store(0, std::memory_order_relaxed);
  g_diag_n_warm_hit.store(0, std::memory_order_relaxed);
  g_diag_n_miss.store(0, std::memory_order_relaxed);
  g_diag_ns_warm_hit.store(0, std::memory_order_relaxed);
  g_diag_ns_miss.store(0, std::memory_order_relaxed);
  g_diag_ns_fallback.store(0, std::memory_order_relaxed);
  g_diag_n_align_fastpath.store(0, std::memory_order_relaxed);
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_warm_segments() {
  return std::make_tuple(
      g_diag_warm_n_hits.load(std::memory_order_relaxed),
      g_diag_warm_ns_lookup.load(std::memory_order_relaxed),
      g_diag_warm_ns_io_build.load(std::memory_order_relaxed),
      g_diag_warm_ns_gil.load(std::memory_order_relaxed),
      g_diag_warm_ns_prep_in.load(std::memory_order_relaxed),
      g_diag_warm_ns_prep_out.load(std::memory_order_relaxed),
      g_diag_warm_ns_run.load(std::memory_order_relaxed),
      g_diag_warm_ns_finalize.load(std::memory_order_relaxed));
}

void diag_reset_warm_segments() {
  g_diag_warm_n_hits.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_lookup.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_io_build.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_gil.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_prep_in.store(0, std::memory_order_relaxed);
  g_diag_warm_ns_prep_out.store(0, std::memory_order_relaxed);
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
  bool is_align_sensitive = false; // op name in align_sensitive_ops() set
  bool is_broadcast_op = false; // op may broadcast tensor args (mul/add/sub/...).
                                // When true, ``try_warmcache_hit`` decides
                                // per-call (via ``needs_last_dim_one_broadcast``)
                                // whether to materialize the post-broadcast
                                // contig buffers before passing data_ptrs to
                                // PrepareInputs. ``build_cache_key`` always
                                // uses RAW input shapes regardless — keying on
                                // post-broadcast shapes earlier caused
                                // (4,8,16)+() and (4,8,16)+(4,8,1) to collide
                                // on a single key (2026-04-30).
};

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

// True iff `strides` is the standard row-major contiguous stride for `shape`,
// i.e. strides[i] == product(shape[i+1:]). Used by
// install_warmcache_from_pending to assert that runtimes only land in the
// cache when compiled for a contig + offset=0 input layout. Matches
// PyTorch's ``Tensor::is_contiguous()`` conventions: zero-numel tensors are
// always contiguous, size-1 dims have a free stride.
bool is_contiguous_row_major(c10::ArrayRef<int64_t> shape, c10::ArrayRef<int64_t> strides) noexcept {
  if (shape.size() != strides.size()) {
    return false;
  }
  int64_t numel = 1;
  for (int64_t d : shape) {
    numel *= d;
  }
  if (numel == 0) {
    return true;
  }
  int64_t expected = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    const int64_t dim = shape[i - 1];
    if (dim != 1 && strides[i - 1] != expected) {
      return false;
    }
    expected *= dim;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Align-penalty fast-path
//
// RBLN device requires last-dim aligned to 64. When an op's output has
// last-dim not divisible by 64, rebel-compiler inserts host pad/depad nodes
// (RblnTensorPadLastDim + RblnTensorWrapHostOps passes) on the op graph,
// producing the chain: host pad → H2D → device → D2H → host depad. For small
// tensors (decode workloads), per-call cost is dominated by this
// orchestration (~700 µs measured) rather than the compute (~5 µs).
//
// We route these calls through cpu_fallback_rbln, which uses v-mem
// host-backed borrow (no D2H copy, no pad). Validated on aten::neg.out for
// LLaMA-1B rotary: decode -10.3% (NEG_ONLY isolation).
//
// Eligibility (`g_align_sensitive_ops`):
//   - Shape-preserving elementwise ops where output last-dim mirrors broadcast
//     of inputs. These include unary (neg, abs, sqrt, log, exp, silu, rsqrt,
//     ceil, floor, sigmoid, logical_not, pow.Tensor_Scalar_out), binary (add,
//     sub, mul, div, div.out_mode, maximum, minimum), comparisons
//     (ne/eq/gt/ge/lt/le), and where.self_out.
//   - clamp.out is shape-preserving with scalar bounds.
//
// Excluded:
//   - mm.out, bmm.out, addmm.out, linear: output last-dim depends on weight,
//     not input. Compute (O(MNK)) usually dominates align cost.
//   - mean.out / max.unary_out / min.unary_out: reductions change output dim.
//   - cat / index ops: not in shim path anyway.
//
// Criterion: check OUT tensor's last-dim (post-broadcast result shape),
// since for binary ops the result shape may be aligned even if one input is
// not (e.g. `mul([B,S,64], [1,1,32]) → [B,S,64]` — device wins, no align
// penalty).
const std::unordered_set<std::string>& align_sensitive_ops() {
  static const std::unordered_set<std::string>* s = new std::unordered_set<std::string>{
      // Unary elementwise
      "aten::neg.out",
      "aten::abs.out",
      "aten::log.out",
      "aten::sigmoid.out",
      "aten::silu.out",
      "aten::rsqrt.out",
      "aten::ceil.out",
      "aten::floor.out",
      "aten::logical_not.out",
      "aten::pow.Tensor_Scalar_out",
      "aten::clamp.out",
      // Binary elementwise (broadcast-aware via OUT tensor check)
      "aten::add.out",
      "aten::sub.out",
      "aten::mul.out",
      "aten::div.out",
      "aten::div.out_mode",
      "aten::maximum.out",
      "aten::minimum.out",
      // Comparisons (output is bool but shape mirrors broadcast)
      "aten::ne.Tensor_out",
      "aten::eq.Tensor_out",
      "aten::gt.Tensor_out",
      "aten::ge.Tensor_out",
      "aten::lt.Tensor_out",
      "aten::le.Tensor_out",
      "aten::ne.Scalar_out",
      "aten::eq.Scalar_out",
      "aten::gt.Scalar_out",
      "aten::ge.Scalar_out",
      "aten::lt.Scalar_out",
      "aten::le.Scalar_out",
      // Ternary (cond, self, other) — output shape is broadcast
      "aten::where.self_out",
  };
  return *s;
}

// Set of ops whose tensor inputs MUST be pre-broadcast in the C++ shim path
// because the compiled runtime expects post-broadcast-shape buffers.
//
// Ops whose tensor inputs MAY require pre-broadcast in the warm-hit path.
// Membership marks the op as "elementwise broadcast-capable"; the actual
// decision is made per-call by ``needs_last_dim_one_broadcast``, which
// inspects the input shapes and triggers ``at::broadcast_tensors`` +
// ``.contiguous()`` only for the specific pattern rebel-backend cannot
// implicit-compile (last-dim ``size==1 → size>1``; see
// ``ops_utils._has_last_dim_size_one_broadcast`` for the full rationale).
//
// Ops NOT in this set get the raw-data-ptr fast path even when their
// inputs have differing shapes — those shapes are assumed to be ones the
// Python wrapper passed RAW to ``compile_rbln_cached`` so the runtime was
// compiled for the pre-broadcast layout (e.g. RMSNorm ``(B,S,H) * (H,)``).
const std::unordered_set<std::string>& broadcast_ops() {
  static const std::unordered_set<std::string>* s = new std::unordered_set<std::string>{
      "aten::add.out",
      "aten::sub.out",
      "aten::mul.out",
      "aten::div.out",
      "aten::div.out_mode",
      "aten::maximum.out",
      "aten::minimum.out",
      "aten::ne.Tensor_out",
      "aten::eq.Tensor_out",
      "aten::gt.Tensor_out",
      "aten::ge.Tensor_out",
      "aten::lt.Tensor_out",
      "aten::le.Tensor_out",
      "aten::where.self_out",
  };
  return *s;
}

// Mirror of ``ops_utils._has_last_dim_size_one_broadcast`` for the warm-hit
// path. Returns true when ANY non-write-alias tensor input has
// ``shape[-1] == 1`` while the broadcast result has ``shape[-1] > 1`` — the
// pattern rebel-compiler raises ``UNEXPECTED_GRAPH`` on (e.g.
// ``output(N,D) * K(N,1)`` from softmax/layernorm backward).
//
// Cheap: walks each tensor's last-dim once, no allocation.
bool needs_last_dim_one_broadcast(torch::jit::Stack* stack, const SchemaCache& cache) {
  auto args = torch::jit::last(stack, cache.num_args);
  int64_t out_last = 0;
  bool any_size_one = false;
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = args[i];
    if (!iv.isTensor())
      continue;
    const auto& t = iv.toTensor();
    if (!t.defined() || cache.is_write_alias[i] || t.dim() == 0)
      continue;
    int64_t last = t.size(t.dim() - 1);
    if (last > out_last)
      out_last = last;
    if (last == 1)
      any_size_one = true;
  }
  return any_size_one && out_last > 1;
}

bool align_penalty_fast_path_check(torch::jit::Stack* stack, const SchemaCache& cache) {
  if (!cache.is_align_sensitive) {
    return false;
  }
  auto args = torch::jit::last(stack, cache.num_args);
  // Prefer checking the OUT tensor's last-dim — it reflects the broadcast
  // result shape, which is what the device graph would produce.
  if (cache.out_positional_idx >= 0) {
    const auto& iv = args[cache.out_positional_idx];
    if (iv.isTensor()) {
      const auto& t = iv.toTensor();
      if (t.defined() && t.dim() > 0) {
        return (t.size(t.dim() - 1) % 64) != 0;
      }
    }
  }
  // No `out` tensor (functional ops): fall back to input shape — pick the
  // first non-write-alias tensor with non-zero dim.
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = args[i];
    if (!iv.isTensor())
      continue;
    const auto& t = iv.toTensor();
    if (!t.defined() || cache.is_write_alias[i] || t.dim() == 0)
      continue;
    return (t.size(t.dim() - 1) % 64) != 0;
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

// fp16/bf16 NaN/Inf bit-pattern check.
//
// Both formats encode NaN/Inf as "exponent field all-ones"; only the field
// width differs (fp16: 5 bits at offset 10, mask ``0x7C00``; bf16: 8 bits
// at offset 7, mask ``0x7F80``). NaN distinguishes itself by a non-zero
// mantissa, Inf has mantissa==0 — we don't care about the distinction for
// fallback routing, either value means "rbln runtime cannot handle this".
inline bool fp16_has_nan_or_inf(const uint16_t* data, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) {
    if ((data[i] & 0x7C00) == 0x7C00) {
      return true;
    }
  }
  return false;
}

inline bool bf16_has_nan_or_inf(const uint16_t* data, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) {
    if ((data[i] & 0x7F80) == 0x7F80) {
      return true;
    }
  }
  return false;
}

using ScannerFn = bool (*)(const uint16_t*, size_t);
inline ScannerFn scanner_for(c10::ScalarType scalar_type) {
  switch (scalar_type) {
    case c10::kHalf:
      return fp16_has_nan_or_inf;
    case c10::kBFloat16:
      return bf16_has_nan_or_inf;
    default:
      TORCH_INTERNAL_ASSERT(false, "missing scanner for ScalarType");
  }
}

// Scan a single tensor for NaN/Inf. Uses ``c10::rbln::borrow_host_ptr`` for
// rbln tensors so a host-latest entry pays no D2H cost; device-latest entries
// will trigger a D2H sync (this is the price of catching NaN/Inf in
// just-computed device data — matches AS-IS Python ``to_cpu(args)`` cost).
//
// Returns false for: undefined / empty / dtype outside the dispatch
// catalog / non-contiguous tensors. Non-catalog dtypes are short-circuited
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
  if (!c10::rbln::is_dispatch_dtype(t.scalar_type())) {
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
    const uint16_t* data = static_cast<const uint16_t*>(t.data_ptr());
    if (data == nullptr)
      return false;
    return scanner(data, n);
  }

  void* ptr = t.data_ptr();
  if (ptr == nullptr)
    return false;

  const size_t nbytes = static_cast<size_t>(t.nbytes());
  if (nbytes == 0)
    return false;

  // Borrow can be rejected for some runtime sub-states (see try_borrow_host_ptr);
  // fall back to a D2H copy and scan that — same answer, at the cost of a copy.
  auto borrowed = c10::rbln::try_borrow_host_ptr(ptr, nbytes);
  if (!borrowed) {
    const at::Tensor cpu_copy = t.cpu();
    const uint16_t* host_data = static_cast<const uint16_t*>(cpu_copy.data_ptr());
    if (host_data == nullptr)
      return false;
    return scanner(host_data, n);
  }
  void* host_raw = reinterpret_cast<void*>(borrowed->host_ptr); // NOLINT(performance-no-int-to-ptr)
  const bool found = scanner(static_cast<const uint16_t*>(host_raw), n);
  c10::rbln::return_borrowed(borrowed->borrow_id, /*updated=*/false);
  return found;
}

// Cheap C++-side pre-check mirroring the cheap branches of
// torch_rbln._internal.ops_utils.is_cpu_fallback_cases():
//   2. dtype outside the dispatch catalog on any input tensor
//   3. all input tensors are scalar (ndim == 0)
//   4. any input tensor is_contiguous() with storage_offset != 0
//   5. NaN/Inf in any input tensor  (non-deploy mode only; mirrors the
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
    // internally filters out dtypes outside the dispatch catalog (catalog
    // dtypes — fp16/bf16 — are the ones that can encode NaN/Inf on the
    // shim path).
    if (nan_inf_scan_enabled && !nan_inf_found && tensor_has_nan_or_inf(t)) {
      nan_inf_found = true;
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
    if (!c10::rbln::is_dispatch_dtype(t.scalar_type())) {
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

// Cheap pre-check: do tensor inputs already share the same shape? If yes,
// broadcast is a no-op — skip at::broadcast_tensors entirely (which has
// non-trivial overhead even in the no-op case from input validation).
inline bool all_input_shapes_equal(torch::jit::Stack* stack, const SchemaCache& cache) {
  auto arguments = torch::jit::last(stack, cache.num_args);
  c10::IntArrayRef ref_shape;
  bool ref_set = false;
  for (size_t i = 0; i < cache.num_args; ++i) {
    const auto& iv = arguments[i];
    if (!iv.isTensor())
      continue;
    const auto& t = iv.toTensor();
    if (!t.defined() || cache.is_write_alias[i])
      continue;
    if (!ref_set) {
      ref_shape = t.sizes();
      ref_set = true;
    } else if (t.sizes() != ref_shape) {
      return false;
    }
  }
  return true;
}

// Build a WarmCache::CacheKey from the current stack's last num_args IValues.
// Tensor args (non-write-alias, defined) become TensorProfiles in their
// positional order. Scalar args become ScalarValues. None/Tensor-list args
// are silently treated as a signal that we cannot warm-cache this call
// (return false; caller skips warm cache and falls through to pybind).
//
// TensorProfile shapes are always the RAW input shapes. Two distinct calls
// that broadcast to the same result shape (e.g. ``(4,8,16) + ()`` and
// ``(4,8,16) + (4,8,1)``) MUST produce different cache keys, otherwise a
// runtime compiled for one would be invoked on the other's inputs (a real
// bug observed 2026-04-30: shape12 runtime was being hit by shape15).
// The caller's ``try_warmcache_hit`` decides per-call whether broadcast is
// needed based on actual input shapes (``needs_last_dim_one_broadcast``);
// install/lookup consistency is guaranteed by raw-shape keys alone.
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
      tp.strides.assign(t.strides().begin(), t.strides().end());
      tp.storage_offset = t.storage_offset();
      tp.device_index = static_cast<int8_t>(t.device().index());
      out_key.inputs.emplace_back(std::move(tp));
    } else if (iv.isNone()) {
      // Treat `None` slot as a Missing scalar — keeps positional structure
      // without requiring us to distinguish "optional scalar absent" from
      // "optional tensor absent"; both just miss if later calls differ.
      out_key.scalars.push_back(ScalarValue::missing());
    } else if (iv.isTensorList() || iv.isList() || iv.isString()) {
      // Bail out to pybind for two distinct kinds of unsupported slots:
      //   * Lists are not handled by the warm-cache path yet (no shim op
      //     uses them).
      //   * Strings (e.g. ``div.out_mode``'s ``rounding_mode='trunc'`` vs
      //     ``'floor'``) are NOT representable in ``ScalarValue`` (which
      //     only knows int/float/bool). Without distinguishing them in the
      //     key, floor's compiled runtime would be hit by a trunc call —
      //     wire mismatch — so we'd rather miss than silently collapse to
      //     Missing.
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

// Hot path: look up the warm-cache entry for `key` and, on hit, drive the
// rebel runtime from C++ — no Python wrapper, no Dynamo recompile check.
// Returns true iff the hit path was taken and the stack has been left with the
// proper return value.
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

  const uint64_t _seg_t0 = now_ns();
  // Hold a shared_ptr to the entry for the rest of the hit path. If a peer
  // thread ``erase``s the same key while we are mid-flight, the entry stays
  // alive until our shared_ptr goes out of scope. Without this, a raw
  // pointer ``find`` could hand back a dangling pointer.
  CacheEntryPtr entry = wc.find(key);
  const uint64_t _seg_t_lookup = now_ns();
  if (!entry)
    return false;

  // Build input-ptr map in the order tensor inputs appear on the stack.
  std::map<uint32_t, uint64_t> dev_in;
  std::map<uint32_t, uintptr_t> cpu_in;

  auto arguments = torch::jit::last(stack, cache.num_args);
  at::Tensor out_tensor;
  uint32_t in_idx = 0;

  // For broadcast ops: collect raw tensor inputs first, broadcast all at once,
  // then materialize each to contig (matching what mul_rbln/add_rbln do in
  // their Python wrappers). Without this, the cached runtime — which was
  // compiled for the broadcast shape — would receive raw-shape data ptrs and
  // fail (size mismatch / OOB read), causing erase + permanent miss.
  // `held_tensors` keeps the materialized contig tensors alive until Run()
  // completes (their data ptrs are what we pass to PrepareInputs).
  std::vector<at::Tensor> held_tensors;
  // Decide whether warm-hit needs to materialize a post-broadcast buffer.
  //
  // - Same-shape inputs: never broadcast; cache key was built from raw shapes
  //   and raw ptrs are what the runtime wants.
  // - Differing shapes: broadcast ONLY when the pattern is one rebel cannot
  //   implicit-compile (last-dim size==1 → size>1; see ``broadcast_ops`` and
  //   ``needs_last_dim_one_broadcast`` above). For other implicit broadcasts
  //   like RMSNorm ``(B,S,H) * (H,)`` the runtime was compiled for raw shapes
  //   in the Python wrapper, so we skip the materialization (~600 ms / step
  //   on LLaMA-1B prefill).
  const bool needs_broadcast =
      cache.is_broadcast_op && !all_input_shapes_equal(stack, cache) && needs_last_dim_one_broadcast(stack, cache);
  if (needs_broadcast) {
    std::vector<at::Tensor> raw_args;
    raw_args.reserve(cache.num_args);
    for (size_t i = 0; i < cache.num_args; ++i) {
      const auto& iv = arguments[i];
      if (!iv.isTensor())
        continue;
      const at::Tensor& t = iv.toTensor();
      if (!t.defined())
        continue;
      if (cache.is_write_alias[i]) {
        // See note in the non-broadcast branch below: non-contig out= writes
        // numel*itemsize contig bytes into a strided view's data_ptr, which
        // corrupts the layout. Bail to pybind miss-path on non-contig out.
        if (!t.is_contiguous()) {
          return false;
        }
        out_tensor = t;
        continue;
      }
      raw_args.push_back(t);
    }
    if (raw_args.size() >= 2) {
      std::vector<at::Tensor> broadcasted;
      try {
        broadcasted = at::broadcast_tensors(raw_args);
      } catch (...) {
        return false;
      }
      held_tensors.reserve(broadcasted.size());
      for (const auto& b : broadcasted) {
        // .contiguous() is a no-op when raw shape already matches broadcast.
        // For expanded views (stride 0), this materializes a contig buffer.
        at::Tensor contig = b.contiguous();
        void* ptr = contig.data_ptr();
        if (ptr == nullptr) {
          return false;
        }
        dev_in.emplace(in_idx++, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
        held_tensors.push_back(std::move(contig));
      }
    } else {
      // Single-tensor broadcast op (shouldn't happen for ops in broadcast_ops()
      // but handle gracefully): fall through to non-broadcast path.
      for (const auto& t : raw_args) {
        void* ptr = t.data_ptr();
        if (ptr == nullptr) {
          return false;
        }
        dev_in.emplace(in_idx++, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
      }
    }
  } else {
    for (size_t i = 0; i < cache.num_args; ++i) {
      const auto& iv = arguments[i];
      if (!iv.isTensor())
        continue;
      const at::Tensor& t = iv.toTensor();
      if (!t.defined())
        continue;
      if (cache.is_write_alias[i]) {
        // Non-contiguous out=view (e.g. ``torch.add(x, y, out=base.t())``)
        // shares the cache key with the contiguous-out variant — out tensors
        // are excluded from build_cache_key — but the cached runtime was
        // compiled assuming contig output. Writing ``numel*itemsize`` contig
        // bytes into a strided view's data_ptr lays values out at the wrong
        // positions (98%+ data mismatch in TestOutTensors pair). Fall through
        // to the pybind path; the Python wrapper materializes a contig out
        // and copies back via the cpu_fallback writeback path.
        if (!t.is_contiguous()) {
          return false;
        }
        out_tensor = t;
        continue;
      }
      // The cache key only includes (dtype, shape) per TensorProfile, so a
      // non-contiguous view shares its key with a contiguous tensor of the
      // same shape — but the cached runtime was compiled assuming contig
      // layout. Driving it with the view's data_ptr() makes the runtime
      // read along the view's strides as if they were contiguous, producing
      // wrong values (observed 2026-05-06: test_compare_cpu_add fp16, 97%
      // mismatch where one input is a select-stride view of a larger base).
      // Fall through to pybind so the Python wrapper materializes via
      // .contiguous() (or the view-aware path) before re-entering compile.
      if (!t.is_contiguous()) {
        return false;
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
  }

  // User-provided out= with a shape that doesn't match the cached runtime's
  // output. PyTorch native eager resizes the out tensor and emits a warning;
  // the hit path can't, since the runtime was compiled for the cached shape
  // and writes ``numel(cached) * itemsize`` contig bytes into out.data_ptr().
  // Fall through to the pybind miss-path so the Python wrapper handles
  // resize-with-warning.
  if (out_tensor.defined() && !entry->out_profiles.empty()) {
    const auto& cached_shape = entry->out_profiles[0].shape;
    if (out_tensor.sizes() != at::IntArrayRef(cached_shape.data(), cached_shape.size())) {
      return false;
    }
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

  const uint64_t _seg_t_io_build = now_ns();

  // The runtime's methods are called through pybind, so the GIL must be held
  // across them. A failure arrives as ``error_already_set``; we clear it and
  // return false so the caller falls through to the miss path (which routes
  // through DynamoRuntime and performs the v-memory bookkeeping that lets the
  // same tensor inputs succeed).
  pybind11::gil_scoped_acquire wc_gil;
  const uint64_t _seg_t_gil = now_ns();
  bool runtime_failed = false;
  bool contract_broken = false;
  // Each phase timer is assigned right after its corresponding runtime call.
  // A throw skips the remaining assignments and lands in the early return
  // below, so the diag accumulators are only read on the all-assigned path.
  // The defaults are therefore never read; they are here so an unassigned
  // timer holds a sane value rather than garbage.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  uint64_t _seg_t_prep_in = _seg_t_gil, _seg_t_prep_out = _seg_t_gil, _seg_t_run = _seg_t_gil;
  auto clear_and_fail = [&]() {
    if (PyErr_Occurred())
      PyErr_Clear();
    runtime_failed = true;
  };
  try {
    entry->prepare_inputs(dev_in, cpu_in);
    _seg_t_prep_in = now_ns();
    entry->prepare_outputs(dev_out, cpu_out);
    _seg_t_prep_out = now_ns();
    entry->run();
    _seg_t_run = now_ns();
  } catch (const pybind11::error_already_set& e) {
    // TypeError is the arity/type mismatch pybind raises before the runtime's
    // body runs: the call this build makes no longer fits rebel's runtime.
    // Unlike the profile-specific failures below, that holds for every profile.
    contract_broken = e.matches(PyExc_TypeError);
    clear_and_fail();
  } catch (const std::exception&) {
    clear_and_fail();
  } catch (...) {
    clear_and_fail();
  }
  if (contract_broken) {
    // Recompiling reinstalls the same call, so retrying costs a compile per
    // dispatch and never succeeds. Shut the fast path off for the process
    // instead; results come from the Python wrapper, which is correct. The
    // flag is consumed by ``warm_cache.install_pending``, which can name what
    // diverged because it holds the contract declaration.
    WarmCache::instance().set_enabled(false);
    WarmCache::instance().clear();
    WarmCache::mark_contract_break();
    return false;
  }
  if (runtime_failed) {
    // The runtime that we cached cannot serve this profile after all (e.g.
    // input v-memory was created by an allocation path the runtime can't
    // resolve). Drop the entry so subsequent dispatches with this key go
    // through the pybind miss path and rebuild via DynamoRuntime, which
    // handles the edge case correctly.
    //
    // Just erasing the C++ entry is not enough: the Python
    // ``compile_rbln_cached`` still holds the compiled callable, and the
    // rebel backend only pushes a DynamoRuntime to ``_runtime_holder`` on
    // its first compile. A bare erase would leave the warm cache empty
    // AND keep the Python compile cache hot, so install_pending would
    // see an empty holder forever. Set the thread-local force-recompile
    // flag — the same thread's next pass through
    // ``compile_and_run_view_aware`` consumes it and forces
    // ``compile_rbln_cached`` to skip its own cache for this key, letting
    // the rebel backend re-instantiate and re-populate the holder.
    WarmCache::instance().erase(key);
    WarmCache::request_force_recompile();
    return false;
  }

  // Pop args, push single return.
  torch::jit::drop(stack, cache.num_args);
  torch::jit::push(stack, out_tensor);
  const uint64_t _seg_t_finalize = now_ns();

  g_diag_warm_n_hits.fetch_add(1, std::memory_order_relaxed);
  g_diag_warm_ns_lookup.fetch_add(_seg_t_lookup - _seg_t0, std::memory_order_relaxed);
  g_diag_warm_ns_io_build.fetch_add(_seg_t_io_build - _seg_t_lookup, std::memory_order_relaxed);
  g_diag_warm_ns_gil.fetch_add(_seg_t_gil - _seg_t_io_build, std::memory_order_relaxed);
  g_diag_warm_ns_prep_in.fetch_add(_seg_t_prep_in - _seg_t_gil, std::memory_order_relaxed);
  g_diag_warm_ns_prep_out.fetch_add(_seg_t_prep_out - _seg_t_prep_in, std::memory_order_relaxed);
  g_diag_warm_ns_run.fetch_add(_seg_t_run - _seg_t_prep_out, std::memory_order_relaxed);
  g_diag_warm_ns_finalize.fetch_add(_seg_t_finalize - _seg_t_run, std::memory_order_relaxed);
  return true;
}

// The boxed kernel that Library::impl points at for every shimmed op.
void generic_shim_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  g_diag_n_total.fetch_add(1, std::memory_order_relaxed);
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
      // Populate align-sensitive flag once per op (avoids per-call string
      // allocation + unordered_set lookup in the hot path).
      const auto& align_set = align_sensitive_ops();
      entry->schema_cache.is_align_sensitive = (align_set.find(op_name) != align_set.end());
      const auto& bcast_set = broadcast_ops();
      entry->schema_cache.is_broadcast_op = (bcast_set.find(op_name) != bcast_set.end());
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
  //
  // Earlier bugs that motivated disabling this shortcut have been fixed at
  // the borrow site: write-alias args are skipped from the borrow loop and
  // the borrow_resize_case is gated on contiguity (see RBLNCPUFallback.cpp).
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
    const uint64_t _fb_t0 = now_ns();
    ::at::native::rbln::cpu_fallback_rbln(op, stack);
    // COST of the fallback (wall ns), so the report can tell "many cheap fallbacks
    // (path overhead)" from "few expensive ones (hidden transfer)". Same slow branch.
    g_diag_ns_fallback.fetch_add(now_ns() - _fb_t0, std::memory_order_relaxed);
    return;
  }

  // Align-penalty fast-path: previously routed shape-preserving elementwise
  // ops with non-64-aligned last-dim through cpu_fallback to skip rebel's
  // host pad → H2D → device → D2H → host depad penalty (~700 µs for tiny
  // decode tensors).
  //
  // **Disabled (2026-04-30):** mixing CPU-fallback (native fp16) and device
  // (cf16) for the same op produces 1-ULP rounding divergence, breaking
  // bit-exact tests (``test/rbln/test_non_zero_storage_offset``) where
  // sibling ops on aligned shapes stay on device. The helper/counter stay
  // compiled so we can re-enable behind an env gate if a workload regresses.
  (void)align_penalty_fast_path_check;
  (void)g_diag_n_align_fastpath;

  // Warm-cache hot path: if we've previously compiled this op for an identical
  // input profile and have the rebel runtime cached, drive the runtime from
  // C++ directly.
  CacheKey key;
  const bool key_ok = build_cache_key(stack, cache, op_name_intern, key);
  if (key_ok) {
    const uint64_t _diag_warm_t0 = now_ns();
    const bool hit = try_warmcache_hit(stack, cache, key);
    if (hit) {
      g_diag_n_warm_hit.fetch_add(1, std::memory_order_relaxed);
      g_diag_ns_warm_hit.fetch_add(now_ns() - _diag_warm_t0, std::memory_order_relaxed);
      return;
    }
  }

  // MISS path: set up thread-local pending install so the Python wrapper can
  // call `_warmcache_install_pending(runtime, out_profiles)` once it finishes
  // compile + first run. The pending context is discarded unconditionally at
  // the end of this function (even on failure / exception) to avoid leaking
  // into subsequent unrelated ops on the same thread.
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
  static constexpr std::array<const char*, 6> kNames = {
      "copy_d2d_host_bounce",
      "copy_h2d_staging",
      "copy_h2d_noncontig_dst",
      "strided_v2v_cpu_fallback",
      "v2v_batch_to_per_entry",
      "host_batch_to_per_entry"};
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
// compile. `out_profiles` is a list of (shape, dtype_str, is_rbln) tuples
// computed by the Python wrapper from the post-compile output tensors.
// ---------------------------------------------------------------------------

bool install_warmcache_from_pending(
    pybind11::object dyn_runtime,
    pybind11::object prepare_inputs,
    pybind11::object prepare_outputs,
    pybind11::object run,
    const std::vector<std::tuple<std::vector<int64_t>, std::string, bool>>& out_profiles) {
  PendingInstall p = take_pending();
  if (!p.valid)
    return false;

  // Layout invariant: ``compile_and_run_view_aware`` only calls
  // ``_install_warm_cache_pending`` when no view recipe was applied
  // (see the ``if not has_views`` gate). The runtime we are about to
  // cache is therefore compiled for a contig + offset=0 input layout.
  //
  // The pending CacheKey was built at C++ shim entry — BEFORE any host
  // materialization or view-recipe replacement — and reflects the
  // original stack tensor's strides and storage_offset. If that
  // original layout is non-contig or offset>0, the next warm-cache hit
  // on the same key would pass the original view's data_ptr to a
  // runtime that expects contig: silent wrong values for permute /
  // transpose / expand views and wrong-offset reads for narrow views.
  // Defense-in-depth against any future install call site that might
  // bypass the Python ``has_views`` gate.
  for (const auto& tp : p.key.inputs) {
    if (tp.storage_offset != 0 || !is_contiguous_row_major(tp.shape, tp.strides)) {
      return false;
    }
  }

  CacheEntry entry;
  entry.py_dyn_runtime = std::move(dyn_runtime);
  entry.prepare_inputs = std::move(prepare_inputs);
  entry.prepare_outputs = std::move(prepare_outputs);
  entry.run = std::move(run);
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
