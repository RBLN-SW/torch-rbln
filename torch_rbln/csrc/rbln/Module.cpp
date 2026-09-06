#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLNModule.hpp>
#include <torch_rbln/csrc/rbln/DispatchShim.h>
#include <torch_rbln/csrc/rbln/WarmCache.h>
#include <torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_adapter.h>
#include <exception>
#include <vector>

namespace {

/**
 * @brief Add PyMethodDef array to a vector, removing null terminators
 *
 * This helper function concatenates PyMethodDef arrays by removing
 * the null terminator from the existing vector and appending new methods.
 * This is used to combine distributed method definitions with the main module.
 *
 * @param method_vector The vector to append methods to
 * @param method_definitions The PyMethodDef array to append
 */
void add_py_method_definitions(std::vector<PyMethodDef>& method_vector, PyMethodDef* method_definitions) {
  if (!method_vector.empty()) {
    // Remove nullptr terminator from existing vector
    method_vector.pop_back();
  }
  while (true) {
    method_vector.push_back(*method_definitions);
    if (!method_definitions->ml_name) {
      break;
    }
    method_definitions++;
  }
}

/**
 * @brief Register public device management API with Python
 *
 * This function registers the public device management functions
 * that are exposed to Python users for device operations.
 *
 * @param module The Python module to register the functions with
 */
void register_public_device_api(py::module_& module) {
  module.def("current_device", &c10::rbln::get_device_index, "Get the current device.");
  // Enumeration never raises (torch treats it as infallible); a malformed RBLN_*
  // config reports 0 with a one-line warning here and raises in full detail from
  // device_count_ensure_non_zero() / the allocation path.
  module.def("device_count", &c10::rbln::get_device_count_nothrow, "Get the number of devices. Never raises.");
  module.def(
      "device_count_ensure_non_zero",
      &c10::rbln::device_count_ensure_non_zero,
      "Number of devices, raising the detailed RBLN_* configuration error if there is not at least one. "
      "For use where a device is actually required.");
  module.def("set_device", &c10::rbln::set_device_index, "Set the current device.");
  module.def(
      "physical_device_count",
      &c10::rbln::get_physical_device_count,
      "Get the number of physical devices (ignores RSD mode).");
  module.def(
      "is_dummy_device",
      &c10::rbln::is_dummy_device,
      "Whether host-backed dummy device mode (RBLN_DUMMY_DEVICE) is active.");
  module.def(
      "is_available",
      &c10::rbln::runtime_available,
      "Whether RBLN is usable as an accelerator: runtime loaded, not shutting down, at least one usable "
      "logical device. The same predicate RBLNHooksInterface::hasRBLN() uses, so Python and C++ cannot "
      "disagree. Never raises.");
  module.def(
      "runtime_available",
      &c10::rbln::runtime_available,
      "Deprecated alias of is_available(), kept for existing callers. Never raises.");
  module.def(
      "runtime_loaded",
      &rbln_runtime_available,
      "Whether the RBLN device runtime is loaded (librbln's rbln_runtime_available()). Never raises.");
  module.def(
      "_set_runtime_shutting_down",
      &c10::rbln::set_runtime_shutting_down,
      "Mark the RBLN runtime as shutting down so late ops stop dispatching into it.");
  module.def(
      "_exchange_device",
      &c10::rbln::exchange_device_index,
      "Exchange the current device and return the original device.");

  // Synchronization
  module.def("synchronize", &c10::rbln::synchronize, "Wait for all pending async transfers on a device.");

  // Memory management functions
  module.def("empty_cache", &c10::rbln::empty_cache, "Release all unoccupied cached memory.");
  module.def("memory_stats", &c10::rbln::memory_stats, "Get memory allocator statistics.");
  module.def(
      "reset_accumulated_memory_stats", &c10::rbln::reset_accumulated_memory_stats, "Reset accumulated memory stats.");
  module.def("reset_peak_memory_stats", &c10::rbln::reset_peak_memory_stats, "Reset peak memory stats.");

  // Register DeviceTopology structures
  py::class_<c10::rbln::DeviceTopologyEntry>(module, "DeviceTopologyEntry")
      .def_property_readonly("logical_device_index", &c10::rbln::DeviceTopologyEntry::getLogicalDeviceIndex)
      .def_property_readonly("physical_device_ids", &c10::rbln::DeviceTopologyEntry::getPhysicalDeviceIds)
      .def_property_readonly("is_aggregated", &c10::rbln::DeviceTopologyEntry::isAggregated);

  py::class_<c10::rbln::DeviceTopology>(module, "DeviceTopology")
      .def_property_readonly("entries", &c10::rbln::DeviceTopology::getEntries)
      .def_property_readonly("unused_physical_device_ids", &c10::rbln::DeviceTopology::getUnusedPhysicalDeviceIds);

  // Direct binding to DeviceMappingManager for device topology
  module.def(
      "_get_device_topology",
      []() -> c10::rbln::DeviceTopology { return c10::rbln::DeviceMappingManager::getInstance().getDeviceTopology(); },
      "Get the complete device topology.");
}

/**
 * @brief Register the low-level stream primitives backing torch.rbln streams.
 *
 * These mirror torch.cuda's _cuda_getCurrentStream / _cuda_setStream: they exchange
 * (stream_id, device_index, device_type) tuples with Python and delegate to the
 * PrivateUse1 guard impl. torch.Stream / torch.Event need no binding of their own.
 *
 * @param module The Python module to register the functions with
 */
void register_stream_api(py::module_& module) {
  using c10::impl::VirtualGuardImpl;
  auto pack = [](const c10::Stream& stream) {
    return std::make_tuple(
        static_cast<int64_t>(stream.id()),
        static_cast<int64_t>(stream.device_index()),
        static_cast<int64_t>(stream.device_type()));
  };
  module.def(
      "_get_current_stream",
      [pack](c10::DeviceIndex device_index) {
        return pack(VirtualGuardImpl(c10::kPrivateUse1).getStream(c10::Device(c10::kPrivateUse1, device_index)));
      },
      "Internal: current rbln stream as (stream_id, device_index, device_type)");
  module.def(
      "_get_default_stream",
      [pack](c10::DeviceIndex device_index) {
        return pack(VirtualGuardImpl(c10::kPrivateUse1).getDefaultStream(c10::Device(c10::kPrivateUse1, device_index)));
      },
      "Internal: default rbln stream as (stream_id, device_index, device_type)");
  module.def(
      "_exchange_stream",
      [pack](int64_t stream_id, c10::DeviceIndex device_index, int32_t device_type) {
        // c10::Stream::unpack3 accepts any valid device type, so a foreign stream id
        // would otherwise be read as an RBLN local stream id.
        TORCH_CHECK(
            static_cast<c10::DeviceType>(device_type) == c10::kPrivateUse1,
            "torch.rbln streams expect device type ",
            c10::DeviceTypeName(c10::kPrivateUse1),
            ", got ",
            c10::DeviceTypeName(static_cast<c10::DeviceType>(device_type)));
        const auto stream = c10::Stream::unpack3(
            static_cast<c10::StreamId>(stream_id), device_index, static_cast<c10::DeviceType>(device_type));
        const auto guard = VirtualGuardImpl(c10::kPrivateUse1);
        // Selecting a stream also selects its device, as torch.cuda and
        // torch.accelerator both do.
        if (guard.getDevice().index() != stream.device_index()) {
          guard.setDevice(stream.device());
        }
        return pack(guard.exchangeStream(stream));
      },
      "Internal: set current rbln stream; returns the previous as (stream_id, device_index, device_type)");
}

// Both callers configure the whole device allocation behind the tensor's base
// address, so a tensor that does not cover its storage is rejected up front
// instead of misbehaving downstream. Covering it is the whole contract: a torch
// view that still spans its storage (base.view(...), base[:]) carries the
// allocation's own address and size, and is accepted.
void check_base_rbln_tensor(const at::Tensor& t, const char* api, const char* name) {
  TORCH_CHECK(t.device().is_privateuseone(), api, ": ", name, " must be an RBLN tensor, got ", t.device());
  TORCH_CHECK(
      t.storage_offset() == 0 && t.is_contiguous() &&
          static_cast<int64_t>(t.storage().nbytes()) == t.numel() * t.element_size(),
      api,
      ": ",
      name,
      " must cover its whole storage (contiguous, storage_offset 0)");
}

/**
 * @brief Register internal API functions with Python
 *
 * This function registers internal API functions that are used
 * by the RBLN backend but not directly exposed to end users.
 * These functions are prefixed with underscore to indicate they are internal.
 *
 * @param module The Python module to register the functions with
 */
void register_internal_api(py::module_& module) {
  // Tensor creation and manipulation functions
  module.def(
      "_create_tensor_from_ptr", &at::native::rbln::create_tensor_from_ptr, "Internal: create tensor from device ptr");

  // Mark the virtual memory as logically zero-initialized without allocating host memory.
  // Preferred implementation of aten::zero_ for large RBLN tensors (e.g. KV-cache).
  module.def(
      "_mark_zeros",
      [](uint64_t vaddr) {
        c10::rbln::mark_zeros(reinterpret_cast<const void*>(vaddr)); // NOLINT(performance-no-int-to-ptr)
      },
      "Internal: mark RBLN virtual memory as zero-initialized (no host alloc)");

  // Materialise a tensor's device allocation up front, for a consumer that reads the physical
  // buffers out of band. Used by torch_rbln.bind_device_memory().
  module.def(
      "_bind_device_memory",
      [](const at::Tensor& tensor) {
        check_base_rbln_tensor(tensor, "bind_device_memory", "tensor");
        c10::rbln::bind_device_memory(tensor.data_ptr(), tensor.storage().nbytes());
      },
      "Internal: give an RBLN tensor a flat single-node device allocation");

  // Set target's device-allocation layout to match ref's, without copying data.
  // Used by torch_rbln.set_device_layout_like().
  module.def(
      "_set_device_layout_like",
      [](const at::Tensor& target, const at::Tensor& ref) {
        // Validate inputs up front so misuse fails with a clear error. dtype
        // must match: a mismatch would reinterpret target's buffer as a
        // different dtype.
        check_base_rbln_tensor(target, "set_device_layout_like", "target");
        check_base_rbln_tensor(ref, "set_device_layout_like", "ref");
        TORCH_CHECK(
            target.device() == ref.device(),
            "set_device_layout_like: target and ref must be on the same device (got ",
            target.device(),
            " and ",
            ref.device(),
            ")");
        TORCH_CHECK(
            target.scalar_type() == ref.scalar_type(),
            "set_device_layout_like: target and ref must have the same dtype (got ",
            target.scalar_type(),
            " and ",
            ref.scalar_type(),
            ")");
        c10::rbln::set_device_layout_like(target.data_ptr(), ref.data_ptr());
      },
      "Internal: configure target's device layout like ref (no data copy)");

  // Logging utilities
  module.def("_log_cpu_fallback", &c10::rbln::log_cpu_fallback, "Internal: log CPU fallback");

  // C++ dispatch shim: install a boxed C++ handler on PrivateUse1 for the given
  // op, with pre-check + cpu_fallback_rbln on fail and Python callback on pass.
  // `skip_dtype_args` names positional arg indices whose dtype must not be
  // compared to float16 (e.g. where.self_out's cond is bool).
  module.def(
      "_dispatch_shim_diag_dump",
      &torch_rbln::shim::diag_dump_dispatch_paths,
      "DIAG: dispatch path counts/timings (n_total, n_fallback, n_warm_hit, n_miss, ns_warm_hit, ns_miss)");
  module.def(
      "_dispatch_shim_diag_reset", &torch_rbln::shim::diag_reset_dispatch_paths, "DIAG: reset dispatch path counters");
  module.def(
      "_dispatch_fallback_by_op",
      &torch_rbln::shim::diag_dump_fallback_by_op,
      "DIAG/PROFILER: per-op CPU-fallback counts (list of (op_name, count), non-zero only)");
  module.def(
      "_dispatch_fallback_by_op_reset",
      &torch_rbln::shim::diag_reset_fallback_by_op,
      "DIAG/PROFILER: reset per-op CPU-fallback counts");
  module.def(
      "_dispatch_recompile_by_op",
      &torch_rbln::shim::diag_dump_recompile_by_op,
      "DIAG/PROFILER: per-op warm-cache miss (recompile) counts (list of (op_name, count), non-zero only)");
  module.def(
      "_dispatch_recompile_by_op_reset",
      &torch_rbln::shim::diag_reset_recompile_by_op,
      "DIAG/PROFILER: reset per-op recompile counts");
  module.def(
      "_dispatch_fallback_reasons",
      &torch_rbln::shim::diag_dump_fallback_reasons,
      "DIAG/PROFILER: cpu_fallback reason counts [dtype-not-fp16, nan/inf input, all-scalar]");
  module.def(
      "_dispatch_fallback_reasons_reset",
      &torch_rbln::shim::diag_reset_fallback_reasons,
      "DIAG/PROFILER: reset cpu_fallback reason counts");
  module.def(
      "_explain_set_trace",
      &torch_rbln::shim::diag_set_trace_enabled,
      "DIAG/PROFILER (A) WHERE: enable/disable opt-in call-site capture (off by default)");
  module.def(
      "_explain_trace_by_op",
      &torch_rbln::shim::diag_dump_trace_by_op,
      "DIAG/PROFILER (A) WHERE: per-op captured call-site (list of (op_name, site))");
  module.def(
      "_explain_trace_by_op_reset",
      &torch_rbln::shim::diag_reset_trace_by_op,
      "DIAG/PROFILER (A) WHERE: reset captured call-sites");
  module.def(
      "_dispatch_shim_align_fastpath_count",
      &torch_rbln::shim::diag_dump_align_fastpath_count,
      "DIAG: count of align-penalty fast-path hits");
  module.def(
      "_dispatch_shim_warm_segments_dump",
      &torch_rbln::shim::diag_dump_warm_segments,
      "DIAG: warm-cache hit per-segment timers (n_hits, ns_lookup, ns_io_build, "
      "ns_prep_in, ns_prep_out, ns_run, ns_finalize)");
  module.def(
      "_dispatch_shim_warm_segments_reset",
      &torch_rbln::shim::diag_reset_warm_segments,
      "DIAG: reset warm-cache hit per-segment timers");

  // PROFILER: hidden host-bounce / fallback counters (always-on, cold-path only).
  // Returns per-site (count, bytes) in BounceSite enum order. See RBLNProfiler.h.
  module.def(
      "_profiler_dump_bounces",
      []() {
        const auto s = c10::rbln::prof::dump_bounces();
        std::vector<std::pair<uint64_t, uint64_t>> out;
        out.reserve(static_cast<size_t>(c10::rbln::prof::kNumBounceSites));
        for (int i = 0; i < c10::rbln::prof::kNumBounceSites; ++i) {
          out.emplace_back(s.count[i], s.bytes[i]);
        }
        return out;
      },
      "PROFILER: per-site (count, bytes) of hidden host bounces, in BounceSite order");
  module.def("_profiler_reset_bounces", &c10::rbln::prof::reset_bounces, "PROFILER: reset hidden-bounce counters");

  // PROFILER: runtime (rebel-compiler) hidden-overhead counters, read from librbln
  // via its public C-API. Per-reason axes are POSITIONAL — their meaning is an
  // internal classification interpreted Python-side, so no internal name crosses
  // this boundary. See rebel/runtime/api/rbln_runtime_api.h (rebel-compiler wheel).
  module.def(
      "_rt_prof_hidden",
      []() {
        const uint32_t n = c10::rbln::rt_prof_hidden_num();
        std::vector<uint64_t> c(n, uint64_t{0}), b(n, uint64_t{0});
        c10::rbln::rt_prof_hidden_get(c.data(), b.data(), n);
        std::vector<std::pair<uint64_t, uint64_t>> out;
        out.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
          out.emplace_back(c[i], b[i]);
        }
        return out;
      },
      "PROFILER: per-cause (count,bytes) of runtime hidden d2h, positional");
  module.def(
      "_rt_prof_reject",
      []() {
        const uint32_t n = c10::rbln::rt_prof_reject_num();
        std::vector<uint64_t> c(n, uint64_t{0}), b(n, uint64_t{0});
        c10::rbln::rt_prof_reject_get(c.data(), b.data(), n);
        std::vector<std::pair<uint64_t, uint64_t>> out;
        out.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
          out.emplace_back(c[i], b[i]);
        }
        return out;
      },
      "PROFILER: per-reason (count,bytes) of v2v plan reject, positional");
  module.def(
      "_rt_prof_host_sync",
      []() {
        uint64_t dc = 0, db = 0, hc = 0, hb = 0;
        c10::rbln::rt_prof_host_sync_d2h(&dc, &db);
        c10::rbln::rt_prof_host_sync_h2d(&hc, &hb);
        std::vector<std::pair<uint64_t, uint64_t>> out{{dc, db}, {hc, hb}}; // [0]=d2h, [1]=h2d
        return out;
      },
      "PROFILER: [(d2h_count,d2h_bytes),(h2d_count,h2d_bytes)] real device<->host transfers");
  module.def(
      "_rt_prof_memory",
      []() {
        uint64_t cur = 0, peak = 0;
        c10::rbln::rt_prof_memory(&cur, &peak);
        return std::make_pair(cur, peak);
      },
      "PROFILER: (current,peak) device-memory gauge (process-global, both alloc paths)");
  module.def(
      "_rt_prof_reset",
      []() { c10::rbln::rt_prof_reset(); },
      "PROFILER: reset runtime hidden/reject/host-sync counters for an explain region");
  module.def(
      "_register_cpp_shim",
      &torch_rbln::shim::register_cpp_shim,
      "Internal: install a C++ dispatch shim for an op and register its Python impl",
      pybind11::arg("op_name"),
      pybind11::arg("py_fn"),
      pybind11::arg("skip_dtype_args") = std::vector<size_t>{});

  // Warm-cache API. The C++ shim populates a thread-local `pending` entry on
  // every miss-path dispatch; the generated Python wrapper calls
  // `_warmcache_install_pending` after a successful first compile + run so the
  // runtime is cached for subsequent invocations with the same input profile.
  module.def(
      "_warmcache_install_pending",
      &torch_rbln::shim::install_warmcache_from_pending,
      "Internal: install a warm-cache entry from the thread-local pending key "
      "set by the shim on the way into the miss path",
      pybind11::arg("dyn_runtime"),
      pybind11::arg("runtime_handle"),
      pybind11::arg("out_profiles"));

  module.def(
      "_warmcache_set_enabled",
      [](bool v) { torch_rbln::warmcache::WarmCache::instance().set_enabled(v); },
      "Internal: enable/disable the warm-runtime cache path globally");
  module.def(
      "_warmcache_is_enabled",
      []() { return torch_rbln::warmcache::WarmCache::instance().is_enabled(); },
      "Internal: query whether warm-cache is currently enabled");
  module.def(
      "_warmcache_size",
      []() { return torch_rbln::warmcache::WarmCache::instance().size(); },
      "Internal: number of entries in the warm-cache (debug/bench only)");
  module.def(
      "_warmcache_clear",
      []() { torch_rbln::warmcache::WarmCache::instance().clear(); },
      "Internal: drop all warm-cache entries (tests / benchmarks)");
  module.def(
      "_warmcache_is_building",
      []() { return torch_rbln::warmcache::WarmCache::is_building_entry(); },
      "Internal: true iff the current thread is inside the miss-path compile");
  module.def(
      "_warmcache_enter_building",
      []() { torch_rbln::warmcache::WarmCache::enter_building(); },
      "Internal: mark the current thread as inside the miss-path compile "
      "(reentrancy guard; pairs with _warmcache_exit_building)");
  module.def(
      "_warmcache_exit_building",
      []() { torch_rbln::warmcache::WarmCache::exit_building(); },
      "Internal: clear the miss-path reentrancy flag set by _warmcache_enter_building");
  module.def(
      "_warmcache_consume_force_recompile",
      []() { return torch_rbln::warmcache::WarmCache::consume_force_recompile(); },
      "Internal: consume the thread-local force-recompile flag set by a failed "
      "warm-cache hit. Returns True iff a flag was pending; clears it.");
  module.def(
      "_warmcache_request_force_recompile",
      []() { torch_rbln::warmcache::WarmCache::request_force_recompile(); },
      "Internal: set the thread-local force-recompile flag. Production callers "
      "rely on the C++ shim auto-setting this on hit-failure; this binding "
      "exists so tests can exercise the consume/clear pair without engineering "
      "a runtime soft-failure.");

  // CPU fast-path registry introspection. Returns True iff a handler is
  // registered for the given fully-qualified op name (e.g. "aten::rsqrt.out").
  // Used by tests to verify the static-init registration fired without
  // needing a live OperatorHandle.
  module.def(
      "_cpu_fast_path_registered",
      [](const std::string& op_name) {
        return at::native::rbln::CPUFastPathRegistry::instance().has_handler_for_op(op_name);
      },
      "Internal: returns True iff a CPU fast-path handler is registered for the given op name");

  // (B) explain: rebel-runtime (librbln) boundary timing. Gated so it is OFF
  // (one relaxed atomic load per boundary call) unless an explain region enables
  // it; lets the profiler split host overhead into runtime vs torch-side dispatch.
  module.def(
      "_rt_timing_enable",
      [](bool on) { c10::rbln::rt_timing_enable(on); },
      "Internal: enable/disable librbln boundary timing for an explain region");
  module.def("_rt_timing_reset", []() { c10::rbln::rt_timing_reset(); }, "Internal: zero the librbln boundary timers");
  module.def(
      "_rt_timing_get",
      []() {
        std::vector<uint64_t> buf(2 * c10::rbln::kRtTimingN, uint64_t{0});
        c10::rbln::rt_timing_get(buf.data());
        std::vector<std::pair<uint64_t, uint64_t>> out;
        out.reserve(c10::rbln::kRtTimingN);
        for (std::size_t i = 0; i < c10::rbln::kRtTimingN; ++i) {
          out.emplace_back(buf[2 * i], buf[2 * i + 1]);
        }
        return out;
      },
      "Internal: per-primitive (ns, calls) spent inside librbln boundary calls this region");

  // Fallback configuration
  module.def(
      "_is_fallback_disabled",
      &c10::rbln::is_fallback_disabled,
      "Internal: check if specified fallback category is disabled");

  // Process-wide RBLN vmemory file offloading toggle. Exposed only as an
  // internal helper so torch.rbln.offload (in torch_rbln/memory.py) can drive
  // it; users should go through the offload() context manager rather than
  // calling this directly.
  module.def(
      "_set_file_offloading_enabled",
      &c10::rbln::set_file_offloading_enabled,
      "Internal: enable or disable process-wide RBLN vmemory file offloading.");

  module.def(
      "_release_offload_temp_storage",
      &c10::rbln::release_offload_temp_storage,
      "Internal: remove this process's file offloading temp files and directories.");

  // torch.profiler (kineto) integration
  module.def(
      "_register_kineto_profiler",
      &rbln::profiler::kineto::register_rbln_kineto_profiler,
      "Internal: register the rbln torch.profiler (kineto) bridge");
}

py::tuple supported_dtypes_to_tuple(c10::ArrayRef<c10::ScalarType> scalar_types) {
  const auto n = scalar_types.size();
  py::tuple out(n);
  for (size_t i = 0; i < n; ++i) {
    // `getTHPDtype` returns a borrowed reference to a process-wide singleton.
    auto* const dtype_obj = reinterpret_cast<PyObject*>(torch::getTHPDtype(scalar_types[i]));
    out[i] = py::reinterpret_borrow<py::object>(dtype_obj);
  }
  return out;
}

/**
 * @brief Register dtype tuple getters consumed by Python `SupportedDtypes`.
 *
 * @param module The Python module to register the functions with
 */
void register_supported_dtypes_api(py::module_& module) {
  module.def(
      "_dispatch_dtypes",
      [] {
        // Catalog plus the TORCH_RBLN_DISPATCH_DTYPES extension, evaluated now: the
        // Python SupportedDtypes snapshot is taken at import, so the env var must be
        // set before `import torch_rbln`.
        const auto v = c10::rbln::dispatch_dtypes_rt();
        return supported_dtypes_to_tuple(c10::ArrayRef<c10::ScalarType>(v));
      },
      "Internal: eager-dispatch supported dtypes (catalog + TORCH_RBLN_DISPATCH_DTYPES)");
  module.def(
      "_dispatch_catalog_dtypes",
      [] { return supported_dtypes_to_tuple(c10::rbln::kDispatchDtypes); },
      "Internal: the built-in eager-dispatch catalog, without the TORCH_RBLN_DISPATCH_DTYPES extension");
  module.def(
      "_dispatch_strict_dtypes",
      [] {
        const auto v = c10::rbln::strict_dispatch_dtypes_rt();
        return supported_dtypes_to_tuple(c10::ArrayRef<c10::ScalarType>(v));
      },
      "Internal: dtypes under strict dispatch (TORCH_RBLN_DISPATCH_STRICT), evaluated now");
  module.def(
      "_sdpa_dtypes",
      [] { return supported_dtypes_to_tuple(c10::rbln::kSdpaDtypes); },
      "Internal: SDPA kernel supported dtypes");
  module.def(
      "_amp_dtypes",
      [] { return supported_dtypes_to_tuple(c10::rbln::kAmpDtypes); },
      "Internal: AMP autocast supported dtypes");
}

/**
 * @brief Register distributed method definitions with the module
 *
 * This function adds distributed method definitions to the global method vector.
 * These methods are used for distributed training functionality such as
 * ProcessGroupRBLN initialization.
 *
 * @param method_vector The global method vector to add distributed methods to
 */
void register_distributed_method(std::vector<PyMethodDef>& method_vector) {
  add_py_method_definitions(method_vector, torch_rbln::distributed::get_distributed_method_definitions());
}

} // anonymous namespace

// Global vector to store all method definitions
static std::vector<PyMethodDef> global_method_definitions;

/**
 * @brief Initialize the torch_rbln._C module
 *
 * This function creates the main torch_rbln._C module and registers:
 * 1. Distributed functions from ProcessGroupRBLNModule
 * 2. RBLN-specific bindings
 *
 * REFACTORING NOTE - Circular Dependency Resolution:
 * ================================================
 * This implementation was refactored to resolve circular dependency issues
 * that occurred when trying to import torch._C._distributed_c10d during module
 * initialization. The original approach caused build errors related to _C.pyi
 * generation and import conflicts.
 *
 * Key changes made:
 * 1. Separated ProcessGroupRBLN bindings into ProcessGroupRBLNModule.cpp/hpp
 * 2. Used lazy loading pattern - distributed bindings are initialized on-demand
 * 3. Structured registration into logical functions for better maintainability:
 *    - register_distributed_method(): Adds distributed method definitions
 *    - register_public_device_api(): Registers public device functions
 *    - register_internal_api(): Registers internal backend functions
 *    - register_supported_dtypes_api(): Registers supported dtype getters
 *
 * This approach follows the ascend-torch pattern and ensures clean separation
 * of concerns while avoiding circular import issues.
 *
 * The initialization is organized into logical sections for better maintainability:
 * - Method definitions setup
 * - Module creation
 * - RBLN-specific bindings registration
 *
 * @return PyObject* The created module
 */
extern "C" PyObject* initModule() {
  // Step 1: Register distributed method definitions
  register_distributed_method(global_method_definitions);

  // Step 2: Create the module definition
  static struct PyModuleDef torch_rbln_module_definition = {
      PyModuleDef_HEAD_INIT, "torch_rbln._C", nullptr, -1, global_method_definitions.data()};
  PyObject* created_module = PyModule_Create(&torch_rbln_module_definition);

  // Step 3: Initialize RBLN-specific bindings
  py::gil_scoped_acquire gil_acquire;
  py::module_ python_module = py::reinterpret_borrow<py::module_>(created_module);

  // Set module documentation
  python_module.doc() = "Torch RBLN low‑level bindings.";

  // Step 4: Register all RBLN components
  register_public_device_api(python_module);
  register_stream_api(python_module);
  register_internal_api(python_module);
  register_supported_dtypes_api(python_module);

  c10::rbln::register_rbln_device_mapping_initialized_callback([]() {
    py::gil_scoped_acquire gil;
    try {
      py::module_ m = py::module_::import("torch_rbln.device.device");
      m.attr("_on_device_mapping_ready_from_cpp")();
    } catch (const py::error_already_set&) {
      PyErr_Clear();
    } catch (const std::exception& e) {
      RBLN_LOG_DEBUG("device_mapping_ready_callback: {}", e.what());
    } catch (...) {
      RBLN_LOG_DEBUG("device_mapping_ready_callback: unknown exception");
    }
  });

  return created_module;
}

/**
 * @brief Python module initialization entry point
 *
 * This is the standard Python C extension entry point that gets called
 * when the module is imported. It delegates to initModule() for the
 * actual initialization work.
 *
 * @return PyObject* The initialized module
 */
PyMODINIT_FUNC PyInit__C(void) {
  return initModule();
}
