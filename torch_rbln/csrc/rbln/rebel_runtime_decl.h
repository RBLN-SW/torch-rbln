#pragma once

// Forward declaration of rebel's PyRblnSyncRuntime.
//
// Including the upstream header (rebel/pyrbln_impl/compiled_model.h) pulls in
// a transitive dependency on absl and other rebel-internal symbols that are
// not part of the shipped prod librbln.so. The linker resolves these method
// symbols against librbln.so at load time; signatures here MUST match
// upstream. Reference:
//   /home/chanheo/rebel_compiler/rebel/include/rebel/pyrbln_impl/
//   compiled_model.h
//
// Centralized in a thin header so every torch-rbln translation unit that
// drives PyRblnSyncRuntime (WarmCache, DispatchShim, future fast-path
// handlers) sees one canonical declaration. Adding/removing a method here
// touches every caller's include set and propagates link-time signature
// checks consistently.

#include <cstdint>
#include <map>

namespace rbln {

class PyRblnSyncRuntime {
 public:
  void Run();
  void PrepareInputs(
      const std::map<uint32_t, uint64_t>& device_inputs,
      const std::map<uint32_t, uintptr_t>& cpu_inputs);
  void PrepareOutputs(
      const std::map<uint32_t, uint64_t>& device_outputs,
      const std::map<uint32_t, uintptr_t>& cpu_outputs);
};

} // namespace rbln
