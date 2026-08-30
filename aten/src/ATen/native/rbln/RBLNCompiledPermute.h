#pragma once

#include <ATen/core/Tensor.h>

#ifdef TORCH_RBLN_BUILD_MAIN_LIB
#define TORCH_RBLN_PERMUTE_API __attribute__((visibility("default")))
#else
#define TORCH_RBLN_PERMUTE_API
#endif

namespace at::native::rbln {

// A permuted device-to-device copy walks one strided range per contiguous inner block, which
// for a head<->token swap is a 256 B block and runs at ~1.3 GB/s. Running the permutation as a
// compiled device program instead moves 117 MB in ~0.4 ms. The program is compiled once per
// (shape, dtype, dims) and cached on disk (TORCH_RBLN_CACHE_DIR).

// `dst.copy_(src)` for a large permuted RBLN source: runs the program straight into `dst`.
// False when it does not apply -- the caller falls back to strided_v2v_copy.
TORCH_RBLN_PERMUTE_API bool try_compiled_permute_copy(const at::Tensor& dst, const at::Tensor& src);

using CompiledPermuteImpl = void (*)(const at::Tensor&, at::IntArrayRef, int64_t, const at::Tensor&);
TORCH_RBLN_PERMUTE_API void set_compiled_permute_impl(CompiledPermuteImpl impl);

} // namespace at::native::rbln
