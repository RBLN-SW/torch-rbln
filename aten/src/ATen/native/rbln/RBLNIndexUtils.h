#pragma once

#include <ATen/core/Tensor.h>
#include <c10/rbln/RBLNLogging.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace at::native::rbln {

/**
 * @brief Pull a 0- or 1-D index tensor onto the host as a contiguous int64
 * vector. Accepts int32 or int64 input; widens int32 to int64.
 *
 * @param index   The index tensor (any device / layout).
 * @param op_name Error-message prefix (e.g. "index_select", "index_copy"),
 *                used so RBLN_CHECK messages name the op the caller is in.
 */
inline std::vector<int64_t> read_index_to_host(const at::Tensor& index, const char* op_name) {
  RBLN_CHECK(index.dim() <= 1, "{}: index must be 0- or 1-D, got {}-D", op_name, index.dim());
  RBLN_CHECK(
      index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
      "{}: index dtype must be int32 or int64, got {}",
      op_name,
      c10::str(index.scalar_type()));

  at::Tensor host = index;
  if (!host.device().is_cpu())
    host = host.cpu();
  if (!host.is_contiguous())
    host = host.contiguous();

  const int64_t n = host.numel();
  std::vector<int64_t> values(n);
  if (host.scalar_type() == at::kLong) {
    std::memcpy(values.data(), host.data_ptr<int64_t>(), n * sizeof(int64_t));
  } else {
    const auto* src = host.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i)
      values[i] = static_cast<int64_t>(src[i]);
  }
  return values;
}

/**
 * @brief A maximal-length run of consecutive +1 increments in an index list.
 *
 * `value` is the axis position the first element of the run points to (i.e.
 * `idx[pos]`), and `pos` is the position in the index list where the run
 * starts. For a run of length L, the run covers indices `idx[pos..pos+L)`
 * with values `value, value+1, ..., value+L-1`.
 *
 * Callers narrow the indexed-into and indexed-from tensors accordingly:
 *   - index_select: self.narrow(axis, run.value, run.length)
 *                   out.narrow(axis, run.pos, run.length)
 *   - index_copy:   out.narrow(axis, run.value, run.length)
 *                   source.narrow(axis, run.pos, run.length)
 */
struct IndexRun {
  int64_t value;
  int64_t pos;
  int64_t length;
};

/**
 * @brief Group a sequence of integer indices into maximal +1-increment runs.
 *
 * Lets a kernel emit one v2v of `length * inner_block_bytes` per run instead
 * of one per element when the input axis values happen to be consecutive.
 */
inline std::vector<IndexRun> coalesce_runs(const std::vector<int64_t>& idx) {
  std::vector<IndexRun> runs;
  if (idx.empty())
    return runs;
  IndexRun cur{idx[0], 0, 1};
  for (int64_t i = 1; i < static_cast<int64_t>(idx.size()); ++i) {
    if (idx[i] == cur.value + cur.length) {
      cur.length += 1;
    } else {
      runs.push_back(cur);
      cur = IndexRun{idx[i], i, 1};
    }
  }
  runs.push_back(cur);
  return runs;
}

} // namespace at::native::rbln
