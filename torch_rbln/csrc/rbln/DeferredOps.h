#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_rbln/csrc/rbln/OpFunction.h>

#include <memory>
#include <optional>
#include <vector>

namespace torch_rbln::deferred {

/**
 * @brief Runs `function` over `inputs` as OpFunction::run does, or, while TORCH_RBLN_DEFER_OPS
 * defers eager ops, defers the run and returns the tensors it will write: those of `out` given,
 * and new ones for the other results. With `own_out`, the tensors of `out` are the call's own, made
 * for its results and held by nothing else yet, as a structured kernel's are.
 *
 * The runs deferred on a stream form a segment, which runs as one function, compiled from the
 * modules of its ops, once anything else is queued on the stream (runtime_stream in
 * c10/rbln/RBLNRuntime.h), on a synchronize, when the cache of the device is emptied, when
 * torch.rbln.set_stream leaves the stream, or once the segment holds TORCH_RBLN_DEFER_OPS runs;
 * until then nothing reads or writes the tensors of its runs, which stand for the bytes they view
 * when the run is submitted. The function writes the tensors of `out` a caller gave, the inputs a
 * run updates in place, and the other results a tensor still holds then. A segment runs op by op
 * the first time it is seen, and when the compiler refuses it or fails on it.
 *
 * A run is deferred only when it does all its work on the device over tensors it binds in place;
 * otherwise, and while RBLN_LAUNCH_BLOCKING has each operation run as it is queued, the run is
 * queued after the segment runs. A run of a function that computes by rows (TORCH_RBLN_EAGER_BUCKETS)
 * over fewer rows gets results whose storage holds every row. A segment whose runs all compute by
 * rows at one extent over one count of rows, and read one another's results by rows, compiles to a
 * function that computes by rows too, or runs op by op when the compiler cannot make one.
 */
std::optional<std::vector<at::Tensor>> submit(
    const std::shared_ptr<OpFunction>& function,
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out = {},
    bool own_out = false);

/**
 * @brief Queues the segment deferred on stream `stream` of device `device_index`, or those on
 * every stream of it when none is given; the runner of c10::rbln::set_deferred_ops_runner.
 */
void run(c10::DeviceIndex device_index, std::optional<c10::StreamId> stream);

/**
 * @brief Queues the segments deferred on every stream.
 */
void run_all();

/**
 * @brief Lets go of the functions of the segments `function` runs in.
 */
void forget(const OpFunction* function);

void init_deferred_ops_bindings(pybind11::module& module);

} // namespace torch_rbln::deferred
