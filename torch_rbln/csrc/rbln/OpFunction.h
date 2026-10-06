#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <rbln/runtime/executor.h>
#include <torch/csrc/utils/pybind.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace torch_rbln {

namespace rt = ::rbln::runtime;

struct StateKey;

/**
 * @brief A compiled function run on torch tensors: an eager op, or a graph torch.compile hands
 * over. The args a call's tensors bind to hold their values as torch holds a contiguous tensor,
 * so a torch-rbln tensor binds in place; a CPU tensor, or one off the alignment of its arg, binds
 * through a copy, which goes back into it once the run is done when the run writes the arg. An
 * arg the compiler lays out otherwise goes through the host: a call's tensor is encoded into it,
 * and a result is decoded from it once the run is done. A tensor a program holds as such an arg
 * holds it (see c10/rbln/RBLNHeld.h) binds in place instead; with `hold_written`, a tensor over a
 * whole allocation that a run writes in place is first put so, as a graph writes the same tensor
 * on every call.
 *
 * `inputs` names the inputs a call passes, in order. The other args are made from state, whose
 * values `state` gives and `set_state` replaces, and are written to each device the function
 * runs on. The results are the function's; those `host_results` flags come back as CPU tensors
 * once the run is done. Each device gets an executor of its own, and runs queue on the current
 * stream of that device.
 */
class OpFunction {
 public:
  struct Output {
    std::vector<int64_t> shape;
    at::ScalarType dtype;
  };

  OpFunction(
      const std::string& bytes,
      const std::vector<std::string>& inputs,
      const std::map<std::string, at::Tensor>& state,
      std::vector<bool> host_results = {},
      bool hold_written = false);

  size_t num_inputs() const {
    return num_inputs_;
  }
  const std::vector<Output>& outputs() const {
    return outputs_;
  }
  // When the function last ran, on a clock of the process that every run of an
  // OpFunction moves; zero before its first run.
  uint64_t last_run() const {
    return last_run_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Replaces the state `values` names, and writes the args made from it anew on every
   * device, once the runs before have finished.
   */
  void set_state(const std::map<std::string, at::Tensor>& values);

  /**
   * @brief Runs over `inputs`, writing the tensor of `out` given for a result and a new one for
   * each other result, and returns the results; none when a tensor does not hold what the
   * function takes where the function takes it.
   */
  std::optional<std::vector<at::Tensor>> run(
      c10::ArrayRef<at::Tensor> inputs,
      c10::ArrayRef<std::optional<at::Tensor>> out = {});

 private:
  // `through_host` when the arg holds its value otherwise than torch holds the tensor.
  struct Binding {
    size_t arg;
    size_t input;
    bool through_host = false;
  };
  // A result is written to arg `arg`, or is input `input` the function updates.
  struct Result {
    std::optional<size_t> arg;
    std::optional<size_t> input;
    bool host = false;
    bool through_host = false;
  };
  struct Slot {
    std::mutex mutex;
    std::shared_ptr<rt::Device> device;
    std::shared_ptr<rt::Executor> executor;
  };
  // A tensor a run binds through a copy of it, which goes back into the tensor once the run is
  // done when the run writes the arg.
  struct Copy {
    at::Tensor tensor;
    at::Tensor copy;
    bool written = false;
  };

  // `tensor` as arg `arg` takes it, or none if it is not the arg's; with `copies`, a CPU tensor,
  // or one held where the arg cannot take it, binds through a copy that `copies` keeps.
  rt::Binding over(
      const at::Tensor& tensor,
      size_t arg,
      c10::DeviceIndex device_index,
      std::vector<Copy>* copies) const;
  // `tensor` in place as arg `arg`, which the compiler lays out otherwise than torch, takes it: a
  // tensor a program holds as the arg holds it, or, with `hold_written_`, one the arg is written
  // through, which it holds so first; none for any other.
  rt::Binding held(const at::Tensor& tensor, size_t arg, c10::DeviceIndex device_index) const;
  // The host tensor of state arg `arg`, encoded from the state now.
  std::shared_ptr<rt::HostTensor> encoded(size_t arg) const;
  // A host tensor of arg `arg` encoded from the value of `tensor`, or none if it is not the arg's.
  std::shared_ptr<rt::HostTensor> encoded(const at::Tensor& tensor, size_t arg) const;
  // What tells the tensor of state arg `arg` on `device` apart from others.
  StateKey state_key(size_t arg, const rt::Device& device) const;
  // `args` with the state args that share pools of pages with them, which a
  // device holds in one buffer with them.
  std::vector<size_t> with_page_sharers(std::vector<size_t> args) const;
  // Binds state `args` of each of `slots` to tensors on its device that hold the state now:
  // those another function holding the arg alike has made, or new ones, each encoded once.
  void bind_state(const std::vector<Slot*>& slots, const std::vector<size_t>& args) const;
  Slot& slot(c10::DeviceIndex device_index);

  std::shared_ptr<rt::Function> fn_;
  size_t num_inputs_ = 0;
  bool hold_written_ = false;
  std::vector<Binding> bindings_;
  std::vector<Result> results_;
  std::vector<Output> outputs_;
  std::vector<size_t> state_args_;
  // Guards the state and the slots.
  std::mutex slots_mutex_;
  std::map<std::string, at::Tensor> state_values_;
  std::map<c10::DeviceIndex, std::unique_ptr<Slot>> slots_;
  std::atomic<uint64_t> last_run_{0};
};

void init_op_function_bindings(pybind11::module& module);

} // namespace torch_rbln
