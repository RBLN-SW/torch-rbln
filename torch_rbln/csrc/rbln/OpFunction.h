#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/rbln/RBLNRuntime.h>
#include <c10/util/ArrayRef.h>
#include <rebel/v2/runtime/executor.h>
#include <rebel/v2/runtime/flags.h>
#include <torch/csrc/utils/pybind.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace torch_rbln {

namespace rt = ::rebel::v2::runtime;

struct StateKey;

// TORCH_RBLN_COMPILED_OPS: the compiled ops a process keeps.
extern const rt::Flag<int64_t> kCompiledOps;

/**
 * @brief A compiled function run on torch tensors: an eager op, or a graph torch.compile hands
 * over. The args a call's tensors bind to hold their values as torch holds a contiguous tensor,
 * so a torch-rbln tensor binds in place; a CPU tensor, one off the alignment of its arg, or one of
 * a type (see c10/rbln/RBLNTyped.h), binds through a copy, which goes back into it once the run is
 * done when the run writes the arg. An arg the compiler lays out otherwise goes through the host: a
 * call's tensor is encoded into it, and a result is decoded from it once the run is done. A tensor
 * made in the arg's type binds to such an arg in place instead.
 *
 * `inputs` names the inputs a call passes, in order. The args made from state alone, whose values
 * `state` gives and `set_state` replaces, are written to each device the function runs on, or kept
 * on the host for the host work that reads them. Every call makes the args the run reads of what
 * the host works out of its inputs, and once the run is done the results the host works out of
 * what the run wrote. The results are the function's; those `host_results` flags come back as CPU
 * tensors once the run is done. Each device gets an executor of its own, and runs queue on the
 * current stream of that device.
 *
 * A function given `rows` computes row i of each result from row i of each input `rows` names, and
 * from no other row of them, as an elementwise op does; its maker vouches for this. A call may pass
 * those inputs with fewer rows than the function takes, the same count for each, and gets results
 * of that many rows. The run computes every row the function takes and drops those past the call's,
 * whatever they were computed from: an input binds over the bytes after its rows up to the end of
 * its allocation's block, or else through a copy padded to the function's extent, on the host for an
 * arg that goes through it. A result keeps the storage of every row, so that a later such call reads
 * it in place. A function whose results or args of those inputs cannot be bound so takes calls of
 * all of its rows only (see `row_extent`).
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
      const std::vector<size_t>& rows = {});

  size_t num_inputs() const {
    return num_inputs_;
  }
  const std::vector<Output>& outputs() const {
    return outputs_;
  }
  // Whether a call may pass fewer rows of the inputs `rows` names.
  bool row_wise() const {
    return !row_inputs_.empty();
  }
  // The inputs a run writes in place, which no result names.
  const std::vector<size_t>& updated_inputs() const {
    return updated_inputs_;
  }
  // When the function last ran, on a clock of the process that every run of an
  // OpFunction moves; zero before its first run.
  uint64_t last_run() const {
    return last_run_.load(std::memory_order_relaxed);
  }
  // Moves `last_run` as a run does, for a run of a function that does this one's work.
  void note_run();

  /**
   * @brief Replaces the state `values` names, and writes the args made from it anew on every
   * device, once the runs before have finished.
   */
  void set_state(const std::map<std::string, at::Tensor>& values);

  /**
   * @brief Runs the function's collectives over `communicator`, on every device it runs on.
   */
  void set_communicator(std::shared_ptr<rt::Communicator> communicator);

  /**
   * @brief Runs over `inputs`, writing the tensor of `out` given for a result and a new one for
   * each other result, and returns the results; none when a tensor does not hold what the
   * function takes where the function takes it.
   */
  std::optional<std::vector<at::Tensor>> run(
      c10::ArrayRef<at::Tensor> inputs,
      c10::ArrayRef<std::optional<at::Tensor>> out = {});

  /**
   * @brief Whether a run does all its work on the device, with no host work before or after it
   * and every arg held as torch holds a contiguous tensor: what a deferred run must be.
   */
  bool runs_on_device() const {
    return runs_on_device_;
  }

  /**
   * @brief Whether a run on device `device_index` over `inputs`, writing the tensors of `out`
   * given, binds each of them in place, through no copy.
   */
  bool binds_in_place(
      c10::ArrayRef<at::Tensor> inputs,
      c10::ArrayRef<std::optional<at::Tensor>> out,
      c10::DeviceIndex device_index) const;

 private:
  // `through_host` when the arg holds its value otherwise than torch holds the tensor; `row` when
  // the input is one of `rows`.
  struct Binding {
    size_t arg;
    size_t input;
    bool through_host = false;
    bool row = false;
  };
  // A result is written to arg `arg`, or is input `input` the function updates; `after` when the
  // host makes the arg once the run is done.
  struct Result {
    std::optional<size_t> arg;
    std::optional<size_t> input;
    bool host = false;
    bool through_host = false;
    bool after = false;
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

  // Where `tensor` lies when arg `arg` takes it there on device `device_index`, or none; with `extent`,
  // its leading axis stands at that extent, over the bytes past its rows to the end of its block.
  std::optional<c10::rbln::Location> in_place(
      const at::Tensor& tensor,
      size_t arg,
      c10::DeviceIndex device_index,
      std::optional<int64_t> extent = std::nullopt) const;
  // `tensor` as arg `arg` takes it, or none; with `copies`, a CPU tensor or one the arg cannot take
  // where it lies binds through a copy that `copies` keeps. `extent` is as for `in_place`.
  rt::Binding over(
      const at::Tensor& tensor,
      size_t arg,
      c10::DeviceIndex device_index,
      std::vector<Copy>* copies,
      std::optional<int64_t> extent = std::nullopt) const;
  // `tensor` in place as arg `arg`, which the compiler lays out otherwise than torch, takes it: a
  // tensor over a whole allocation of the arg's type; none for any other.
  rt::Binding typed(const at::Tensor& tensor, size_t arg, c10::DeviceIndex device_index) const;
  // The host tensor of state arg `arg`, encoded from the state now.
  std::shared_ptr<rt::HostTensor> encoded(size_t arg) const;
  // A host tensor of arg `arg` encoded from the value of `tensor`, or none if it is not the arg's;
  // with `extent`, from the value padded with zeros to that leading extent.
  std::shared_ptr<rt::HostTensor> encoded(
      const at::Tensor& tensor,
      size_t arg,
      std::optional<int64_t> extent = std::nullopt) const;
  // What tells the tensor of state arg `arg` on `device` apart from others.
  StateKey state_key(size_t arg, const rt::Device& device) const;
  // `args` with the state args that share pools of pages with them, which a
  // device holds in one buffer with them.
  std::vector<size_t> with_page_sharers(std::vector<size_t> args) const;
  // Binds state `args` of each of `slots` to tensors on its device that hold the state now:
  // those another function holding the arg alike has made, or new ones, each encoded once.
  void bind_state(const std::vector<Slot*>& slots, const std::vector<size_t>& args) const;
  Slot& slot(c10::DeviceIndex device_index);
  // The leading extent of the results, when a call can pass the inputs `rows` names with fewer
  // rows: each result is a tensor the function makes on the device, all of one leading extent, and
  // each arg of those inputs is one the run only reads, along that extent, and no host work of a
  // call reads those inputs.
  std::optional<int64_t> row_extent(const std::vector<size_t>& rows) const;

  std::shared_ptr<rt::Function> fn_;
  size_t num_inputs_ = 0;
  // The inputs `rows` names, when the function is row-wise, and the leading extent of every result
  // and of those inputs' args.
  std::vector<size_t> row_inputs_;
  int64_t row_extent_ = 0;
  std::vector<Binding> bindings_;
  std::vector<size_t> updated_inputs_;
  std::vector<Result> results_;
  std::vector<Output> outputs_;
  std::vector<size_t> state_args_;
  // The args each call makes on the host before its run, and those the run writes for the host
  // to read after it; the transforms that make the args before the run and the results the run
  // does not write after it, with the input each value they read is, in order.
  std::vector<size_t> made_args_;
  std::vector<size_t> left_args_;
  std::shared_ptr<const rt::Maker> before_, after_;
  std::vector<size_t> before_inputs_, after_inputs_;
  // By arg made from state alone that the host work of calls reads, its host tensor.
  std::map<size_t, std::shared_ptr<rt::HostTensor>> held_state_;
  // Guards the state and the slots.
  std::mutex slots_mutex_;
  std::map<std::string, at::Tensor> state_values_;
  std::map<c10::DeviceIndex, std::unique_ptr<Slot>> slots_;
  std::shared_ptr<rt::Communicator> communicator_;
  std::atomic<uint64_t> last_run_{0};
  bool runs_on_device_ = false;
};

/**
 * @brief A compiled function whose args tensors are made in the types of (see
 * c10/rbln/RBLNTyped.h), such as the KV caches the graphs of a model write in place.
 */
class TypedFunction {
 public:
  explicit TypedFunction(const std::string& bytes);

  /**
   * @brief A contiguous tensor of `shape` on device `device_index` in the type of arg `arg`, all
   * zero when `zero`; a tensor as torch holds it when the arg holds its value so.
   */
  at::Tensor empty(const std::string& arg, const std::vector<int64_t>& shape, c10::DeviceIndex device_index, bool zero)
      const;

 private:
  std::shared_ptr<rt::Function> fn_;
};

void init_op_function_bindings(pybind11::module& module);

} // namespace torch_rbln
