/**
 * @file ProcessGroupRBLN.cpp
 * @brief Implementation of ProcessGroupRBLN for RBLN distributed computing
 *
 * The collectives run through a runtime Communicator on the process group's device, queued on
 * the current stream of the device after the work before them; a Work is done once they are.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include <c10/core/StreamGuard.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNRuntime.h>
#include <c10/util/error.h>

#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <rbln/runtime/communicator.h>
#include <rbln/runtime/stream.h>
#include <torch/torch.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.hpp>
// TEMPORARY: remove together with RdmaIpAutoDiscovery.{hpp,cpp} once
// librbln-ccl performs RoCE GID auto-discovery internally.
#include <torch_rbln/csrc/distributed/c10d/rbln/RdmaIpAutoDiscovery.hpp>

namespace c10d {

namespace {

namespace rt = ::rbln::runtime;
using Place = rt::Communicator::Place;

// What the collectives library takes of one call (RCCL_DATA_ALIGNSIZE_FOR_CS and the size of
// the command stream it builds): byte counts on these alignments, and at most these sizes.
constexpr size_t RCCL_ALLGATHER_ALIGNMENT = 128;
constexpr size_t RCCL_REDUCE_ALIGNMENT = 512;
constexpr size_t RCCL_ALLREDUCE_MAX_BYTES_PER_RANK = size_t{1} << 20;
constexpr size_t RCCL_REDUCE_SCATTER_MAX_BYTES_PER_WORLD = size_t{32} << 20;
constexpr size_t RCCL_MAX_BYTES_PER_REDUCE_SCATTER_OP = size_t{2} << 20;
constexpr size_t RCCL_MAX_COMMAND_BUFFER_SUB_COMMANDS_COUNT = 15;
constexpr size_t RCCL_ALLGATHER_MAX_OUTPUT_BYTES = size_t{64} << 20;

/// Collectives that only move bytes move them as int8.
constexpr const char* kBytes = "int8";
constexpr const char* kSum = "sum";

size_t roundUp(size_t n, size_t unit) {
  return (n + unit - 1) / unit * unit;
}

/**
 * @brief The dtype the collectives library reduces tensors of `type` by `reduceOp` in on the
 * device, if it reduces them there: it sums bfloat16.
 */
std::optional<std::string> deviceReduceDtype(at::ScalarType type, const ReduceOp& reduceOp) {
  if (type == at::kBFloat16 && reduceOp == ReduceOp::SUM) {
    return "bfloat16";
  }
  return std::nullopt;
}

/**
 * @brief The device memory a contiguous tensor holds its elements in.
 */
Place placeOf(const at::Tensor& tensor) {
  auto location = c10::rbln::caching::locate(tensor.const_data_ptr());
  RBLN_CHECK(
      location.available >= tensor.nbytes(),
      "a tensor of {} bytes reaches past its allocation of {}",
      tensor.nbytes(),
      location.available);
  return {location.buffer, location.offset};
}

/**
 * @brief Whether `tensors`, each contiguous and of `nbytes`, lie side by side in one buffer in
 * their order.
 */
bool sideBySide(const std::vector<at::Tensor>& tensors, size_t nbytes) {
  if (tensors.empty() || !tensors[0].is_contiguous()) {
    return false;
  }
  auto first = placeOf(tensors[0]);
  for (size_t k = 1; k < tensors.size(); ++k) {
    if (!tensors[k].is_contiguous()) {
      return false;
    }
    auto place = placeOf(tensors[k]);
    if (place.buffer != first.buffer || place.offset != first.offset + k * nbytes) {
      return false;
    }
  }
  return true;
}

/**
 * @brief The elements of `tensor` as a flat tensor over its memory, or over a copy when it is
 * not contiguous; `writeBack` copies such a copy into `tensor`.
 */
struct Flat {
  explicit Flat(const at::Tensor& tensor)
      : tensor(tensor), flat(tensor.is_contiguous() ? tensor.view(-1) : tensor.contiguous().view(-1)) {}
  void writeBack() {
    if (!tensor.is_contiguous()) {
      tensor.copy_(flat.view(tensor.sizes()));
    }
  }
  at::Tensor tensor;
  at::Tensor flat;
};

c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.empty()) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}

// ============================================================================
// RBLNWork Base Class
// ============================================================================

/**
 * @brief Base class for the work of a collective
 *
 * `run` queues the collective on `stream`, the stream current on the process group's device
 * when the work was made; `execute` then waits for the stream and completes the work.
 */
class RBLNWork : public Work {
 public:
  explicit RBLNWork(
      std::vector<std::vector<at::Tensor>> outputTensors,
      OpType opType,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors = std::nullopt);

  ~RBLNWork() override = default;

  static void execute(const c10::intrusive_ptr<RBLNWork>& work);

  virtual void run() = 0;

  std::vector<at::Tensor> result() override;

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  uint64_t getSequencenumber() const override;

  inline at::ThreadLocalState getTLS() const {
    return tls_;
  }

 protected:
  friend class ProcessGroupRBLN;

  rt::Stream* stream() const {
    return runtime_.get();
  }

 private:
  void finishWorkRBLN();
  void finishWorkRBLNError(const std::exception_ptr& eptr);
  inline void recordRBLNWorkProfilingInfo(
      const char* profilingTitle,
      const std::optional<std::vector<at::Tensor>>& inputTensors);

  const std::vector<std::vector<at::Tensor>> outputTensors_;
  c10::intrusive_ptr<at::ivalue::Future> future_;
  std::function<void()> recordFunctionBeforeCallback_;
  uint64_t seq_;
  std::optional<c10::Stream> stream_;
  std::shared_ptr<rt::Stream> runtime_;
  at::ThreadLocalState tls_;
};

RBLNWork::RBLNWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    OpType opType,
    uint64_t seq,
    std::optional<c10::Stream> stream,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)),
      recordFunctionBeforeCallback_(nullptr),
      seq_(seq),
      stream_(stream),
      runtime_(stream ? c10::rbln::runtime_stream(*stream) : nullptr) {
  recordRBLNWorkProfilingInfo(profilingTitle, inputTensors);
}

void RBLNWork::recordRBLNWorkProfilingInfo(
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  if (profilingTitle != nullptr) {
    auto recordingFunction = std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      recordingFunction->_setAsync();
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(profilingTitle, c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
      std::function<void()> end_handler = [recordingFunction]() { recordingFunction->end(); };
      recordFunctionBeforeCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

// static
void RBLNWork::execute(const c10::intrusive_ptr<RBLNWork>& work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    std::optional<c10::StreamGuard> guard;
    if (work->stream_) {
      guard.emplace(*work->stream_);
    }
    work->run();
    if (work->runtime_) {
      work->runtime_->synchronize();
    }
  } catch (...) {
    work->finishWorkRBLNError(std::current_exception());
    return;
  }
  work->finishWorkRBLN();
}

std::vector<at::Tensor> RBLNWork::result() {
  RBLN_CHECK(isCompleted(), "Work needs to be completed before calling result(). Should call wait() before result().");
  RBLN_CHECK(outputTensors_.size() <= 1, "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.empty() ? std::vector<at::Tensor>() : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future> RBLNWork::getFuture() {
  return future_;
}

uint64_t RBLNWork::getSequencenumber() const {
  return seq_;
}

void RBLNWork::finishWorkRBLN() {
  if (future_) {
    returnFutureWithOutput(future_, outputTensors_);
  }
  finish();
}

void RBLNWork::finishWorkRBLNError(const std::exception_ptr& eptr) {
  if (future_) {
    future_->setError(eptr);
  }
  finish(eptr);
}

// ============================================================================
// Specific RBLNWork Implementations
// ============================================================================

/**
 * @brief RBLNWork that sends a tensor to another rank
 */
class SendRBLNWork : public RBLNWork {
 public:
  SendRBLNWork(
      at::Tensor& tensor,
      int dstRank,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{},
            OpType::SEND,
            seq,
            stream,
            "rbln:send",
            std::optional<std::vector<at::Tensor>>({tensor})),
        tensor_(tensor),
        dstRank_(dstRank),
        comm_(std::move(comm)) {}

  void run() override {
    RECORD_FUNCTION("rbln::send", std::vector<c10::IValue>({tensor_}));
    RBLN_CHECK(comm_, "send needs a communicator");
    Flat src(tensor_);
    if (src.flat.nbytes() > 0) {
      comm_->send(stream(), placeOf(src.flat), src.flat.nbytes(), kBytes, dstRank_);
    }
  }

 private:
  at::Tensor tensor_;
  int dstRank_;
  std::shared_ptr<rt::Communicator> comm_;
};

/**
 * @brief RBLNWork that receives a tensor from another rank
 */
class RecvRBLNWork : public RBLNWork {
 public:
  RecvRBLNWork(
      at::Tensor& tensor,
      int srcRank,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{},
            OpType::RECV,
            seq,
            stream,
            "rbln:recv",
            std::optional<std::vector<at::Tensor>>({tensor})),
        tensor_(tensor),
        srcRank_(srcRank),
        comm_(std::move(comm)) {}

  void run() override {
    RECORD_FUNCTION("rbln::recv", std::vector<c10::IValue>({tensor_}));
    RBLN_CHECK(comm_, "recv needs a communicator");
    Flat dst(tensor_);
    if (dst.flat.nbytes() > 0) {
      comm_->recv(stream(), placeOf(dst.flat), dst.flat.nbytes(), kBytes, srcRank_);
    }
    dst.writeBack();
  }

 private:
  at::Tensor tensor_;
  int srcRank_;
  std::shared_ptr<rt::Communicator> comm_;
};

/**
 * @brief RBLNWork that broadcasts a tensor from the root rank to every rank
 */
class BroadcastRBLNWork : public RBLNWork {
 public:
  BroadcastRBLNWork(
      std::vector<at::Tensor>& inputs,
      int rootRank,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int rank,
      int worldSize)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{inputs},
            OpType::BROADCAST,
            seq,
            stream,
            "rbln:broadcast",
            std::optional<std::vector<at::Tensor>>(inputs)),
        inputs_(inputs),
        rootRank_(rootRank),
        comm_(std::move(comm)),
        rank_(rank),
        worldSize_(worldSize) {}

  void run() override {
    RBLN_CHECK(inputs_.size() == 1, "BroadcastRBLNWork expects exactly one input tensor, got {}", inputs_.size());
    RECORD_FUNCTION("rbln::broadcast", std::vector<c10::IValue>{inputs_[0]});
    if (worldSize_ == 1 || inputs_[0].nbytes() == 0) {
      return;
    }
    RBLN_CHECK(comm_, "broadcast needs a communicator");
    Flat data(inputs_[0]);
    auto place = placeOf(data.flat);
    comm_->broadcast(stream(), rank_ == rootRank_ ? place : Place{}, place, data.flat.nbytes(), kBytes, rootRank_);
    data.writeBack();
  }

 private:
  std::vector<at::Tensor> inputs_;
  int rootRank_;
  std::shared_ptr<rt::Communicator> comm_;
  int rank_;
  int worldSize_;
};

/**
 * @brief RBLNWork that gathers a tensor of every rank into each rank's list of them
 *
 * A call moves a whole multiple of the alignment and a bounded output, so a large or unaligned
 * input goes in parts, each padded into staging; outputs side by side take a single call
 * directly.
 */
class AllgatherRBLNWork : public RBLNWork {
 public:
  AllgatherRBLNWork(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int worldSize)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::ALLGATHER,
            seq,
            stream,
            "rbln:all_gather",
            std::optional<std::vector<at::Tensor>>(inputs)),
        outputs_(outputs),
        inputs_(inputs),
        comm_(std::move(comm)),
        worldSize_(worldSize) {}

  void run() override {
    RECORD_FUNCTION("rbln::all_gather", std::vector<c10::IValue>(inputs_.begin(), inputs_.end()));
    for (const auto i : c10::irange(inputs_.size())) {
      auto& output = outputs_[i];
      RBLN_CHECK(
          static_cast<int>(output.size()) == worldSize_,
          "output tensor count must equal worldSize, got {} expected {}",
          output.size(),
          worldSize_);
      if (worldSize_ == 1) {
        output[0].copy_(inputs_[i]);
        continue;
      }
      RBLN_CHECK(comm_, "all_gather needs a communicator");
      gather(Flat(inputs_[i]).flat, output);
    }
  }

 private:
  void gather(const at::Tensor& input, std::vector<at::Tensor>& output) {
    const size_t numel = input.numel();
    const size_t elem = input.element_size();
    const size_t nbytes = numel * elem;
    if (nbytes == 0) {
      return;
    }
    const size_t world = worldSize_;
    if (nbytes % RCCL_ALLGATHER_ALIGNMENT == 0 && nbytes * world <= RCCL_ALLGATHER_MAX_OUTPUT_BYTES &&
        sideBySide(output, nbytes)) {
      comm_->allGather(stream(), placeOf(input), placeOf(output[0]), nbytes, kBytes);
      return;
    }
    std::vector<Flat> flats;
    flats.reserve(world);
    for (const auto& t : output) {
      flats.emplace_back(t);
    }
    const size_t align = RCCL_ALLGATHER_ALIGNMENT / elem;
    const size_t most = std::max(align, RCCL_ALLGATHER_MAX_OUTPUT_BYTES / elem / world / align * align);
    const size_t part = std::min(most, roundUp(numel, align));
    auto staged = at::empty({static_cast<int64_t>(part)}, input.options());
    auto gathered = at::empty({static_cast<int64_t>(world), static_cast<int64_t>(part)}, input.options());
    for (size_t at = 0; at < numel; at += part) {
      const auto n = static_cast<int64_t>(std::min(part, numel - at));
      const auto from = static_cast<int64_t>(at);
      staged.slice(0, 0, n).copy_(input.slice(0, from, from + n));
      comm_->allGather(stream(), placeOf(staged), placeOf(gathered), part * elem, kBytes);
      for (const auto j : c10::irange(world)) {
        flats[j].flat.slice(0, from, from + n).copy_(gathered[static_cast<int64_t>(j)].slice(0, 0, n));
      }
    }
    for (auto& flat : flats) {
      flat.writeBack();
    }
  }

  std::vector<std::vector<at::Tensor>> outputs_;
  std::vector<at::Tensor> inputs_;
  std::shared_ptr<rt::Communicator> comm_;
  int worldSize_;
};

/**
 * @brief RBLNWork that reduces tensors across the ranks in place
 *
 * The device reduces what the collectives library reduces, a bounded and aligned part at a time;
 * the rest is reduced on the host by the Gloo backend.
 */
class AllreduceRBLNWork : public RBLNWork {
 public:
  AllreduceRBLNWork(
      std::vector<at::Tensor>& inputs,
      const ReduceOp& reduceOp,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int worldSize,
      c10::intrusive_ptr<Backend> glooBackend)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{inputs},
            OpType::ALLREDUCE,
            seq,
            stream,
            "rbln:all_reduce",
            std::optional<std::vector<at::Tensor>>(inputs)),
        inputs_(inputs),
        reduceOp_(reduceOp),
        comm_(std::move(comm)),
        worldSize_(worldSize),
        glooBackend_(std::move(glooBackend)) {}

  void run() override {
    RECORD_FUNCTION("rbln::all_reduce", std::vector<c10::IValue>(inputs_.begin(), inputs_.end()));
    if (worldSize_ == 1) {
      return;
    }
    auto dtype = deviceReduceDtype(inputs_[0].scalar_type(), reduceOp_);
    if (!dtype) {
      onHost();
      return;
    }
    RBLN_CHECK(comm_, "all_reduce needs a communicator");
    for (auto& input : inputs_) {
      Flat data(input);
      const size_t numel = data.flat.numel();
      const size_t elem = data.flat.element_size();
      const size_t align = RCCL_REDUCE_ALIGNMENT / elem;
      const size_t most = RCCL_ALLREDUCE_MAX_BYTES_PER_RANK * worldSize_ / elem / align * align;
      at::Tensor staged;
      for (size_t at = 0; at < numel;) {
        const size_t n = std::min(most, numel - at);
        const auto from = static_cast<int64_t>(at);
        auto part = data.flat.slice(0, from, from + static_cast<int64_t>(n));
        if (n % align == 0) {
          auto place = placeOf(part);
          comm_->allReduce(stream(), place, place, n, *dtype, kSum);
        } else {
          if (!staged.defined()) {
            staged = at::empty({static_cast<int64_t>(roundUp(n, align))}, data.flat.options());
          }
          staged.slice(0, 0, static_cast<int64_t>(n)).copy_(part);
          auto place = placeOf(staged);
          comm_->allReduce(stream(), place, place, staged.numel(), *dtype, kSum);
          part.copy_(staged.slice(0, 0, static_cast<int64_t>(n)));
        }
        at += n;
      }
      data.writeBack();
    }
  }

 private:
  void onHost() {
    RBLN_CHECK(
        glooBackend_,
        "all_reduce of {} other than a bfloat16 sum runs on the host and needs a Gloo backend; pass gloo_backend "
        "when creating ProcessGroupRBLN",
        c10::toString(inputs_[0].scalar_type()));
    std::vector<at::Tensor> cpu_tensors;
    cpu_tensors.reserve(inputs_.size());
    for (auto& input : inputs_) {
      cpu_tensors.emplace_back(input.to("cpu"));
    }
    AllreduceOptions opts;
    opts.reduceOp = reduceOp_;
    glooBackend_->allreduce(cpu_tensors, opts)->wait();
    for (const auto i : c10::irange(inputs_.size())) {
      inputs_[i].copy_(cpu_tensors[i]);
    }
  }

  std::vector<at::Tensor> inputs_;
  ReduceOp reduceOp_;
  std::shared_ptr<rt::Communicator> comm_;
  int worldSize_;
  c10::intrusive_ptr<Backend> glooBackend_;
};

/**
 * @brief RBLNWork that reduces each rank's list of tensors across the ranks, this rank taking
 * the reduction of its own entry of the lists
 *
 * A call reduces a bounded number of bytes whose part for each rank is aligned, so a large or
 * unaligned list goes in parts staged side by side; inputs side by side take a single call
 * directly when they fit.
 */
class ReduceScatterRBLNWork : public RBLNWork {
 public:
  ReduceScatterRBLNWork(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceOp& reduceOp,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int worldSize,
      c10::intrusive_ptr<Backend> glooBackend)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::REDUCE_SCATTER,
            seq,
            stream,
            "rbln:reduce_scatter",
            std::optional<std::vector<at::Tensor>>(outputs)),
        outputs_(outputs),
        inputs_(inputs),
        reduceOp_(reduceOp),
        comm_(std::move(comm)),
        worldSize_(worldSize),
        glooBackend_(std::move(glooBackend)) {}

  void run() override {
    RECORD_FUNCTION("rbln::reduce_scatter", std::vector<c10::IValue>(outputs_.begin(), outputs_.end()));
    if (worldSize_ == 1) {
      outputs_[0].copy_(inputs_[0][0]);
      return;
    }
    for (const auto i : c10::irange(outputs_.size())) {
      RBLN_CHECK(static_cast<int>(inputs_[i].size()) == worldSize_, "inputs_[i] must contain worldSize_ tensors");
      for (const auto& input : inputs_[i]) {
        RBLN_CHECK(input.sizes() == outputs_[i].sizes(), "input shape should be the same as output shape");
      }
      auto dtype = deviceReduceDtype(outputs_[i].scalar_type(), reduceOp_);
      if (!dtype) {
        onHost(i);
        continue;
      }
      RBLN_CHECK(comm_, "reduce_scatter needs a communicator");
      scatter(outputs_[i], inputs_[i], *dtype);
    }
  }

 private:
  void scatter(at::Tensor& output, const std::vector<at::Tensor>& inputs, const std::string& dtype) {
    const size_t world = worldSize_;
    const size_t numel = output.numel();
    const size_t elem = output.element_size();
    const size_t align = RCCL_REDUCE_ALIGNMENT / elem;
    // The per-rank size a call takes: its send buffer bounded twice over by the world, and its
    // parts within the command buffer's sub-commands.
    const size_t by_buffer = RCCL_REDUCE_SCATTER_MAX_BYTES_PER_WORLD / world / world / world / elem;
    const size_t by_commands = RCCL_MAX_COMMAND_BUFFER_SUB_COMMANDS_COUNT / (world - 1) *
        RCCL_MAX_BYTES_PER_REDUCE_SCATTER_OP / world / world / world / elem;
    const size_t most = std::max(align, std::min(by_buffer, by_commands) / align * align);
    if (numel == 0) {
      return;
    }
    if (numel % align == 0 && numel <= most && output.is_contiguous() && sideBySide(inputs, numel * elem)) {
      comm_->reduceScatter(stream(), placeOf(inputs[0]), placeOf(output), numel, dtype, kSum);
      return;
    }
    Flat out(output);
    std::vector<at::Tensor> ins;
    ins.reserve(world);
    for (const auto& input : inputs) {
      ins.push_back(Flat(input).flat);
    }
    const size_t part = std::min(most, roundUp(numel, align));
    auto staged = at::empty({static_cast<int64_t>(world), static_cast<int64_t>(part)}, output.options());
    auto reduced = at::empty({static_cast<int64_t>(part)}, output.options());
    for (size_t at = 0; at < numel; at += part) {
      const auto n = static_cast<int64_t>(std::min(part, numel - at));
      const auto from = static_cast<int64_t>(at);
      for (const auto k : c10::irange(world)) {
        staged[static_cast<int64_t>(k)].slice(0, 0, n).copy_(ins[k].slice(0, from, from + n));
      }
      comm_->reduceScatter(stream(), placeOf(staged), placeOf(reduced), part, dtype, kSum);
      out.flat.slice(0, from, from + n).copy_(reduced.slice(0, 0, n));
    }
    out.writeBack();
  }

  void onHost(size_t i) {
    RBLN_CHECK(
        glooBackend_,
        "reduce_scatter of {} other than a bfloat16 sum runs on the host and needs a Gloo backend; pass "
        "gloo_backend when creating ProcessGroupRBLN",
        c10::toString(outputs_[i].scalar_type()));
    std::vector<at::Tensor> cpu_inputs;
    cpu_inputs.reserve(inputs_[i].size());
    for (auto& tensor : inputs_[i]) {
      cpu_inputs.emplace_back(tensor.to("cpu"));
    }
    std::vector<std::vector<at::Tensor>> cpu_inputs_wrapped = {cpu_inputs};
    std::vector<at::Tensor> cpu_outputs = {outputs_[i].to("cpu")};
    ReduceScatterOptions opts;
    opts.reduceOp = reduceOp_;
    glooBackend_->reduce_scatter(cpu_outputs, cpu_inputs_wrapped, opts)->wait();
    outputs_[i].copy_(cpu_outputs[0]);
  }

  std::vector<at::Tensor> outputs_;
  std::vector<std::vector<at::Tensor>> inputs_;
  ReduceOp reduceOp_;
  std::shared_ptr<rt::Communicator> comm_;
  int worldSize_;
  c10::intrusive_ptr<Backend> glooBackend_;
};

/**
 * @brief RBLNWork that hands each rank its entry of the root rank's list of tensors
 */
class ScatterRBLNWork : public RBLNWork {
 public:
  ScatterRBLNWork(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int rank,
      int worldSize)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::SCATTER,
            seq,
            stream,
            "rbln:scatter",
            std::optional<std::vector<at::Tensor>>(outputs)),
        outputs_(outputs),
        inputs_(inputs),
        root_(root),
        comm_(std::move(comm)),
        rank_(rank),
        worldSize_(worldSize) {}

  void run() override {
    RECORD_FUNCTION("rbln::scatter", std::vector<c10::IValue>(outputs_.begin(), outputs_.end()));
    for (const auto i : c10::irange(outputs_.size())) {
      if (worldSize_ == 1) {
        outputs_[i].copy_(inputs_[i][0]);
        continue;
      }
      RBLN_CHECK(comm_, "scatter needs a communicator");
      Flat out(outputs_[i]);
      const size_t nbytes = out.flat.nbytes();
      if (nbytes == 0) {
        continue;
      }
      Place src;
      at::Tensor staged;
      if (rank_ == root_) {
        const auto& inputs = inputs_[i];
        RBLN_CHECK(static_cast<int>(inputs.size()) == worldSize_, "the root scatters worldSize tensors");
        if (sideBySide(inputs, nbytes)) {
          src = placeOf(inputs[0]);
        } else {
          staged = at::empty({static_cast<int64_t>(worldSize_), out.flat.numel()}, out.flat.options());
          for (const auto k : c10::irange(inputs.size())) {
            staged[static_cast<int64_t>(k)].copy_(inputs[k].reshape(-1));
          }
          src = placeOf(staged);
        }
      }
      comm_->scatter(stream(), src, placeOf(out.flat), nbytes, kBytes, root_);
      out.writeBack();
    }
  }

 private:
  std::vector<at::Tensor> outputs_;
  std::vector<std::vector<at::Tensor>> inputs_;
  int root_;
  std::shared_ptr<rt::Communicator> comm_;
  int rank_;
  int worldSize_;
};

/**
 * @brief RBLNWork that waits for the work before it and meets every rank in a broadcast
 */
class BarrierRBLNWork : public RBLNWork {
 public:
  BarrierRBLNWork(
      std::vector<c10::weak_intrusive_ptr<Work>> priorWork,
      uint64_t seq,
      std::optional<c10::Stream> stream,
      std::shared_ptr<rt::Communicator> comm,
      int rank,
      int device_id,
      int size)
      : RBLNWork(std::vector<std::vector<at::Tensor>>{}, OpType::BARRIER, seq, stream, "rbln:barrier"),
        priorWork_(std::move(priorWork)),
        comm_(std::move(comm)),
        rank_(rank),
        device_id_(device_id),
        size_(size) {}

  void run() override {
    RECORD_FUNCTION("rbln::barrier", std::vector<c10::IValue>());
    for (auto& weakWork : priorWork_) {
      if (auto work = weakWork.lock()) {
        work->wait();
      }
    }
    if (size_ == 1) {
      return;
    }
    RBLN_CHECK(comm_, "barrier needs a communicator");
    const auto device = c10::Device(c10::kPrivateUse1, static_cast<c10::DeviceIndex>(device_id_));
    auto token = at::zeros(
        {static_cast<int64_t>(RCCL_ALLGATHER_ALIGNMENT)}, c10::TensorOptions().device(device).dtype(at::kByte));
    auto place = placeOf(token);
    comm_->broadcast(stream(), rank_ == 0 ? place : Place{}, place, token.nbytes(), kBytes, 0);
  }

 private:
  std::vector<c10::weak_intrusive_ptr<Work>> priorWork_;
  std::shared_ptr<rt::Communicator> comm_;
  int rank_;
  int device_id_;
  int size_;
};

} // namespace

// ============================================================================
// ProcessGroupRBLN Implementation
// ============================================================================

namespace {

// The process's default group while one lives: the device each of its global ranks runs on, which a
// sub group takes.
struct DefaultGroup {
  std::mutex mutex;
  bool live = false;
  std::unordered_map<int, int> device_of_rank;
};

DefaultGroup& default_group() {
  static DefaultGroup group;
  return group;
}

} // namespace

ProcessGroupRBLN::ProcessGroupRBLN(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    int group_id,
    const std::vector<int>& global_ranks_in_group,
    const c10::intrusive_ptr<Options>& options,
    c10::intrusive_ptr<Backend> glooBackend)
    : Backend(rank, size),
      store_(store),
      group_id_(group_id),
      glooBackend_(std::move(glooBackend)),
      backendName_(RBLN_BACKEND_NAME),
      global_ranks_in_group_(global_ranks_in_group) {
  // TEMPORARY: auto-fill RBLN_RDMA_IP before the communicator is made, which is when
  // librbln-ccl reads the env. Remove this call once librbln-ccl performs the discovery.
  ::torch_rbln::detail::MaybeAutoDiscoverRbnRdmaIp();

  c10::rbln::get_device_count();
  c10::rbln::commit_device_mapping();

  if (!global_ranks_in_group_.empty()) {
    RBLN_CHECK(rank_ < global_ranks_in_group_.size(), "rank_ should be less than global_ranks_in_group_'s length");
  }
  global_rank_ = global_ranks_in_group_.empty() ? rank_ : global_ranks_in_group_[rank_];

  device_id_ = -1;
  {
    auto& group = default_group();
    std::lock_guard<std::mutex> lock(group.mutex);
    if (global_ranks_in_group_.empty()) {
      RBLN_CHECK(!group.live, "ProcessGroupRBLN: a default group lives already; destroy it before making another");
      device_id_ = static_cast<int>(c10::rbln::get_device_index()); // NOLINT(bugprone-signed-char-misuse)
      group.device_of_rank[global_rank_] = device_id_;
      group.live = true;
    } else {
      RBLN_CHECK(group.live, "Default group must be initialized before creating sub groups");
      auto it = group.device_of_rank.find(global_rank_);
      RBLN_CHECK(it != group.device_of_rank.end(), "Global rank {} not found in default group mapping", global_rank_);
      device_id_ = it->second;
    }
  }
  RBLN_LOG_DEBUG("device_id: {}", device_id_);

  connect();

  // Last, as unwinding past a joinable std::thread member terminates.
  const char* async_env = std::getenv("TORCH_RBLN_C10D_ASYNC");
  if (async_env != nullptr && std::string(async_env) == "1") {
    sync_mode_ = false;
  }

  if (!sync_mode_) {
    const int numThreads = DEFAULT_NUM_WORKERS;
    workInProgress_.resize(numThreads);
    threads_.resize(numThreads);
    for (const auto i : c10::irange(threads_.size())) {
      threads_[i] = std::thread(&ProcessGroupRBLN::runLoop, this, i);
    }
  }
}

ProcessGroupRBLN::~ProcessGroupRBLN() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });
  stop_ = true;
  lock.unlock();
  workProduceCV_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
  if (global_ranks_in_group_.empty()) {
    auto& group = default_group();
    std::lock_guard<std::mutex> group_lock(group.mutex);
    group.live = false;
    group.device_of_rank.clear();
  }
}

uint32_t ProcessGroupRBLN::nextTag() noexcept {
  return tag_++;
}

uint64_t ProcessGroupRBLN::nextSeq() noexcept {
  return seq_++;
}

c10::Stream ProcessGroupRBLN::currentStream() const {
  return c10::rbln::get_current_stream(static_cast<c10::DeviceIndex>(device_id_));
}

namespace {

// The communicator of the process's default group. RCCL makes one group id a process, so a sub
// group splits this one.
std::weak_ptr<::rbln::runtime::Communicator>& default_communicator() {
  static std::weak_ptr<::rbln::runtime::Communicator> communicator;
  return communicator;
}

} // namespace

void ProcessGroupRBLN::connect() {
  if (size_ == 1) {
    return;
  }
  // Without an NPU a group can still be built for graph capture; its collectives run only on NPUs.
  if (c10::rbln::get_device_count() == 0) {
    RBLN_LOG_INFO("No NPU detected; making no communicator (rank={} size={})", rank_, size_);
    return;
  }
  auto device = c10::rbln::runtime_device(static_cast<c10::DeviceIndex>(device_id_));
  if (device->dummy()) {
    RBLN_LOG_INFO("A dummy device has no communicator (rank={} size={})", rank_, size_);
    return;
  }
  if (!global_ranks_in_group_.empty()) {
    auto root = default_communicator().lock();
    RBLN_CHECK(
        root != nullptr,
        "ProcessGroupRBLN: a sub group splits the default group's communicator, and the default group has none");
    RBLN_LOG_INFO("Communicator split rank={} size={} group_id={}", rank_, size_, group_id_);
    comm_ = root->split(global_ranks_in_group_, group_id_);
    return;
  }
  RBLN_CHECK(store_ != nullptr, "ProcessGroupRBLN: the ranks share a group id through the store, and there is none");
  const std::string key = "rbln_rccl_uid_" + std::to_string(group_id_);
  std::string id;
  if (rank_ == 0) {
    id = ::rbln::runtime::Communicator::uniqueId(device);
    store_->set(key, std::vector<uint8_t>(id.begin(), id.end()));
  } else {
    auto bytes = store_->get(key);
    id.assign(bytes.begin(), bytes.end());
  }
  RBLN_LOG_INFO("Communicator init rank={} size={} device_id={}", rank_, size_, device_id_);
  comm_ = std::make_shared<::rbln::runtime::Communicator>(rank_, size_, device, id);
  default_communicator() = comm_;
}

void ProcessGroupRBLN::enqueue(c10::intrusive_ptr<Work> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  workQueue_.push_back(std::move(work));
  lock.unlock();
  workProduceCV_.notify_one();
}

void ProcessGroupRBLN::enqueueOrExecute(c10::intrusive_ptr<Work> work) {
  if (sync_mode_) {
    auto rblnWork = c10::intrusive_ptr<RBLNWork>::reclaim(static_cast<RBLNWork*>(work.get()));
    work.release();
    RBLNWork::execute(rblnWork);
  } else {
    enqueue(work);
  }
}

void ProcessGroupRBLN::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }
    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();
    workConsumeCV_.notify_one();

    auto rblnWork = c10::intrusive_ptr<RBLNWork>::reclaim(static_cast<RBLNWork*>(work.get()));
    work.release();
    RBLNWork::execute(rblnWork);

    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

// ============================================================================
// Collective Communication Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::broadcast(std::vector<at::Tensor>& inputs, const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::broadcast: {}", msg);
  };

  RBLN_CHECK(inputs.size() == 1, "ProcessGroupRBLN::broadcast: Expecting one tensor only but got {}", inputs.size());

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::broadcast: CPU tensors require gloo_backend");
    return glooBackend_->broadcast(inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<BroadcastRBLNWork>(
      inputs, opts.rootRank, nextSeq(), currentStream(), comm_, rank_, getSize());
  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allgather: {}", msg);
  };

  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }
  if (inputs.size() != outputs.size()) {
    invalidArgument("requires input/output tensor lists to have the same length");
  }
  for (const auto i : c10::irange(outputs.size())) {
    const auto expected = getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) + " (expected length " + std::to_string(expected) +
          ", got " + std::to_string(actual) + ")");
    }
  }

  assertDense(invalidArgument, inputs);

  const auto& options = inputs[0].options();
  const auto& sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto& output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allgather: CPU tensors require gloo_backend");
    return glooBackend_->allgather(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<AllgatherRBLNWork>(outputs, inputs, nextSeq(), currentStream(), comm_, getSize());
  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  auto tensor_list = at::chunk(output_tensor, getSize(), 0);
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  std::vector<at::Tensor> inputs = {input_tensor};
  return allgather(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allgather_into_tensor_coalesced: {}", msg);
  };

  RBLN_CHECK(outputTensors.size() == inputTensors.size());
  if (inputTensors.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  const auto& device = inputTensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allgather_into_tensor_coalesced: CPU tensors require gloo_backend");
    return glooBackend_->allgather_into_tensor_coalesced(outputTensors, inputTensors, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  const auto worldSize = getSize();
  std::vector<std::vector<at::Tensor>> output_lists(outputTensors.size());
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    output_lists[i] = outputTensors[i].chunk(worldSize);
  }
  return allgather(output_lists, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allreduce(std::vector<at::Tensor>& inputs, const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allreduce: {}", msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allreduce: CPU tensors require gloo_backend");
    return glooBackend_->allreduce(inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<AllreduceRBLNWork>(
      inputs, opts.reduceOp, nextSeq(), currentStream(), comm_, getSize(), glooBackend_);
  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::reduce_scatter: {}", msg);
  };

  const auto worldSize = getSize();

  RBLN_CHECK(outputs.size() == 1, "reduce_scatter only supports 1 output");
  RBLN_CHECK(outputs.size() == inputs.size(), "requires input/output tensor lists to have the same length");
  RBLN_CHECK(static_cast<int>(inputs[0].size()) == worldSize, "invalid input tensor list size, must be world size");

  for (const auto i : c10::irange(inputs[0].size())) {
    RBLN_CHECK(outputs[0].dtype() == inputs[0][i].dtype());
    RBLN_CHECK(outputs[0].sizes().vec() == inputs[0][i].sizes().vec());
  }

  const auto& device = outputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::reduce_scatter: CPU tensors require gloo_backend");
    return glooBackend_->reduce_scatter(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<ReduceScatterRBLNWork>(
      outputs, inputs, opts.reduceOp, nextSeq(), currentStream(), comm_, worldSize, glooBackend_);
  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  const auto worldSize = getSize();

  RBLN_CHECK(outputTensor.dtype() == inputTensor.dtype());
  RBLN_CHECK(inputTensor.numel() == (outputTensor.numel() * worldSize));

  auto input_chunks = at::chunk(inputTensor, worldSize, 0);
  std::vector<std::vector<at::Tensor>> inputs = {input_chunks};
  std::vector<at::Tensor> outputs = {outputTensor};
  return reduce_scatter(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::scatter: {}", msg); };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertNonEmpty(invalidArgument, outputs);

  if (getRank() == opts.rootRank) {
    RBLN_CHECK(!inputs.empty(), "inputs cannot be empty on root rank");
    RBLN_CHECK(!inputs[0].empty(), "inputs[0] cannot be empty on root rank");
    assertDense(invalidArgument, inputs[0]);
  }

  assertDense(invalidArgument, outputs);

  const auto& device = outputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::scatter: CPU tensors require gloo_backend");
    return glooBackend_->scatter(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<ScatterRBLNWork>(
      outputs, inputs, opts.rootRank, nextSeq(), currentStream(), comm_, rank_, size_);
  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Point-to-Point Communication Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::send: {}", msg); };

  assertNonEmpty(invalidArgument, tensors);
  assertLayoutMatch(invalidArgument, tensors);
  assertTypeAndSizesMatch(invalidArgument, tensors);
  if (dstRank < 0 || dstRank >= size_) {
    invalidArgument(c10::str("invalid dstRank ", dstRank, " (world size ", size_, ")"));
  }
  if (dstRank == rank_) {
    invalidArgument(c10::str("cannot send to self (rank ", rank_, ")"));
  }

  const auto& device = tensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::send: CPU tensors require gloo_backend");
    return glooBackend_->send(tensors, dstRank, tag);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<SendRBLNWork>(tensors[0], dstRank, nextSeq(), currentStream(), comm_);
  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::recv: {}", msg); };

  assertNonEmpty(invalidArgument, tensors);
  assertLayoutMatch(invalidArgument, tensors);
  assertTypeAndSizesMatch(invalidArgument, tensors);
  if (srcRank < 0 || srcRank >= size_) {
    invalidArgument(c10::str("invalid srcRank ", srcRank, " (world size ", size_, ")"));
  }
  if (srcRank == rank_) {
    invalidArgument(c10::str("cannot recv from self (rank ", rank_, ")"));
  }

  const auto& device = tensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::recv: CPU tensors require gloo_backend");
    return glooBackend_->recv(tensors, srcRank, tag);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  nextTag();
  auto work = c10::make_intrusive<RecvRBLNWork>(tensors[0], srcRank, nextSeq(), currentStream(), comm_);
  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Synchronization Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::barrier(const BarrierOptions& opts) {
  if (glooBackend_) {
    return glooBackend_->barrier(opts);
  }

  // The barrier completes after every work in progress or pending now.
  std::vector<c10::weak_intrusive_ptr<Work>> priorWork;
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    priorWork.reserve(workInProgress_.size() + workQueue_.size());
    priorWork.insert(priorWork.end(), workInProgress_.begin(), workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }

  nextTag();
  auto work = c10::make_intrusive<BarrierRBLNWork>(
      std::move(priorWork), nextSeq(), size_ > 1 ? std::optional(currentStream()) : std::nullopt, comm_, rank_, device_id_, size_);
  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Sequence Number Management
// ============================================================================

void ProcessGroupRBLN::setSequenceNumberForGroup() {
  // Sequence numbers start at 0, as GLOO's and NCCL's do, and are kept by the instance.
}

uint64_t ProcessGroupRBLN::getSequenceNumberForGroup() {
  return seq_;
}

} // namespace c10d
