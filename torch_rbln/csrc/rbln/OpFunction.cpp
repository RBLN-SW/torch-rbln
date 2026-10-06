#include <torch_rbln/csrc/rbln/OpFunction.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNRuntime.h>
#include <rbln/artifact/file.h>
#include <rbln/runtime/flags.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>

namespace torch_rbln {

namespace {

namespace artifact = ::rbln::artifact;

// Recent bindings an executor keeps its program patched for; an eager op meets the same
// few blocks of the caching allocator again and again.
constexpr size_t kBindCache = 8;

std::atomic<uint64_t> run_clock{0};

const rt::Flag<int64_t> kCompiledOps(
    "TORCH_RBLN_COMPILED_OPS",
    "The ops compiled for profiles of their arguments a process keeps, letting go of the least "
    "recently run past them, with their device memory and programs.",
    2048,
    [](const std::string& raw) {
      int64_t count = rt::parseInt(raw);
      if (count < 1) {
        throw std::invalid_argument("expected one or more");
      }
      return count;
    });

const char* dtype_name(at::ScalarType type) {
  switch (type) {
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    case at::kFloat:
      return "float32";
    case at::kDouble:
      return "float64";
    case at::kLong:
      return "int64";
    case at::kInt:
      return "int32";
    case at::kShort:
      return "int16";
    case at::kChar:
      return "int8";
    case at::kByte:
      return "uint8";
    case at::kBool:
      return "bool";
    default:
      return nullptr;
  }
}

at::ScalarType scalar_type_of(const std::string& name) {
  for (auto type :
       {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble, at::kLong, at::kInt, at::kShort, at::kChar, at::kByte,
        at::kBool}) {
    if (name == dtype_name(type)) {
      return type;
    }
  }
  throw std::invalid_argument("no torch dtype holds " + name);
}

// Whether the bytes of `tensor` share memory with those of an input; a run neither reads
// what it writes nor writes what it reads.
bool overlaps_an_input(const at::Tensor& tensor, c10::ArrayRef<at::Tensor> inputs) {
  const auto begin = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  const auto end = begin + tensor.nbytes();
  return std::any_of(inputs.begin(), inputs.end(), [&](const at::Tensor& input) {
    if (!input.defined() || input.device() != tensor.device()) {
      return false;
    }
    const auto other = reinterpret_cast<uintptr_t>(input.const_data_ptr());
    return other < end && begin < other + input.nbytes();
  });
}

uint64_t logical_nbytes(const artifact::Arg& arg) {
  const auto& shape = arg.logical.shape;
  const auto numel = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
  return static_cast<uint64_t>(numel) * rt::dtypeSize(arg.logical.dtype);
}

// How torch holds a contiguous tensor of the shape and dtype of `arg`.
artifact::Layout torch_layout(const artifact::Arg& arg) {
  return artifact::layoutOf(arg.logical);
}

// Whether `arg` holds its value in one buffer as torch holds the tensor, so that a tensor over
// torch memory binds to it.
bool holds_as_torch(const artifact::Arg& arg) {
  return arg.shards.size() == 1 && arg.transform.empty() && artifact::alike(arg.physical, torch_layout(arg));
}

// A tensor of arg `arg` over `buffer`, which holds a value as torch holds a contiguous tensor.
template <class B>
std::shared_ptr<rt::BasicTensor<B>> over_torch_bytes(const artifact::Arg& arg, std::shared_ptr<B> buffer) {
  return std::make_shared<rt::BasicTensor<B>>(
      arg.logical, torch_layout(arg), "", std::vector{std::move(buffer)});
}

bool missing(const rt::Binding& binding) {
  return !binding.device && !binding.host;
}

} // namespace

OpFunction::OpFunction(
    const std::string& bytes,
    const std::vector<std::string>& inputs,
    const std::map<std::string, at::Tensor>& state,
    std::vector<bool> host_results)
    : num_inputs_(inputs.size()) {
  auto owner = std::make_shared<const std::string>(bytes);
  fn_ = rt::Function::from(
      artifact::deserialize(reinterpret_cast<const uint8_t*>(owner->data()), owner->size(), owner));
  const auto& f = fn_->artifact();
  std::map<std::string, size_t> input_of;
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_of.emplace(inputs[i], i);
  }
  auto require_static = [&](const artifact::Arg& arg) {
    if (!arg.logical.dynamic_axes.empty()) {
      throw std::invalid_argument(arg.name + " has a dynamic axis, which no torch tensor leaves open");
    }
  };

  std::map<std::string, size_t> input_arg;
  for (size_t index = 0; index < f.args.size(); ++index) {
    const auto& arg = f.args[index];
    if (arg.sources.empty()) {
      continue;
    }
    if (arg.sources.size() == 1 && input_of.count(arg.sources.front())) {
      require_static(arg);
      bindings_.push_back({index, input_of.at(arg.sources.front()), !holds_as_torch(arg)});
      input_arg.emplace(arg.name, input_of.at(arg.sources.front()));
      continue;
    }
    for (const auto& source : arg.sources) {
      auto it = state.find(source);
      if (it == state.end()) {
        throw std::invalid_argument(arg.name + " is made from " + source + ", which is neither an input nor state");
      }
      state_values_[source] = it->second.cpu().contiguous();
    }
    state_args_.push_back(index);
  }

  TORCH_CHECK(
      host_results.empty() || host_results.size() == f.results.size(),
      "host_results names ",
      host_results.size(),
      " results of ",
      f.results.size());
  for (size_t k = 0; k < f.results.size(); ++k) {
    const auto index = fn_->arg(f.results[k]);
    const auto& arg = f.args[index];
    auto it = input_arg.find(f.results[k]);
    if (it != input_arg.end()) {
      results_.push_back({std::nullopt, it->second});
    } else {
      require_static(arg);
      results_.push_back({index, std::nullopt, !host_results.empty() && host_results[k], !holds_as_torch(arg)});
    }
    outputs_.push_back({arg.logical.shape, scalar_type_of(arg.logical.dtype)});
  }
}

rt::Binding OpFunction::over(
    const at::Tensor& tensor,
    size_t index,
    c10::DeviceIndex device_index,
    std::vector<at::Tensor>* staged) const {
  const auto& arg = fn_->artifact().args[index];
  if (!tensor.defined()) {
    return {};
  }
  const char* dtype = dtype_name(tensor.scalar_type());
  if (dtype == nullptr || arg.logical.dtype != dtype || tensor.sizes() != c10::IntArrayRef(arg.logical.shape) ||
      !tensor.is_contiguous()) {
    return {};
  }
  const auto nbytes = logical_nbytes(arg);
  if (tensor.device().is_cpu()) {
    auto buffer = rt::HostBuffer::allocate(nbytes);
    std::memcpy(buffer->data(), tensor.const_data_ptr(), nbytes);
    return rt::Binding(over_torch_bytes(arg, std::move(buffer)));
  }
  if (!tensor.device().is_privateuseone() || tensor.device().index() != device_index) {
    return {};
  }
  auto location = c10::rbln::caching::try_locate(tensor.const_data_ptr());
  if (!location || location->available < nbytes ||
      (arg.alignment > 1 && (location->buffer->address() + location->offset) % arg.alignment != 0)) {
    if (!staged) {
      return {};
    }
    // An input the arg cannot take where it is, as a view off its alignment, is read from a copy.
    staged->push_back(at::empty(tensor.sizes(), tensor.options()).copy_(tensor));
    return over(staged->back(), index, device_index, nullptr);
  }
  return rt::Binding(over_torch_bytes(arg, rt::DeviceBuffer::view(location->buffer, location->offset, nbytes)));
}

std::shared_ptr<rt::HostTensor> OpFunction::encoded(size_t index) const {
  const auto& arg = fn_->artifact().args[index];
  auto host = fn_->emptyHostLike(index);
  std::vector<rt::HostArray> sources;
  for (const auto& source : arg.sources) {
    const auto& value = state_values_.at(source);
    sources.push_back({const_cast<void*>(value.const_data_ptr()), value.sizes().vec()});
  }
  fn_->encode(index, sources, *host);
  return host;
}

std::shared_ptr<rt::HostTensor> OpFunction::encoded(const at::Tensor& tensor, size_t index) const {
  const auto& arg = fn_->artifact().args[index];
  const char* dtype = dtype_name(tensor.scalar_type());
  if (!tensor.defined() || dtype == nullptr || arg.logical.dtype != dtype ||
      tensor.sizes() != c10::IntArrayRef(arg.logical.shape)) {
    return nullptr;
  }
  auto value = tensor.to(at::kCPU).contiguous();
  auto host = fn_->emptyHostLike(index);
  fn_->encode(index, {{value.data_ptr(), value.sizes().vec()}}, *host);
  c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kOpArgThroughHost, value.nbytes());
  return host;
}

std::vector<size_t> OpFunction::with_page_sharers(std::vector<size_t> args) const {
  const auto& f = fn_->artifact();
  std::set<size_t> pools;
  for (const auto index : args) {
    for (uint32_t shard = 0; shard < f.args[index].shards.size(); ++shard) {
      if (auto pooled = fn_->poolOf(index, shard)) {
        pools.insert(pooled->first);
      }
    }
  }
  for (const auto pool : pools) {
    for (const auto& member : f.pools[pool].members) {
      bool state = std::find(state_args_.begin(), state_args_.end(), member.arg) != state_args_.end();
      if (state && std::find(args.begin(), args.end(), member.arg) == args.end()) {
        args.push_back(member.arg);
      }
    }
  }
  return args;
}

void OpFunction::bind_state(const std::vector<Slot*>& slots, const std::vector<size_t>& args) const {
  if (args.empty()) {
    return;
  }
  std::vector<std::vector<std::shared_ptr<rt::Tensor>>> on_device;
  for (auto* slot : slots) {
    on_device.push_back(fn_->emptyLike(args, {slot->device}));
  }
  for (size_t k = 0; k < args.size(); ++k) {
    auto host = encoded(args[k]);
    for (auto& tensors : on_device) {
      rt::copy(*tensors[k], *host);
    }
  }
  for (size_t i = 0; i < slots.size(); ++i) {
    std::vector<std::pair<std::string, rt::Binding>> bound;
    for (size_t k = 0; k < args.size(); ++k) {
      bound.emplace_back(fn_->artifact().args[args[k]].name, rt::Binding(on_device[i][k]));
    }
    std::lock_guard<std::mutex> slot_lock(slots[i]->mutex);
    slots[i]->executor->bind(bound);
  }
}

void OpFunction::set_state(const std::map<std::string, at::Tensor>& values) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  for (const auto& [name, value] : values) {
    auto it = state_values_.find(name);
    TORCH_CHECK(it != state_values_.end(), "the function is made from no state named ", name);
    it->second = value.cpu().contiguous();
  }
  const auto& args = fn_->artifact().args;
  std::vector<size_t> changed;
  for (const auto index : state_args_) {
    const auto& sources = args[index].sources;
    if (std::any_of(sources.begin(), sources.end(), [&](const auto& source) { return values.count(source) != 0; })) {
      changed.push_back(index);
    }
  }
  std::vector<Slot*> slots;
  for (auto& [device_index, slot] : slots_) {
    slots.push_back(slot.get());
  }
  bind_state(slots, with_page_sharers(std::move(changed)));
}

OpFunction::Slot& OpFunction::slot(c10::DeviceIndex device_index) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  auto& entry = slots_[device_index];
  if (entry) {
    return *entry;
  }
  auto slot = std::make_unique<Slot>();
  slot->device = c10::rbln::runtime_device(device_index);
  slot->executor = std::make_shared<rt::Executor>(fn_, std::vector{slot->device}, kBindCache);
  bind_state({slot.get()}, state_args_);
  entry = std::move(slot);
  return *entry;
}

std::optional<std::vector<at::Tensor>> OpFunction::run(
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out) {
  TORCH_CHECK(inputs.size() == num_inputs_, "the function takes ", num_inputs_, " tensors, not ", inputs.size());
  TORCH_CHECK(out.size() <= results_.size(), "the function has ", results_.size(), " results, not ", out.size());
  last_run_.store(run_clock.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  c10::DeviceIndex device_index = -1;
  for (const auto& input : inputs) {
    if (input.defined() && input.device().is_privateuseone()) {
      device_index = input.device().index();
      break;
    }
  }
  if (device_index < 0) {
    device_index = c10::rbln::get_device_index();
  }

  const auto& args = fn_->artifact().args;
  std::vector<std::pair<std::string, rt::Binding>> bound;
  bound.reserve(bindings_.size() + results_.size());
  // Kept until the run is queued, so that no tensor made for a result takes their memory first.
  std::vector<at::Tensor> staged;
  // Inputs the run writes through the host, with the host tensors it leaves them in.
  std::vector<std::pair<const Binding*, std::shared_ptr<rt::HostTensor>>> written_back;
  for (const auto& binding : bindings_) {
    rt::Binding tensor;
    if (binding.through_host) {
      auto host = encoded(inputs[binding.input], binding.arg);
      if (host && args[binding.arg].access != artifact::Access::kRead) {
        written_back.emplace_back(&binding, host);
      }
      tensor = rt::Binding(std::move(host));
    } else {
      tensor = over(inputs[binding.input], binding.arg, device_index, &staged);
    }
    if (missing(tensor)) {
      return std::nullopt;
    }
    bound.emplace_back(args[binding.arg].name, std::move(tensor));
  }
  std::vector<at::Tensor> results;
  results.reserve(results_.size());
  bool on_host = false;
  // Results the run leaves in host tensors laid out as the function writes them, by result.
  std::vector<std::pair<size_t, std::shared_ptr<rt::HostTensor>>> to_decode;
  for (size_t k = 0; k < results_.size(); ++k) {
    const auto& result = results_[k];
    if (result.input) {
      results.push_back(inputs[*result.input]);
      continue;
    }
    if (result.through_host) {
      const bool given = k < out.size() && out[k].has_value() && out[k]->defined();
      if (given &&
          (out[k]->scalar_type() != outputs_[k].dtype || out[k]->sizes() != c10::IntArrayRef(outputs_[k].shape))) {
        return std::nullopt;
      }
      auto host = fn_->emptyHostLike(*result.arg);
      bound.emplace_back(args[*result.arg].name, host);
      to_decode.emplace_back(k, std::move(host));
      results.push_back(given ? *out[k] : at::Tensor());
      continue;
    }
    if (result.host) {
      const auto& arg = args[*result.arg];
      auto buffer = rt::HostBuffer::allocate(logical_nbytes(arg));
      bound.emplace_back(arg.name, over_torch_bytes(arg, buffer));
      results.push_back(at::from_blob(
          buffer->data(), outputs_[k].shape, [buffer](void*) {}, at::TensorOptions().dtype(outputs_[k].dtype)));
      on_host = true;
      continue;
    }
    const bool given = k < out.size() && out[k].has_value() && out[k]->defined();
    if (given && overlaps_an_input(*out[k], inputs)) {
      return std::nullopt;
    }
    at::Tensor tensor = given
        ? *out[k]
        : at::empty(
              outputs_[k].shape,
              at::TensorOptions().dtype(outputs_[k].dtype).device(c10::DeviceType::PrivateUse1, device_index));
    auto over_tensor = over(tensor, *result.arg, device_index, nullptr);
    if (missing(over_tensor)) {
      TORCH_CHECK(given, "a new tensor does not hold result ", k, " as the function writes it");
      return std::nullopt;
    }
    bound.emplace_back(args[*result.arg].name, std::move(over_tensor));
    results.push_back(std::move(tensor));
  }

  auto stream = c10::rbln::runtime_stream(c10::rbln::get_current_stream(device_index));
  auto& s = slot(device_index);
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.executor->bind(bound);
    s.executor->run(stream);
  }
  if (on_host || !to_decode.empty() || !written_back.empty()) {
    stream->synchronize();
  }
  for (const auto& [binding, host] : written_back) {
    const auto& input = inputs[binding->input];
    auto value = at::empty(input.sizes(), input.options().device(at::kCPU));
    fn_->decode(binding->arg, *host, {value.data_ptr(), value.sizes().vec()});
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kOpArgThroughHost, value.nbytes());
    input.copy_(value);
  }
  for (const auto& [k, host] : to_decode) {
    const auto& result = results_[k];
    auto value = at::empty(outputs_[k].shape, at::TensorOptions().dtype(outputs_[k].dtype));
    fn_->decode(*result.arg, *host, {value.data_ptr(), value.sizes().vec()});
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kOpArgThroughHost, value.nbytes());
    if (result.host) {
      results[k] = std::move(value);
      continue;
    }
    if (!results[k].defined()) {
      results[k] = at::empty(
          outputs_[k].shape,
          at::TensorOptions().dtype(outputs_[k].dtype).device(c10::DeviceType::PrivateUse1, device_index));
    }
    results[k].copy_(value);
  }
  return results;
}

void init_op_function_bindings(pybind11::module& module) {
  namespace py = pybind11;
  py::class_<OpFunction, std::shared_ptr<OpFunction>>(module, "_OpFunction")
      .def(
          py::init([](const py::bytes& function,
                      const std::vector<std::string>& inputs,
                      const std::map<std::string, at::Tensor>& state,
                      const std::vector<bool>& host_results) {
            std::string bytes = function;
            py::gil_scoped_release release;
            return std::make_shared<OpFunction>(bytes, inputs, state, host_results);
          }),
          py::arg("function"),
          py::arg("inputs"),
          py::arg("state") = std::map<std::string, at::Tensor>{},
          py::arg("host_results") = std::vector<bool>{},
          "Internal: a compiled function run on torch tensors")
      .def_property_readonly("num_inputs", &OpFunction::num_inputs)
      .def_property_readonly("last_run", &OpFunction::last_run)
      .def(
          "set_state",
          &OpFunction::set_state,
          py::arg("values"),
          py::call_guard<py::gil_scoped_release>(),
          "Internal: replaces the state `values` names and writes the args made from it anew")
      .def(
          "run",
          [](OpFunction& self, const std::vector<at::Tensor>& inputs, const std::vector<std::optional<at::Tensor>>& out) {
            return self.run(inputs, out);
          },
          py::arg("inputs"),
          py::arg("out") = std::vector<std::optional<at::Tensor>>{},
          py::call_guard<py::gil_scoped_release>(),
          "Internal: runs over `inputs` and returns the results, or None when a tensor does not fit");
}

} // namespace torch_rbln
