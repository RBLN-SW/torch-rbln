#include <torch_rbln/csrc/rbln/OpFunction.h>

#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.hpp>
#include <torch_rbln/csrc/rbln/DeferredOps.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNRuntime.h>
#include <c10/rbln/RBLNTyped.h>
#include <c10/util/accumulate.h>
#include <rebel/v2/artifact/file.h>
#include <rebel/v2/runtime/flags.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace torch_rbln {

// What tells a tensor of state on a device apart: the device, how the arg holds it (its type
// and the pools its shards are in), and the torch tensors it is made from, as they are now.
struct StateKey {
  uint32_t device = 0;
  std::string type;
  std::vector<std::string> pools;
  std::vector<std::tuple<const void*, int64_t, std::vector<int64_t>, std::vector<int64_t>, int>> sources;

  bool operator<(const StateKey& other) const {
    return std::tie(device, type, pools, sources) < std::tie(other.device, other.type, other.pools, other.sources);
  }
};

namespace {

namespace artifact = ::rebel::v2::artifact;

// Recent bindings an executor keeps its program patched for; an eager op meets the same
// few blocks of the caching allocator again and again.
constexpr size_t kBindCache = 8;

std::atomic<uint64_t> run_clock{0};

// The tensors of state the process holds on devices, which the functions that hold one alike
// share. Each function holding one holds the torch tensors it is made from, so no other tensor
// takes their memory while it lives.
class StateTensors {
 public:
  static StateTensors& get() {
    static StateTensors tensors;
    return tensors;
  }

  std::shared_ptr<rt::Tensor> find(const StateKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = held_.find(key);
    return it == held_.end() ? nullptr : it->second.lock();
  }

  void add(StateKey key, const std::shared_ptr<rt::Tensor>& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = held_.begin(); it != held_.end();) {
      it = it->second.expired() ? held_.erase(it) : std::next(it);
    }
    held_[std::move(key)] = tensor;
  }

 private:
  std::mutex mutex_;
  std::map<StateKey, std::weak_ptr<rt::Tensor>> held_;
};

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

// Whether a tensor of `sizes` has the shape of `arg`, its dynamic axis within the range of that axis.
bool fits(const artifact::Arg& arg, c10::IntArrayRef sizes) {
  const auto& logical = arg.logical;
  if (sizes.size() != logical.shape.size()) {
    return false;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (!logical.dynamic_axes.empty() && logical.dynamic_axes.front().axis == i) {
      const auto& axis = logical.dynamic_axes.front();
      if (sizes[i] < axis.min || (axis.max && sizes[i] > axis.max)) {
        return false;
      }
    } else if (sizes[i] != logical.shape[i]) {
      return false;
    }
  }
  return true;
}

// Whether `arg` holds its value in one buffer as torch holds the tensor, so that a tensor over
// torch memory binds to it.
bool holds_as_torch(const artifact::Arg& arg) {
  return arg.shards.size() == 1 && arg.transform.empty() && artifact::alike(arg.physical, torch_layout(arg));
}

// The sizes of `tensor`, with a leading extent of `extent` when one is given.
c10::SmallVector<int64_t, 6> sizes_at(const at::Tensor& tensor, std::optional<int64_t> extent) {
  c10::SmallVector<int64_t, 6> sizes(tensor.sizes().begin(), tensor.sizes().end());
  if (extent) {
    sizes.front() = *extent;
  }
  return sizes;
}

// Whether `tensor`, standing at `sizes`, holds a value of `arg` as torch holds a contiguous tensor.
bool of_arg(const artifact::Arg& arg, const at::Tensor& tensor, c10::IntArrayRef sizes) {
  const char* dtype = tensor.defined() ? c10::rbln::dtype_name(tensor.scalar_type()) : nullptr;
  return dtype != nullptr && arg.logical.dtype == dtype && fits(arg, sizes) && tensor.is_contiguous();
}

// A tensor of arg `arg` and of `shape` over `buffer`, which holds a value as torch holds a contiguous
// tensor.
template <class B>
std::shared_ptr<rt::BasicTensor<B>> over_torch_bytes(
    const artifact::Arg& arg,
    const std::vector<int64_t>& shape,
    std::shared_ptr<B> buffer) {
  artifact::Logical logical{arg.logical.dtype, shape, {}};
  auto layout = artifact::layoutOf(logical);
  return std::make_shared<rt::BasicTensor<B>>(
      std::move(logical), std::move(layout), "", std::vector{std::move(buffer)});
}

bool missing(const rt::Binding& binding) {
  return !binding.device && !binding.host;
}

// The first rows of tensor, over all of its storage, as a tensor not marked as a view of it.
at::Tensor first_rows(const at::Tensor& tensor, int64_t rows) {
  auto sizes = tensor.sizes().vec();
  sizes.front() = rows;
  auto out = at::detail::make_tensor<c10::TensorImpl>(c10::Storage(tensor.storage()), tensor.key_set(), tensor.dtype());
  out.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  return out;
}

} // namespace

const rt::Flag<int64_t> kCompiledOps(
    "TORCH_RBLN_COMPILED_OPS",
    "The ops compiled for profiles of their arguments a process keeps, letting go of the least "
    "recently run past them, with their device memory and programs; as many segments of deferred "
    "eager ops besides.",
    2048,
    [](const std::string& raw) {
      int64_t count = rt::parseInt(raw);
      if (count < 1) {
        throw std::invalid_argument("expected one or more");
      }
      return count;
    });

OpFunction::OpFunction(
    const std::string& bytes,
    const std::vector<std::string>& inputs,
    const std::map<std::string, at::Tensor>& state,
    std::vector<bool> host_results,
    const std::vector<size_t>& rows)
    : num_inputs_(inputs.size()) {
  auto owner = std::make_shared<const std::string>(bytes);
  fn_ =
      rt::Function::from(artifact::deserialize(reinterpret_cast<const uint8_t*>(owner->data()), owner->size(), owner));
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

  using Kind = artifact::Location::Kind;
  // Whether the transforms make arg `index` of values alone, as they encode an input.
  auto of_values = [&](size_t index) {
    bool made = false;
    for (const auto& t : f.transforms) {
      if (std::none_of(t.outputs.begin(), t.outputs.end(), [&](const auto& l) {
            return l.kind == Kind::kArg && l.index == index;
          })) {
        continue;
      }
      made = true;
      if (std::any_of(t.inputs.begin(), t.inputs.end(), [](const auto& l) { return l.kind != Kind::kValue; })) {
        return false;
      }
    }
    return made;
  };
  auto of_state = [&](const artifact::Arg& arg) {
    return std::all_of(
        arg.sources.begin(), arg.sources.end(), [&](const auto& source) { return state.count(source) != 0; });
  };
  // The args the host work of calls reads to make other args; a decode makes a value alone.
  std::set<size_t> read_by_calls;
  for (const auto& t : f.transforms) {
    const bool makes_args =
        std::any_of(t.outputs.begin(), t.outputs.end(), [](const auto& l) { return l.kind != Kind::kValue; });
    for (const auto& l : t.inputs) {
      if (!t.cold && makes_args && l.kind == Kind::kArg) {
        read_by_calls.insert(l.index);
      }
    }
  }
  const std::set<std::string> results(f.results.begin(), f.results.end());

  std::map<std::string, size_t> input_arg;
  for (size_t index = 0; index < f.args.size(); ++index) {
    const auto& arg = f.args[index];
    const bool used = artifact::used(f, index);
    if (arg.sources.empty()) {
      if (used && arg.access == artifact::Access::kWrite && !results.count(arg.name)) {
        left_args_.push_back(index);
      }
      continue;
    }
    if (arg.sources.size() == 1 && input_of.count(arg.sources.front()) && of_values(index)) {
      input_arg.emplace(arg.name, input_of.at(arg.sources.front()));
      if (used) {
        bindings_.push_back({index, input_of.at(arg.sources.front()), !holds_as_torch(arg)});
      }
      continue;
    }
    if (of_state(arg)) {
      for (const auto& source : arg.sources) {
        state_values_[source] = state.at(source);
      }
      if (used) {
        state_args_.push_back(index);
      }
      if (read_by_calls.count(index)) {
        held_state_[index] = nullptr;
      }
      continue;
    }
    for (const auto& source : arg.sources) {
      if (!input_of.count(source) && !state.count(source)) {
        throw std::invalid_argument(arg.name + " is made from " + source + ", which is neither an input nor state");
      }
    }
    if (used) {
      require_static(arg);
      made_args_.push_back(index);
    }
  }

  TORCH_CHECK(
      host_results.empty() || host_results.size() == f.results.size(),
      "host_results names ",
      host_results.size(),
      " results of ",
      f.results.size());
  std::vector<size_t> after;
  for (size_t k = 0; k < f.results.size(); ++k) {
    const auto index = fn_->arg(f.results[k]);
    const auto& arg = f.args[index];
    auto it = input_arg.find(f.results[k]);
    if (it != input_arg.end()) {
      results_.push_back({std::nullopt, it->second});
    } else {
      require_static(arg);
      const bool made_after = !artifact::used(f, index);
      if (made_after) {
        after.push_back(index);
      }
      results_.push_back(
          {index,
           std::nullopt,
           !host_results.empty() && host_results[k],
           !holds_as_torch(arg) || read_by_calls.count(index) != 0,
           made_after});
    }
    outputs_.push_back({arg.logical.shape, c10::rbln::scalar_type_of(arg.logical.dtype)});
  }

  // The host work of a call reads the values of inputs, and the host tensors of the args made
  // of state, of those it makes and of those the run leaves on the host.
  auto inputs_read = [&](const rt::Maker& maker) {
    std::vector<size_t> out;
    for (const auto v : maker.reads()) {
      auto it = input_of.find(f.values[v].name);
      if (it == input_of.end()) {
        throw std::invalid_argument("the host work of a call reads " + f.values[v].name + ", which is no input");
      }
      out.push_back(it->second);
    }
    return out;
  };
  std::vector<size_t> held;
  for (auto& [index, host] : held_state_) {
    host = encoded(index);
    held.push_back(index);
  }
  if (!made_args_.empty()) {
    before_ = fn_->maker(made_args_, {}, held);
    before_inputs_ = inputs_read(*before_);
  }
  held.insert(held.end(), made_args_.begin(), made_args_.end());
  held.insert(held.end(), left_args_.begin(), left_args_.end());
  for (const auto& result : results_) {
    if (result.arg && result.through_host && !result.after &&
        std::find(held.begin(), held.end(), *result.arg) == held.end()) {
      held.push_back(*result.arg);
    }
  }
  if (!after.empty()) {
    after_ = fn_->maker(after, {}, held);
    after_inputs_ = inputs_read(*after_);
  }
  for (const auto input : rows) {
    TORCH_CHECK(input < num_inputs_, "the function takes ", num_inputs_, " tensors, not input ", input);
  }
  if (const auto extent = row_extent(rows)) {
    row_inputs_ = rows;
    row_extent_ = *extent;
    for (auto& binding : bindings_) {
      binding.row = std::ranges::find(row_inputs_, binding.input) != row_inputs_.end();
    }
  }
  for (const auto& binding : bindings_) {
    if (f.args[binding.arg].access != artifact::Access::kRead) {
      updated_inputs_.push_back(binding.input);
    }
  }
  runs_on_device_ = made_args_.empty() && left_args_.empty() && held_state_.empty() &&
      std::ranges::none_of(bindings_, [](const Binding& b) { return b.through_host; }) &&
      std::ranges::all_of(results_, [](const Result& r) { return r.arg && !r.host && !r.through_host && !r.after; });
}

std::optional<int64_t> OpFunction::row_extent(const std::vector<size_t>& rows) const {
  if (rows.empty() || results_.empty()) {
    return std::nullopt;
  }
  int64_t extent = 0;
  for (size_t k = 0; k < results_.size(); ++k) {
    const auto& shape = outputs_[k].shape;
    if (!results_[k].arg || results_[k].host || shape.empty() || (k > 0 && shape.front() != extent)) {
      return std::nullopt;
    }
    extent = shape.front();
  }
  for (const auto input : rows) {
    if (std::ranges::find(before_inputs_, input) != before_inputs_.end() ||
        std::ranges::find(after_inputs_, input) != after_inputs_.end()) {
      return std::nullopt;
    }
  }
  const auto& args = fn_->artifact().args;
  for (const auto& binding : bindings_) {
    if (std::ranges::find(rows, binding.input) == rows.end()) {
      continue;
    }
    const auto& arg = args[binding.arg];
    if (arg.access != artifact::Access::kRead || arg.logical.shape.empty() || arg.logical.shape.front() != extent) {
      return std::nullopt;
    }
  }
  return extent;
}

std::optional<c10::rbln::Location> OpFunction::in_place(
    const at::Tensor& tensor,
    size_t index,
    c10::DeviceIndex device_index,
    std::optional<int64_t> extent) const {
  const auto& arg = fn_->artifact().args[index];
  if (!tensor.defined() || !tensor.device().is_privateuseone() || tensor.device().index() != device_index) {
    return std::nullopt;
  }
  const auto sizes = sizes_at(tensor, extent);
  if (!of_arg(arg, tensor, sizes)) {
    return std::nullopt;
  }
  // A tensor of a type holds its value otherwise than torch holds it (see c10/rbln/RBLNTyped.h).
  auto found = c10::rbln::caching::try_locate_typed(tensor.const_data_ptr());
  const uint64_t nbytes = c10::multiply_integers(sizes) * tensor.element_size();
  if (!found || found->type || (extent ? found->block_available : found->location.available) < nbytes ||
      (arg.alignment > 1 && (found->location.buffer->address() + found->location.offset) % arg.alignment != 0)) {
    return std::nullopt;
  }
  return found->location;
}

void OpFunction::note_run() {
  last_run_.store(run_clock.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

bool OpFunction::binds_in_place(
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out,
    c10::DeviceIndex device_index) const {
  if (inputs.size() != num_inputs_ || out.size() > results_.size()) {
    return false;
  }
  for (const auto& binding : bindings_) {
    if (!in_place(inputs[binding.input], binding.arg, device_index)) {
      return false;
    }
  }
  for (size_t k = 0; k < out.size(); ++k) {
    if (out[k] && out[k]->defined() && !in_place(*out[k], *results_[k].arg, device_index)) {
      return false;
    }
  }
  return true;
}

rt::Binding OpFunction::over(
    const at::Tensor& tensor,
    size_t index,
    c10::DeviceIndex device_index,
    std::vector<Copy>* copies,
    std::optional<int64_t> extent) const {
  const auto& arg = fn_->artifact().args[index];
  if (!tensor.defined()) {
    return {};
  }
  const auto sizes = sizes_at(tensor, extent);
  const std::vector<int64_t> shape(sizes.begin(), sizes.end());
  if (auto location = in_place(tensor, index, device_index, extent)) {
    const auto nbytes = c10::multiply_integers(sizes) * tensor.element_size();
    return rt::Binding(
        over_torch_bytes(arg, shape, rt::DeviceBuffer::view(location->buffer, location->offset, nbytes)));
  }
  if (!copies || !of_arg(arg, tensor, sizes)) {
    return {};
  }
  const bool read = arg.access != artifact::Access::kWrite;
  const bool written = arg.access != artifact::Access::kRead;
  if (tensor.device().is_cpu()) {
    auto buffer = rt::HostBuffer::allocate(c10::multiply_integers(sizes) * tensor.element_size());
    if (read) {
      std::memcpy(buffer->data(), tensor.const_data_ptr(), tensor.nbytes());
    }
    copies->push_back({tensor, at::from_blob(buffer->data(), sizes, [buffer](void*) {}, tensor.options()), written});
    return rt::Binding(over_torch_bytes(arg, shape, std::move(buffer)));
  }
  if (!tensor.device().is_privateuseone() || tensor.device().index() != device_index) {
    return {};
  }
  // A tensor of a type binds through a copy, as torch holds it, which the copy moves through the
  // host (see c10/rbln/RBLNTyped.h).
  auto found = c10::rbln::caching::try_locate_typed(tensor.const_data_ptr());
  if (found && found->type && written && c10::rbln::is_fallback_disabled("host_round_trip")) {
    throw c10::rbln::FallbackDisabled(
        "arg " + arg.name + " is written in place, but its tensor is of the type of arg " +
        c10::rbln::typed::describe(*found->type) +
        ", so the run would take it through the host (TORCH_RBLN_DISABLE_FALLBACK=host_round_trip)");
  }
  auto copy = at::empty(sizes, tensor.options());
  if (read) {
    (extent ? copy.narrow(0, 0, tensor.size(0)) : copy).copy_(tensor);
  }
  copies->push_back({tensor, copy, written});
  return over(copy, index, device_index, nullptr);
}

rt::Binding OpFunction::typed(const at::Tensor& tensor, size_t index, c10::DeviceIndex device_index) const {
  const auto& arg = fn_->artifact().args[index];
  if (!tensor.defined() || tensor.numel() == 0 || !tensor.device().is_privateuseone() ||
      tensor.device().index() != device_index || arg.shards.size() != 1 || fn_->poolOf(index, 0) ||
      !c10::rbln::caching::any_typed()) {
    return {};
  }
  const char* dtype = c10::rbln::dtype_name(tensor.scalar_type());
  if (dtype == nullptr || arg.logical.dtype != dtype || !fits(arg, tensor.sizes()) || !tensor.is_contiguous()) {
    return {};
  }
  // A type is that of a whole allocation, which a tensor over all of it holds.
  const auto& storage = tensor.storage();
  if (tensor.const_data_ptr() != storage.data() || tensor.nbytes() != storage.nbytes()) {
    return {};
  }
  auto found = c10::rbln::caching::try_locate_typed(tensor.const_data_ptr());
  auto shape = tensor.sizes().vec();
  if (!found || !found->type || found->start != tensor.const_data_ptr() || found->type->id != artifact::typeId(arg) ||
      found->type->shape != shape) {
    return {};
  }
  const auto nbytes = artifact::minNbytes(arg, arg.shards.front(), shape);
  if (found->location.available < nbytes ||
      (arg.alignment > 1 && (found->location.buffer->address() + found->location.offset) % arg.alignment != 0)) {
    return {};
  }
  artifact::Logical logical{arg.logical.dtype, std::move(shape), {}};
  return rt::Binding(
      std::make_shared<rt::Tensor>(
          std::move(logical),
          arg.physical,
          arg.transform,
          std::vector{rt::DeviceBuffer::view(found->location.buffer, found->location.offset, nbytes)}));
}

std::shared_ptr<rt::HostTensor> OpFunction::encoded(size_t index) const {
  const auto& arg = fn_->artifact().args[index];
  auto host = fn_->emptyHostLike(index);
  std::vector<at::Tensor> values;
  std::vector<rt::HostArray> sources;
  for (const auto& source : arg.sources) {
    values.push_back(state_values_.at(source).cpu().contiguous());
    sources.push_back({values.back().data_ptr(), values.back().sizes().vec()});
  }
  fn_->encode(index, sources, *host);
  return host;
}

std::shared_ptr<rt::HostTensor> OpFunction::encoded(
    const at::Tensor& tensor,
    size_t index,
    std::optional<int64_t> extent) const {
  const auto& arg = fn_->artifact().args[index];
  if (!tensor.defined()) {
    return nullptr;
  }
  auto sizes = tensor.sizes().vec();
  if (extent) {
    sizes.front() = *extent;
  }
  const char* dtype = c10::rbln::dtype_name(tensor.scalar_type());
  if (dtype == nullptr || arg.logical.dtype != dtype || !fits(arg, sizes)) {
    return nullptr;
  }
  auto value = tensor.to(at::kCPU).contiguous();
  if (extent) {
    auto padded = at::zeros(sizes, value.options());
    padded.narrow(0, 0, value.size(0)).copy_(value);
    value = std::move(padded);
  }
  auto host = fn_->emptyHostLike(index, value.sizes().vec());
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

StateKey OpFunction::state_key(size_t index, const rt::Device& device) const {
  const auto& arg = fn_->artifact().args[index];
  StateKey key{device.id(), artifact::typeId(arg), {}, {}};
  for (uint32_t shard = 0; shard < arg.shards.size(); ++shard) {
    auto pooled = fn_->poolOf(index, shard);
    key.pools.push_back(pooled ? fn_->artifact().pools[pooled->first].id : "");
  }
  for (const auto& source : arg.sources) {
    const auto& value = state_values_.at(source);
    key.sources.emplace_back(
        value.const_data_ptr(),
        value._version(),
        value.sizes().vec(),
        value.strides().vec(),
        static_cast<int>(value.scalar_type()));
  }
  return key;
}

void OpFunction::bind_state(const std::vector<Slot*>& slots, const std::vector<size_t>& args) const {
  if (args.empty()) {
    return;
  }
  // Args whose shards share pages are held in one buffer, so they are found or made together.
  std::map<size_t, std::vector<size_t>> groups;
  for (const auto index : args) {
    auto pooled = fn_->poolOf(index, 0);
    groups[pooled ? pooled->first : fn_->artifact().pools.size() + index].push_back(index);
  }
  std::map<size_t, std::shared_ptr<rt::HostTensor>> hosts;
  for (auto* slot : slots) {
    std::map<size_t, std::shared_ptr<rt::Tensor>> tensors;
    std::vector<size_t> missing;
    for (const auto& [group, members] : groups) {
      std::vector<std::shared_ptr<rt::Tensor>> found;
      for (const auto index : members) {
        if (auto tensor = StateTensors::get().find(state_key(index, *slot->device))) {
          found.push_back(std::move(tensor));
        }
      }
      if (found.size() == members.size()) {
        for (size_t k = 0; k < members.size(); ++k) {
          tensors[members[k]] = found[k];
        }
      } else {
        missing.insert(missing.end(), members.begin(), members.end());
      }
    }
    auto made = fn_->emptyLike(missing, {slot->device});
    for (size_t k = 0; k < missing.size(); ++k) {
      auto& host = hosts[missing[k]];
      if (!host) {
        host = encoded(missing[k]);
      }
      rt::copy(*made[k], *host);
      StateTensors::get().add(state_key(missing[k], *slot->device), made[k]);
      tensors[missing[k]] = made[k];
    }
    std::vector<std::pair<std::string, rt::Binding>> bound;
    for (const auto index : args) {
      bound.emplace_back(fn_->artifact().args[index].name, rt::Binding(tensors.at(index)));
    }
    std::lock_guard<std::mutex> slot_lock(slot->mutex);
    slot->executor->bind(bound);
  }
}

void OpFunction::set_state(const std::map<std::string, at::Tensor>& values) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  for (const auto& [name, value] : values) {
    auto it = state_values_.find(name);
    TORCH_CHECK(it != state_values_.end(), "the function is made from no state named ", name);
    it->second = value;
  }
  const auto& args = fn_->artifact().args;
  for (auto& [index, host] : held_state_) {
    const auto& sources = args[index].sources;
    if (std::any_of(sources.begin(), sources.end(), [&](const auto& source) { return values.count(source) != 0; })) {
      host = encoded(index);
    }
  }
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

void OpFunction::set_communicator(std::shared_ptr<rt::Communicator> communicator) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  communicator_ = std::move(communicator);
  for (auto& [device_index, slot] : slots_) {
    std::lock_guard<std::mutex> slot_lock(slot->mutex);
    slot->executor->setCommunicator(communicator_);
  }
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
  if (communicator_) {
    slot->executor->setCommunicator(communicator_);
  }
  bind_state({slot.get()}, state_args_);
  entry = std::move(slot);
  return *entry;
}

std::optional<std::vector<at::Tensor>> OpFunction::run(
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out) {
  TORCH_CHECK(inputs.size() == num_inputs_, "the function takes ", num_inputs_, " tensors, not ", inputs.size());
  TORCH_CHECK(out.size() <= results_.size(), "the function has ", results_.size(), " results, not ", out.size());
  note_run();
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
  // The call's rows along the results' leading axis, when fewer than the function computes.
  std::optional<int64_t> rows;
  if (!row_inputs_.empty()) {
    const auto& first = inputs[row_inputs_.front()];
    if (!first.defined() || first.dim() == 0) {
      return std::nullopt;
    }
    const int64_t count = first.size(0);
    for (const auto input : row_inputs_) {
      const auto& tensor = inputs[input];
      if (!tensor.defined() || tensor.dim() == 0 || tensor.size(0) != count) {
        return std::nullopt;
      }
    }
    if (count < 1 || count > row_extent_) {
      return std::nullopt;
    }
    if (count < row_extent_) {
      rows = count;
    }
  }

  const auto& args = fn_->artifact().args;
  std::vector<std::pair<std::string, rt::Binding>> bound;
  bound.reserve(bindings_.size() + results_.size());
  // Kept until the run is done, so that no tensor made for a result takes their memory first.
  std::vector<Copy> copies;
  // Inputs the run writes through the host, with the host tensors it leaves them in.
  std::vector<std::pair<const Binding*, std::shared_ptr<rt::HostTensor>>> written_back;
  for (const auto& binding : bindings_) {
    rt::Binding tensor;
    if (binding.through_host) {
      tensor = typed(inputs[binding.input], binding.arg, device_index);
    }
    if (binding.through_host && missing(tensor)) {
      const auto& arg = args[binding.arg];
      const bool written = arg.access != artifact::Access::kRead;
      if (written && c10::rbln::is_fallback_disabled("host_round_trip")) {
        throw c10::rbln::FallbackDisabled(
            "input " + arg.sources.front() +
            " is written in place, but the device holds it otherwise than torch holds the tensor, and the tensor "
            "is not of the arg's type (torch.rbln.empty_typed), so the run would take it through the host "
            "(TORCH_RBLN_DISABLE_FALLBACK=host_round_trip)");
      }
      auto host = encoded(
          inputs[binding.input], binding.arg, rows && binding.row ? std::optional<int64_t>(row_extent_) : std::nullopt);
      if (host && written) {
        written_back.emplace_back(&binding, host);
      }
      tensor = rt::Binding(std::move(host));
    } else if (!binding.through_host) {
      tensor = over(
          inputs[binding.input],
          binding.arg,
          device_index,
          &copies,
          rows && binding.row ? std::optional<int64_t>(row_extent_) : std::nullopt);
    }
    if (missing(tensor)) {
      return std::nullopt;
    }
    bound.emplace_back(args[binding.arg].name, std::move(tensor));
  }
  // What the host work of the call reads and writes: the values of inputs on the host, by input,
  // and host tensors of args, by arg, which it holds until the call is done.
  std::map<size_t, at::Tensor> values;
  std::map<size_t, std::shared_ptr<rt::HostTensor>> host_of;
  auto arrays_of = [&](const rt::Maker& maker, const std::vector<size_t>& from) {
    std::optional<std::vector<rt::HostArray>> out(std::in_place);
    for (size_t k = 0; k < from.size(); ++k) {
      const auto& value = fn_->artifact().values[maker.reads()[k]];
      auto& host = values[from[k]];
      if (!host.defined()) {
        const auto& tensor = inputs[from[k]];
        const char* dtype = c10::rbln::dtype_name(tensor.scalar_type());
        if (!tensor.defined() || dtype == nullptr || value.logical.dtype != dtype ||
            (value.logical.dynamic_axes.empty() && tensor.sizes() != c10::IntArrayRef(value.logical.shape))) {
          return decltype(out)();
        }
        host = tensor.to(at::kCPU).contiguous();
        if (!tensor.device().is_cpu()) {
          c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kOpArgThroughHost, tensor.nbytes());
        }
      }
      out->push_back({host.data_ptr(), host.sizes().vec()});
    }
    return out;
  };
  auto tensors_of = [&](const rt::Maker& maker) {
    std::vector<const rt::HostTensor*> out;
    for (const auto index : maker.tensors()) {
      out.push_back(host_of.at(index).get());
    }
    return out;
  };
  std::optional<std::vector<rt::HostArray>> before_arrays, after_arrays;
  if (before_ && !(before_arrays = arrays_of(*before_, before_inputs_))) {
    return std::nullopt;
  }
  if (after_ && !(after_arrays = arrays_of(*after_, after_inputs_))) {
    return std::nullopt;
  }
  if (!held_state_.empty()) {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    host_of.insert(held_state_.begin(), held_state_.end());
  }
  if (before_) {
    for (const auto index : made_args_) {
      host_of[index] = fn_->emptyHostLike(index);
      bound.emplace_back(args[index].name, host_of[index]);
    }
    (*before_)(*before_arrays, tensors_of(*before_));
  }
  for (const auto index : left_args_) {
    host_of[index] = fn_->emptyHostLike(index);
    bound.emplace_back(args[index].name, host_of[index]);
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
    // A tensor of the call's rows does not hold the rows the run computes past them.
    if (rows && k < out.size() && out[k].has_value() && out[k]->defined()) {
      return std::nullopt;
    }
    if (result.after) {
      const bool given = k < out.size() && out[k].has_value() && out[k]->defined();
      if (given &&
          (out[k]->scalar_type() != outputs_[k].dtype || out[k]->sizes() != c10::IntArrayRef(outputs_[k].shape))) {
        return std::nullopt;
      }
      auto& host = host_of[*result.arg];
      host = fn_->emptyHostLike(*result.arg);
      to_decode.emplace_back(k, host);
      results.push_back(given ? *out[k] : at::Tensor());
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
      host_of[*result.arg] = host;
      to_decode.emplace_back(k, std::move(host));
      results.push_back(given ? *out[k] : at::Tensor());
      continue;
    }
    if (result.host) {
      const auto& arg = args[*result.arg];
      auto buffer = rt::HostBuffer::allocate(logical_nbytes(arg));
      bound.emplace_back(arg.name, over_torch_bytes(arg, arg.logical.shape, buffer));
      results.push_back(
          at::from_blob(
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
    auto over_tensor = over(tensor, *result.arg, device_index, &copies);
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
  const bool written_on_host =
      std::any_of(copies.begin(), copies.end(), [](const Copy& c) { return c.written && c.copy.device().is_cpu(); });
  if (on_host || written_on_host || !to_decode.empty() || !written_back.empty()) {
    stream->synchronize();
  }
  for (const auto& c : copies) {
    if (c.written) {
      c.tensor.copy_(c.copy);
    }
  }
  for (const auto& [binding, host] : written_back) {
    const auto& input = inputs[binding->input];
    auto value = at::empty(input.sizes(), input.options().device(at::kCPU));
    fn_->decode(binding->arg, *host, {value.data_ptr(), value.sizes().vec()});
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kOpArgThroughHost, value.nbytes());
    input.copy_(value);
  }
  if (after_) {
    (*after_)(*after_arrays, tensors_of(*after_));
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
  if (rows) {
    for (auto& result : results) {
      result = first_rows(result, *rows);
    }
  }
  return results;
}

TypedFunction::TypedFunction(const std::string& bytes) {
  auto owner = std::make_shared<const std::string>(bytes);
  fn_ =
      rt::Function::from(artifact::deserialize(reinterpret_cast<const uint8_t*>(owner->data()), owner->size(), owner));
}

at::Tensor TypedFunction::empty(
    const std::string& name,
    const std::vector<int64_t>& shape,
    c10::DeviceIndex device_index,
    bool zero) const {
  const auto index = fn_->arg(name);
  const auto& arg = fn_->artifact().args[index];
  const auto options = at::TensorOptions()
                           .dtype(c10::rbln::scalar_type_of(arg.logical.dtype))
                           .device(c10::DeviceType::PrivateUse1, device_index);
  if (holds_as_torch(arg)) {
    return zero ? at::zeros(shape, options) : at::empty(shape, options);
  }
  auto type = c10::rbln::typed::of(fn_, index, shape);
  const c10::DeviceGuard guard(options.device());
  const auto nbytes = c10::rbln::typed::nbytes(*type);
  auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes, allocator->allocate(nbytes), allocator, false);
  c10::rbln::caching::set_type(storage.mutable_data(), std::move(type));
  auto tensor = at::empty({0}, options);
  tensor.set_(storage, 0, shape);
  if (zero) {
    tensor.zero_();
  }
  return tensor;
}

void init_op_function_bindings(pybind11::module& module) {
  namespace py = pybind11;
  py::class_<OpFunction, std::shared_ptr<OpFunction>>(module, "_OpFunction")
      .def(
          py::init([](const py::bytes& function,
                      const std::vector<std::string>& inputs,
                      const std::map<std::string, at::Tensor>& state,
                      const std::vector<bool>& host_results,
                      const std::vector<size_t>& rows) {
            std::string bytes = function;
            py::gil_scoped_release release;
            return std::make_shared<OpFunction>(bytes, inputs, state, host_results, rows);
          }),
          py::arg("function"),
          py::arg("inputs"),
          py::arg("state") = std::map<std::string, at::Tensor>{},
          py::arg("host_results") = std::vector<bool>{},
          py::arg("rows") = std::vector<size_t>{},
          "Internal: a compiled function run on torch tensors")
      .def_property_readonly("num_inputs", &OpFunction::num_inputs)
      .def_property_readonly("row_wise", &OpFunction::row_wise)
      .def_property_readonly("last_run", &OpFunction::last_run)
      .def(
          "set_state",
          &OpFunction::set_state,
          py::arg("values"),
          py::call_guard<py::gil_scoped_release>(),
          "Internal: replaces the state `values` names and writes the args made from it anew")
      .def(
          "set_communicator",
          [](OpFunction& self, const py::object& backend) {
            self.set_communicator(backend.cast<::c10d::ProcessGroupRBLN&>().communicator());
          },
          py::arg("backend"),
          "Internal: runs the function's collectives over the communicator of `backend`, a "
          "ProcessGroupRBLN")
      .def(
          "run",
          [](OpFunction& self,
             const std::vector<at::Tensor>& inputs,
             const std::vector<std::optional<at::Tensor>>& out) { return self.run(inputs, out); },
          py::arg("inputs"),
          py::arg("out") = std::vector<std::optional<at::Tensor>>{},
          py::call_guard<py::gil_scoped_release>(),
          "Internal: runs over `inputs` and returns the results, or None when a tensor does not fit")
      .def(
          "submit",
          [](const std::shared_ptr<OpFunction>& self,
             const std::vector<at::Tensor>& inputs,
             const std::vector<std::optional<at::Tensor>>& out,
             bool own_out) { return deferred::submit(self, inputs, out, own_out); },
          py::arg("inputs"),
          py::arg("out") = std::vector<std::optional<at::Tensor>>{},
          py::arg("own_out") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Internal: runs over `inputs` as `run` does, or defers the run while TORCH_RBLN_DEFER_OPS defers "
          "eager ops, and returns the tensors of the results, or None when a tensor does not fit; `own_out` "
          "says the tensors of `out` are the call's own, made for its results by a structured kernel");
  py::class_<TypedFunction, std::shared_ptr<TypedFunction>>(module, "_TypedFunction")
      .def(
          py::init([](const py::bytes& function) {
            std::string bytes = function;
            py::gil_scoped_release release;
            return std::make_shared<TypedFunction>(bytes);
          }),
          py::arg("function"),
          "Internal: a compiled function whose args tensors are made in the types of")
      .def(
          "empty",
          &TypedFunction::empty,
          py::arg("arg"),
          py::arg("shape"),
          py::arg("device_index"),
          py::arg("zero"),
          py::call_guard<py::gil_scoped_release>(),
          "Internal: a tensor in the type of arg `arg`, as torch.rbln.empty_typed makes it");
}

} // namespace torch_rbln
