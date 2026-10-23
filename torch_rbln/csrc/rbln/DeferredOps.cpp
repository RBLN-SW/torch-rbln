#include <torch_rbln/csrc/rbln/DeferredOps.h>

#include <ATen/ops/empty.h>
#include <c10/core/StreamGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNRuntime.h>
#include <c10/util/ScopeExit.h>
#include <rebel/v2/runtime/flags.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace torch_rbln::deferred {

namespace {

namespace py = pybind11;

const rt::Flag<int64_t> kDeferOps(
    "TORCH_RBLN_DEFER_OPS",
    "The most eager ops a stream defers into one function, which runs them once anything else is "
    "queued on the stream or needs their results; 0 runs each op as it is called.",
    0,
    rt::parseCount,
    rt::FlagMutability::kMutable);

// A segment is compiled the second time it runs: one that runs once, as code outside a loop does,
// would pay a compile that costs many times what fusing it saves.
constexpr int kRunsBeforeFusing = 2;

// Whether the runtime runs each operation as it is queued (RBLN_LAUNCH_BLOCKING), which
// deferring would undo.
bool launch_blocking() {
  static const auto& flag = dynamic_cast<const rt::Flag<bool>&>(rt::FlagBase::find("RBLN_LAUNCH_BLOCKING"));
  return flag.value();
}

// What an op of a segment reads: result `index` of op `op`, or input `index` of the segment.
struct Value {
  static constexpr int32_t kInput = -1;
  int32_t op = kInput;
  int32_t index = 0;
  bool operator==(const Value&) const = default;
};

// A tensor the ops of a segment read or write, as they see it: where its bytes lie, and how.
struct Place {
  // A tensor of its own over them, for one the segment reads or a caller gave as `out`; any other
  // result is held through its storage weakly, so that the segment does not write one nothing holds.
  at::Tensor tensor;
  std::optional<c10::weak_intrusive_ptr<c10::StorageImpl>> written_only{};
  const c10::StorageImpl* storage = nullptr;
  uintptr_t begin = 0;
  uintptr_t end = 0;
  int64_t offset = 0;
  std::vector<int64_t> sizes{};
  std::vector<int64_t> strides{};
  at::ScalarType dtype = at::ScalarType::Undefined;
  c10::DeviceIndex device = -1;
  // What an op of the segment last wrote to it.
  std::optional<Value> value;
  // Whether an op of the segment writes it in place, as an input it updates.
  bool updated = false;
  // Its index among the inputs of the segment, when an op reads it before any writes it.
  int32_t input = -1;

  bool live() const {
    return !written_only || !written_only->expired();
  }
  bool written() const {
    return value.has_value() || updated;
  }
  bool holds(const at::Tensor& t) const {
    return storage == t.storage().unsafeGetStorageImpl() && offset == t.storage_offset() && dtype == t.scalar_type() &&
        t.sizes() == c10::IntArrayRef(sizes) && t.strides() == c10::IntArrayRef(strides);
  }
};

Place place_of(const at::Tensor& t) {
  const auto& storage = t.storage();
  const auto begin = reinterpret_cast<uintptr_t>(storage.data());
  return Place{
      .storage = storage.unsafeGetStorageImpl(),
      .begin = begin,
      .end = begin + storage.nbytes(),
      .offset = t.storage_offset(),
      .sizes = t.sizes().vec(),
      .strides = t.strides().vec(),
      .dtype = t.scalar_type(),
      .device = t.device().index()};
}

bool overlap(const Place& a, const Place& b) {
  return a.begin < b.end && b.begin < a.end;
}

// A tensor of its own over the bytes `tensor` views now, which a later change of `tensor`'s view,
// such as set_ or .data, leaves as it is.
at::Tensor snapshot(const at::Tensor& tensor) {
  auto own = at::detail::make_tensor<c10::TensorImpl>(c10::Storage(tensor.storage()), tensor.key_set(), tensor.dtype());
  own.unsafeGetTensorImpl()->set_sizes_and_strides(tensor.sizes(), tensor.strides(), tensor.storage_offset());
  return own;
}

struct Op {
  std::shared_ptr<OpFunction> function;
  std::vector<Value> inputs{};
};

struct Segment {
  std::optional<c10::Stream> stream;
  std::vector<Op> ops;
  std::vector<Place> places;
  std::vector<at::Tensor> inputs;
  // Whether an op updates an input in place, which the segment does whatever it leaves.
  bool updates_inputs = false;
};

constexpr int kNone = -1;
constexpr int kConflict = -2;

// The place of `segment` that holds `tensor` as it is, or kNone; kConflict when a live place shares its
// bytes through another view and either that place or `tensor` is written.
int find(const Segment& segment, const at::Tensor& tensor, bool written) {
  const auto place = place_of(tensor);
  int found = kNone;
  for (size_t i = 0; i < segment.places.size(); ++i) {
    const auto& p = segment.places[i];
    if (!p.live()) {
      continue;
    }
    if (p.holds(tensor)) {
      found = static_cast<int>(i);
    } else if (overlap(p, place) && (written || p.written())) {
      return kConflict;
    }
  }
  return found;
}

// Adds a run of `function` over `inputs` writing `out` to `segment`, and returns its results; none
// when its tensors conflict, as `find` says, with the segment's or with one another.
std::optional<std::vector<at::Tensor>> add(
    Segment& segment,
    const std::shared_ptr<OpFunction>& function,
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out,
    bool own_out,
    c10::DeviceIndex device_index) {
  const auto& updated = function->updated_inputs();
  auto updates = [&updated](size_t i) { return std::ranges::find(updated, i) != updated.end(); };
  std::vector<at::Tensor> written;
  for (const auto& o : out) {
    if (o && o->defined()) {
      written.push_back(*o);
    }
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const int index = find(segment, inputs[i], updates(i));
    // An input updated in place is read as an input of the segment, so no op before writes it.
    if (index == kConflict || (updates(i) && index != kNone && segment.places[index].value)) {
      return std::nullopt;
    }
    if (updates(i)) {
      written.push_back(inputs[i]);
    }
  }
  for (size_t j = 0; j < written.size(); ++j) {
    if (find(segment, written[j], true) == kConflict) {
      return std::nullopt;
    }
    const auto place = place_of(written[j]);
    for (const auto& other : inputs) {
      if (!place.holds(other) && overlap(place, place_of(other))) {
        return std::nullopt;
      }
    }
    for (size_t k = 0; k < written.size(); ++k) {
      if (k != j && overlap(place, place_of(written[k]))) {
        return std::nullopt;
      }
    }
  }

  // The results are made before the segment changes, so that a failed allocation leaves it as it was.
  const auto& outputs = function->outputs();
  std::vector<at::Tensor> results;
  results.reserve(outputs.size());
  for (size_t k = 0; k < outputs.size(); ++k) {
    const bool given = k < out.size() && out[k] && out[k]->defined();
    results.push_back(
        given ? *out[k]
              : at::empty(
                    outputs[k].shape,
                    at::TensorOptions().dtype(outputs[k].dtype).device(c10::DeviceType::PrivateUse1, device_index)));
  }

  Op op{.function = function, .inputs = {}};
  op.inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    int index = find(segment, inputs[i], updates(i));
    if (index == kNone) {
      index = static_cast<int>(segment.places.size());
      segment.places.push_back(place_of(inputs[i]));
      segment.places.back().tensor = snapshot(inputs[i]);
    }
    auto& p = segment.places[index];
    if (p.value) {
      op.inputs.push_back(*p.value);
      continue;
    }
    if (p.input < 0) {
      p.input = static_cast<int32_t>(segment.inputs.size());
      segment.inputs.push_back(p.tensor);
    }
    op.inputs.push_back({Value::kInput, p.input});
    if (updates(i)) {
      p.updated = true;
      segment.updates_inputs = true;
    }
  }
  for (size_t k = 0; k < results.size(); ++k) {
    const bool given = k < out.size() && out[k] && out[k]->defined();
    int index = given ? find(segment, results[k], true) : kNone;
    if (index == kNone) {
      index = static_cast<int>(segment.places.size());
      segment.places.push_back(place_of(results[k]));
      if (given && !own_out) {
        segment.places.back().tensor = snapshot(results[k]);
      } else {
        segment.places.back().written_only = results[k].storage().getWeakStorageImpl();
      }
    }
    segment.places[index].value =
        Value{.op = static_cast<int32_t>(segment.ops.size()), .index = static_cast<int32_t>(k)};
  }
  segment.ops.push_back(std::move(op));
  return results;
}

// A value a segment leaves in a tensor once it has run, and whether the segment reads the tensor.
struct Left {
  Value value;
  at::Tensor tensor;
  bool read = false;
};

// What tells segments apart: their ops, what each reads, and the values they leave.
struct SegmentKey {
  std::vector<const OpFunction*> ops;
  std::vector<Value> reads;
  std::vector<Value> left;
  bool operator==(const SegmentKey&) const = default;
};

struct SegmentKeyHash {
  size_t operator()(const SegmentKey& key) const noexcept {
    size_t h = 0;
    auto mix = [&h](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (const auto* op : key.ops) {
      mix(std::hash<const void*>{}(op));
    }
    for (const auto* values : {&key.reads, &key.left}) {
      mix(values->size());
      for (const auto& v : *values) {
        mix((static_cast<size_t>(static_cast<uint32_t>(v.op)) << 32) | static_cast<uint32_t>(v.index));
      }
    }
    return h;
  }
};

// What the process knows of a segment: how often it ran, and the function it runs as once
// compiled. It holds the functions of its ops, which its key names.
struct Fusion {
  std::vector<std::shared_ptr<OpFunction>> ops{};
  int runs = 0;
  uint64_t used = 0;
  std::shared_ptr<OpFunction> function;
  bool refused = false;
};

std::atomic<uint64_t> stat_deferred{0};
std::atomic<uint64_t> stat_fused_runs{0};
std::atomic<uint64_t> stat_op_runs{0};
std::atomic<uint64_t> stat_compiled{0};
std::atomic<uint64_t> stat_refused{0};

class Fusions {
 public:
  static Fusions& get() {
    // Leaky: fusions hold runtime objects, which must not outlive the runtime's statics at exit.
    static auto* fusions = new Fusions();
    return *fusions;
  }

  // What the fusion of the segment `key` names holds, now that it runs once more.
  struct Seen {
    std::shared_ptr<Fusion> fusion;
    int runs = 0;
    std::shared_ptr<OpFunction> function;
    bool refused = false;
  };
  Seen ran(SegmentKey key, const Segment& segment) {
    std::vector<std::shared_ptr<Fusion>> dropped;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& fusion = map_[std::move(key)];
    if (!fusion) {
      fusion = std::make_shared<Fusion>();
      for (const auto& op : segment.ops) {
        fusion->ops.push_back(op.function);
      }
    }
    ++fusion->runs;
    fusion->used = ++clock_;
    Seen seen{.fusion = fusion, .runs = fusion->runs, .function = fusion->function, .refused = fusion->refused};
    const auto limit = static_cast<size_t>(kCompiledOps.value());
    while (map_.size() > limit) {
      auto oldest = std::ranges::min_element(map_, {}, [](const auto& entry) { return entry.second->used; });
      dropped.push_back(std::move(oldest->second));
      map_.erase(oldest);
    }
    return seen;
  }

  void compiled(Fusion& fusion, std::shared_ptr<OpFunction> function) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (function) {
      fusion.function = std::move(function);
    } else {
      fusion.refused = true;
    }
  }

  void forget(const OpFunction* function) {
    std::vector<std::shared_ptr<Fusion>> dropped;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = map_.begin(); it != map_.end();) {
      if (std::find(it->first.ops.begin(), it->first.ops.end(), function) != it->first.ops.end()) {
        dropped.push_back(std::move(it->second));
        it = map_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void clear() {
    std::unordered_map<SegmentKey, std::shared_ptr<Fusion>, SegmentKeyHash> dropped;
    std::lock_guard<std::mutex> lock(mutex_);
    dropped.swap(map_);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<SegmentKey, std::shared_ptr<Fusion>, SegmentKeyHash> map_;
  uint64_t clock_ = 0;
};

// Locks `lock`, letting go of the GIL while it waits: the thread running a segment may need the
// GIL to compile it.
void lock_without_gil(std::unique_lock<std::mutex>& lock) {
  if (lock.try_lock()) {
    return;
  }
  if (Py_IsInitialized() && PyGILState_Check()) {
    py::gil_scoped_release release;
    lock.lock();
  } else {
    lock.lock();
  }
}

// The function of the ops of `segment`, compiled to leave `left`; none when the compiler refuses.
std::shared_ptr<OpFunction> compile(const Segment& segment, const std::vector<Left>& left) {
  py::gil_scoped_acquire gil;
  py::list functions, reads, values, inputs;
  for (const auto& op : segment.ops) {
    functions.append(py::cast(op.function));
    py::list read;
    for (const auto& v : op.inputs) {
      read.append(py::make_tuple(v.op, v.index));
    }
    reads.append(read);
  }
  for (const auto& l : left) {
    values.append(py::make_tuple(l.value.op, l.value.index));
  }
  for (const auto& input : segment.inputs) {
    inputs.append(py::cast(input));
  }
  auto function = py::module_::import("torch_rbln._internal.compile_cache")
                      .attr("compile_segment")(functions, reads, values, inputs, segment.stream->device_index());
  return function.is_none() ? nullptr : function.cast<std::shared_ptr<OpFunction>>();
}

void copy_unless_same(const at::Tensor& to, const at::Tensor& from) {
  if (!to.is_same(from)) {
    to.copy_(from);
  }
}

void run_fused(OpFunction& function, const Segment& segment, const std::vector<Left>& left) {
  std::vector<std::optional<at::Tensor>> out;
  for (const auto& l : left) {
    out.push_back(l.read ? std::nullopt : std::optional<at::Tensor>(l.tensor));
  }
  auto results = function.run(segment.inputs, out);
  if (!results) {
    results = function.run(segment.inputs);
  }
  TORCH_CHECK(results, "the tensors of a deferred segment do not hold what its function takes");
  // The run is one of each op's work, so the compile cache keeps the ops it runs most.
  for (const auto& op : segment.ops) {
    op.function->note_run();
  }
  for (size_t k = 0; k < left.size(); ++k) {
    copy_unless_same(left[k].tensor, (*results)[k]);
  }
}

void run_each(const Segment& segment, const std::vector<Left>& left) {
  std::map<std::pair<int32_t, int32_t>, const Left*> left_by_value;
  for (const auto& l : left) {
    left_by_value[{l.value.op, l.value.index}] = &l;
  }
  std::vector<std::vector<at::Tensor>> results(segment.ops.size());
  for (size_t j = 0; j < segment.ops.size(); ++j) {
    const auto& op = segment.ops[j];
    std::vector<at::Tensor> inputs;
    inputs.reserve(op.inputs.size());
    for (const auto& v : op.inputs) {
      inputs.push_back(v.op == Value::kInput ? segment.inputs[v.index] : results[v.op][v.index]);
    }
    std::vector<std::optional<at::Tensor>> out(op.function->outputs().size());
    for (size_t k = 0; k < out.size(); ++k) {
      auto it = left_by_value.find({static_cast<int32_t>(j), static_cast<int32_t>(k)});
      if (it != left_by_value.end() && !it->second->read) {
        out[k] = it->second->tensor;
      }
    }
    auto r = op.function->run(inputs, out);
    if (!r) {
      r = op.function->run(inputs);
    }
    TORCH_CHECK(r, "the tensors of a deferred op do not hold what its function takes");
    results[j] = std::move(*r);
  }
  for (const auto& l : left) {
    copy_unless_same(l.tensor, results[l.value.op][l.value.index]);
  }
}

void execute(const Segment& segment) {
  std::vector<Left> left;
  for (const auto& p : segment.places) {
    if (!p.value) {
      continue;
    }
    at::Tensor tensor = p.tensor;
    if (p.written_only) {
      auto storage = p.written_only->lock();
      if (!storage) {
        continue;
      }
      tensor = at::empty({0}, at::TensorOptions().dtype(p.dtype).device(c10::DeviceType::PrivateUse1, p.device));
      tensor.set_(c10::Storage(std::move(storage)), p.offset, p.sizes, p.strides);
    }
    left.push_back({*p.value, std::move(tensor), p.input >= 0});
  }
  // Its ops write nothing but their results and the inputs they update, so a segment that leaves no
  // value anywhere a tensor holds and updates no input changes nothing.
  if (left.empty() && !segment.updates_inputs) {
    return;
  }
  if (segment.ops.size() == 1 || left.empty()) {
    run_each(segment, left);
    stat_op_runs.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  SegmentKey key;
  for (const auto& op : segment.ops) {
    key.ops.push_back(op.function.get());
    key.reads.insert(key.reads.end(), op.inputs.begin(), op.inputs.end());
  }
  for (const auto& l : left) {
    key.left.push_back(l.value);
  }
  auto seen = Fusions::get().ran(std::move(key), segment);
  auto function = seen.function;
  if (!function && !seen.refused && seen.runs >= kRunsBeforeFusing && Py_IsInitialized()) {
    try {
      function = compile(segment, left);
    } catch (...) {
      // A compile cut short, as by KeyboardInterrupt, still leaves the ops to run.
      run_each(segment, left);
      stat_op_runs.fetch_add(1, std::memory_order_relaxed);
      throw;
    }
    Fusions::get().compiled(*seen.fusion, function);
    (function ? stat_compiled : stat_refused).fetch_add(1, std::memory_order_relaxed);
  }
  if (function) {
    run_fused(*function, segment, left);
    stat_fused_runs.fetch_add(1, std::memory_order_relaxed);
  } else {
    run_each(segment, left);
    stat_op_runs.fetch_add(1, std::memory_order_relaxed);
  }
}

struct Queue {
  // Guards `segment`.
  std::mutex mutex;
  // Held while a segment of the queue runs, so that its segments are queued in order.
  std::mutex running;
  Segment segment;
};

// The ops deferred on every queue, which have not run yet.
std::atomic<size_t> deferred_ops{0};

std::mutex queues_mutex;

std::map<std::pair<c10::DeviceIndex, c10::StreamId>, std::unique_ptr<Queue>>& queues() {
  // Leaky: a queue holds tensors, which must not outlive the allocator at exit.
  static auto* q = new std::map<std::pair<c10::DeviceIndex, c10::StreamId>, std::unique_ptr<Queue>>();
  return *q;
}

Queue& queue(c10::Stream stream) {
  std::lock_guard<std::mutex> lock(queues_mutex);
  auto& q = queues()[{stream.device_index(), stream.id()}];
  if (!q) {
    q = std::make_unique<Queue>();
  }
  return *q;
}

// Whether this thread runs a segment, whose own work and any run it submits are not deferred.
thread_local bool running_here = false;

struct RunningHere {
  bool was = std::exchange(running_here, true);
  ~RunningHere() {
    running_here = was;
  }
};

void run_queue(Queue& q) {
  std::unique_lock<std::mutex> running(q.running, std::defer_lock);
  lock_without_gil(running);
  Segment segment;
  {
    std::lock_guard<std::mutex> lock(q.mutex);
    segment = std::exchange(q.segment, Segment{});
  }
  if (segment.ops.empty()) {
    return;
  }
  // The ops count as deferred until they are queued, so that another thread that must queue its work
  // after them waits on `q.running` instead of passing them.
  const auto queued =
      c10::make_scope_exit([n = segment.ops.size()] { deferred_ops.fetch_sub(n, std::memory_order_acq_rel); });
  RunningHere here;
  c10::StreamGuard guard(*segment.stream);
  execute(segment);
}

c10::DeviceIndex device_of(c10::ArrayRef<at::Tensor> inputs) {
  for (const auto& input : inputs) {
    if (input.defined() && input.device().is_privateuseone()) {
      return input.device().index();
    }
  }
  return c10::rbln::get_device_index();
}

} // namespace

std::optional<std::vector<at::Tensor>> submit(
    const std::shared_ptr<OpFunction>& function,
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<std::optional<at::Tensor>> out,
    bool own_out) {
  const int64_t limit = running_here || launch_blocking() ? 0 : kDeferOps.value();
  if (limit == 0 || !function->runs_on_device()) {
    return function->run(inputs, out);
  }
  const auto device_index = device_of(inputs);
  if (!function->binds_in_place(inputs, out, device_index)) {
    return function->run(inputs, out);
  }
  const auto stream = c10::rbln::get_current_stream(device_index);
  auto& q = queue(stream);
  // A run that conflicts with the segment starts the next one; one that conflicts with itself runs as
  // it is called.
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::optional<std::vector<at::Tensor>> results;
    bool full = false;
    {
      std::lock_guard<std::mutex> lock(q.mutex);
      results = add(q.segment, function, inputs, out, own_out, device_index);
      if (results) {
        q.segment.stream = stream;
        deferred_ops.fetch_add(1, std::memory_order_acq_rel);
        full = static_cast<int64_t>(q.segment.ops.size()) >= limit;
      }
    }
    if (results) {
      stat_deferred.fetch_add(1, std::memory_order_relaxed);
      if (full) {
        run_queue(q);
      }
      return results;
    }
    run_queue(q);
  }
  return function->run(inputs, out);
}

void run(c10::DeviceIndex device_index, std::optional<c10::StreamId> stream) {
  if (running_here || deferred_ops.load(std::memory_order_acquire) == 0) {
    return;
  }
  std::vector<Queue*> found;
  {
    std::lock_guard<std::mutex> lock(queues_mutex);
    for (const auto& [key, q] : queues()) {
      if (key.first == device_index && (!stream || key.second == *stream)) {
        found.push_back(q.get());
      }
    }
  }
  for (auto* q : found) {
    run_queue(*q);
  }
}

void run_all() {
  if (running_here || deferred_ops.load(std::memory_order_acquire) == 0) {
    return;
  }
  std::vector<Queue*> found;
  {
    std::lock_guard<std::mutex> lock(queues_mutex);
    for (const auto& [key, q] : queues()) {
      found.push_back(q.get());
    }
  }
  for (auto* q : found) {
    run_queue(*q);
  }
}

void forget(const OpFunction* function) {
  Fusions::get().forget(function);
}

void init_deferred_ops_bindings(pybind11::module& module) {
  c10::rbln::set_deferred_ops_runner(&run);
  module.def(
      "_deferred_forget",
      [](const std::shared_ptr<OpFunction>& function) { forget(function.get()); },
      "Internal: lets go of the functions of the deferred segments `function` runs in",
      py::arg("function"));
  module.def(
      "_deferred_clear", []() { Fusions::get().clear(); }, "Internal: lets go of the functions of deferred segments");
  module.def(
      "_deferred_stats",
      []() {
        return std::map<std::string, uint64_t>{
            {"deferred_ops", stat_deferred.load(std::memory_order_relaxed)},
            {"fused_runs", stat_fused_runs.load(std::memory_order_relaxed)},
            {"op_runs", stat_op_runs.load(std::memory_order_relaxed)},
            {"compiled", stat_compiled.load(std::memory_order_relaxed)},
            {"refused", stat_refused.load(std::memory_order_relaxed)}};
      },
      "DIAG: the eager ops deferred, the segments run as one function and op by op, and those "
      "compiled and refused, since the process started");
}

} // namespace torch_rbln::deferred
