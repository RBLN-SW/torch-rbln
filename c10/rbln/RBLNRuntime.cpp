#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNRuntime.h>
#include <rebel/v2/runtime/flags.h>

#include <sys/mman.h>

#include <cstdlib>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace c10::rbln {

namespace {

struct Allocation {
  std::shared_ptr<rt::DeviceBuffer> buffer;
  uint64_t nbytes = 0;
  c10::DeviceIndex device_index = -1;
};

// Segments by the start of their handle range; a handle range is reserved without access,
// so the ranges of live segments never overlap.
std::mutex allocations_mutex;
std::map<uintptr_t, Allocation> allocations;

void* reserve_handle(size_t nbytes) {
  void* handle = ::mmap(nullptr, nbytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  RBLN_CHECK(handle != MAP_FAILED, "cannot reserve {} bytes of address space for a device segment", nbytes);
  return handle;
}

std::optional<Location> find(uintptr_t address) {
  std::lock_guard<std::mutex> lock(allocations_mutex);
  auto it = allocations.upper_bound(address);
  if (it == allocations.begin()) {
    return std::nullopt;
  }
  --it;
  const auto offset = address - it->first;
  // One past the end is the end of an empty view at the segment's end.
  if (offset > it->second.nbytes) {
    return std::nullopt;
  }
  return Location{it->second.buffer, offset, it->second.nbytes - offset, it->second.device_index};
}

struct DeviceStreams {
  std::shared_ptr<rt::Stream> default_stream;
  std::vector<std::shared_ptr<rt::Stream>> pool;
};

std::mutex streams_mutex;
std::map<c10::DeviceIndex, DeviceStreams> streams;

} // namespace

std::shared_ptr<rt::Device> runtime_device(c10::DeviceIndex device_index) {
  if (is_dummy_device()) {
    const std::string npu = rt::flags::kForceNpuName.value();
    return rt::Device::openDummy(npu.empty() ? "RBLN-CA25" : npu);
  }
  const auto ids = DeviceMappingManager::getInstance().getPhysicalDeviceIds(device_index);
  RBLN_CHECK(!ids.empty(), "no NPU is mapped to rbln:{}", static_cast<int>(device_index));
  RBLN_CHECK(
      ids.size() == 1,
      "rbln:{} maps {} NPUs; tensors on several NPUs come with functions over them",
      static_cast<int>(device_index),
      ids.size());
  return rt::Device::open(DeviceMappingManager::getInstance().systemNpu(ids.front()));
}

void* allocate_segment(c10::DeviceIndex device_index, size_t nbytes) {
  auto buffer = rt::DeviceBuffer::allocate(runtime_device(device_index), 0, nbytes);
  void* handle = reserve_handle(nbytes);
  std::lock_guard<std::mutex> lock(allocations_mutex);
  allocations.emplace(reinterpret_cast<uintptr_t>(handle), Allocation{std::move(buffer), nbytes, device_index});
  return handle;
}

void release_segment(void* handle) {
  const auto address = reinterpret_cast<uintptr_t>(handle);
  Allocation allocation;
  {
    std::lock_guard<std::mutex> lock(allocations_mutex);
    auto it = allocations.find(address);
    RBLN_CHECK(it != allocations.end(), "{} is no RBLN segment, or it was released already", fmt::ptr(handle));
    allocation = std::move(it->second);
    allocations.erase(it);
  }
  ::munmap(handle, allocation.nbytes);
}

std::optional<Location> locate_segment(const void* ptr) noexcept {
  try {
    return find(reinterpret_cast<uintptr_t>(ptr));
  } catch (...) {
    return std::nullopt;
  }
}

std::shared_ptr<rt::Stream> runtime_stream(c10::Stream stream) {
  const auto device_index = stream.device_index();
  const auto id = stream.id();
  std::lock_guard<std::mutex> lock(streams_mutex);
  auto& entry = streams[device_index];
  if (id == 0) {
    if (!entry.default_stream) {
      entry.default_stream = rt::Stream::defaultFor({runtime_device(device_index)});
    }
    return entry.default_stream;
  }
  RBLN_CHECK(
      id > 0 && static_cast<size_t>(id) <= entry.pool.size(),
      "rbln:{} has no stream {}",
      static_cast<int>(device_index),
      static_cast<int64_t>(id));
  return entry.pool[id - 1];
}

std::vector<std::shared_ptr<rt::Stream>> runtime_streams(c10::DeviceIndex device_index) {
  std::lock_guard<std::mutex> lock(streams_mutex);
  auto& entry = streams[device_index];
  std::vector<std::shared_ptr<rt::Stream>> out;
  if (entry.default_stream) {
    out.push_back(entry.default_stream);
  }
  out.insert(out.end(), entry.pool.begin(), entry.pool.end());
  return out;
}

c10::StreamId add_pool_stream(c10::DeviceIndex device_index) {
  auto stream = std::make_shared<rt::Stream>(std::vector<std::shared_ptr<rt::Device>>{runtime_device(device_index)});
  std::lock_guard<std::mutex> lock(streams_mutex);
  auto& pool = streams[device_index].pool;
  pool.push_back(std::move(stream));
  return static_cast<c10::StreamId>(pool.size());
}

const char* dtype_name(c10::ScalarType type) {
  switch (type) {
    case c10::kHalf:
      return "float16";
    case c10::kBFloat16:
      return "bfloat16";
    case c10::kFloat:
      return "float32";
    case c10::kDouble:
      return "float64";
    case c10::kLong:
      return "int64";
    case c10::kInt:
      return "int32";
    case c10::kShort:
      return "int16";
    case c10::kChar:
      return "int8";
    case c10::kByte:
      return "uint8";
    case c10::kBool:
      return "bool";
    default:
      return nullptr;
  }
}

c10::ScalarType scalar_type_of(const std::string& name) {
  for (auto type :
       {c10::kHalf,
        c10::kBFloat16,
        c10::kFloat,
        c10::kDouble,
        c10::kLong,
        c10::kInt,
        c10::kShort,
        c10::kChar,
        c10::kByte,
        c10::kBool}) {
    if (name == dtype_name(type)) {
      return type;
    }
  }
  throw std::invalid_argument("no torch dtype holds " + name);
}

} // namespace c10::rbln
