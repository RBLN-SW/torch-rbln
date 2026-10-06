#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHeld.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <rbln/artifact/function.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

namespace c10::rbln::held {

namespace {

namespace artifact = ::rbln::artifact;

const artifact::Arg& arg_of(const Type& type) {
  return type.fn->artifact().args.at(type.arg);
}

uint64_t logical_nbytes(const Type& type) {
  const auto numel = std::accumulate(type.shape.begin(), type.shape.end(), int64_t{1}, std::multiplies<>());
  return static_cast<uint64_t>(numel) * rt::dtypeSize(arg_of(type).logical.dtype);
}

// Converts `count` elements a chunk at a time, each chunk in a host tensor of a shape the arg takes:
// whole steps of its dynamic axis, within its range, or its one shape. An elementwise type converts
// each element by itself, so what pads a chunk does not reach the elements kept.
void convert(const Type& type, bool encoding, const void* in, void* out, size_t count) {
  const auto& arg = arg_of(type);
  RBLN_CHECK(type.elementwise, "arg {} lays its elements out otherwise than torch", arg.name);
  const auto element = rt::dtypeSize(arg.logical.dtype);
  auto shape = arg.logical.shape;
  const auto total = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
  const auto* axis = arg.logical.dynamic_axes.empty() ? nullptr : &arg.logical.dynamic_axes.front();
  const auto step = axis ? total / shape[axis->axis] : total;
  std::vector<uint8_t> values;
  for (size_t done = 0; done < count;) {
    if (axis) {
      auto steps = std::max<int64_t>(axis->min, (static_cast<int64_t>(count - done) + step - 1) / step);
      shape[axis->axis] = axis->max ? std::min(steps, axis->max) : steps;
    }
    const auto capacity = static_cast<size_t>(step * (axis ? shape[axis->axis] : 1));
    const auto n = std::min(capacity, count - done);
    auto host = type.fn->emptyHostLike(type.arg, shape);
    auto* bytes = host->shards().front()->data();
    values.assign(capacity * element, 0);
    if (encoding) {
      std::memcpy(values.data(), static_cast<const uint8_t*>(in) + done * element, n * element);
      type.fn->encode(type.arg, {{values.data(), shape}}, *host);
      std::memcpy(static_cast<uint8_t*>(out) + done * element, bytes, n * element);
    } else {
      std::memcpy(bytes, static_cast<const uint8_t*>(in) + done * element, n * element);
      type.fn->decode(type.arg, *host, {values.data(), shape});
      std::memcpy(static_cast<uint8_t*>(out) + done * element, values.data(), n * element);
    }
    done += n;
  }
}

} // namespace

void decode(const Type& type, const void* bytes, void* values, size_t count) {
  convert(type, false, bytes, values, count);
}

void encode(const Type& type, const void* values, void* bytes, size_t count) {
  convert(type, true, values, bytes, count);
}

bool alike(const void* a, const void* b) {
  if (!caching::any_held()) {
    return false;
  }
  auto x = caching::try_locate_held(a);
  auto y = caching::try_locate_held(b);
  return x && y && x->type && y->type && x->type->elementwise && y->type->elementwise && x->type->id == y->type->id;
}

AsHeld::AsHeld() : before_(caching::locate_as_held(true)) {}

AsHeld::~AsHeld() {
  caching::locate_as_held(before_);
}

void hold(void* data, std::shared_ptr<const Type> type) {
  RBLN_CHECK(arg_of(*type).shards.size() == 1, "arg {} holds its value in more than one shard", arg_of(*type).name);
  const Type& to = *type;
  caching::convert_held(
      data,
      [&] {
        std::vector<uint8_t> value(logical_nbytes(to));
        memcpy_v2h(value.data(), data, value.size());
        auto host = to.fn->emptyHostLike(to.arg, to.shape);
        to.fn->encode(to.arg, {{value.data(), to.shape}}, *host);
        const auto& bytes = host->shards().front();
        memcpy_h2v(data, bytes->data(), bytes->nbytes());
      },
      std::move(type));
}

void release(void* data, const Type& type) {
  auto host = type.fn->emptyHostLike(type.arg, type.shape);
  const auto& bytes = host->shards().front();
  memcpy_v2h(bytes->data(), data, bytes->nbytes());
  std::vector<uint8_t> value(logical_nbytes(type));
  type.fn->decode(type.arg, *host, {value.data(), type.shape});
  memcpy_h2v(data, value.data(), value.size());
  prof::record_bounce(prof::BounceSite::kHeldReleased, value.size());
}

} // namespace c10::rbln::held
