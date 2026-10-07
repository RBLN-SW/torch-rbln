#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNTyped.h>
#include <rebel/v2/artifact/function.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

namespace c10::rbln::typed {

namespace {

namespace artifact = ::rebel::v2::artifact;

const artifact::Arg& arg_of(const Type& type) {
  return type.fn->artifact().args.at(type.arg);
}

uint64_t logical_nbytes(const artifact::Arg& arg, const std::vector<int64_t>& shape) {
  const auto numel = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
  return static_cast<uint64_t>(numel) * rt::dtypeSize(arg.logical.dtype);
}

// Converts `count` elements a chunk at a time, each chunk in a host tensor of a shape the arg takes:
// whole steps of its dynamic axis, within its range, or its one shape. An elementwise type converts
// each element by itself, so what pads a chunk does not reach the elements kept.
void convert(const Type& type, bool encoding, const void* in, void* out, size_t count) {
  const auto& arg = arg_of(type);
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

std::shared_ptr<const Type> of(std::shared_ptr<const rt::Function> fn, size_t arg, std::vector<int64_t> shape) {
  const auto& a = fn->artifact().args.at(arg);
  RBLN_CHECK(a.shards.size() == 1, "arg {} holds its value in {} shards, not one", a.name, a.shards.size());
  artifact::minNbytes(a, a.shards.front(), shape);
  RBLN_CHECK(
      artifact::elementwise(a),
      "arg {} lays its elements out otherwise than torch ({}), so no copy can move them one by one",
      a.name,
      artifact::toString(a.physical));
  Type type{std::move(fn), arg, std::move(shape), artifact::typeId(a)};
  const auto element = rt::dtypeSize(a.logical.dtype);
  std::vector<uint8_t> zero(element, 0);
  std::vector<uint8_t> bytes(element, 0xff);
  encode(type, zero.data(), bytes.data(), 1);
  type.zero_is_zero_bytes = std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0; });
  return std::make_shared<const Type>(std::move(type));
}

std::string describe(const Type& type) {
  const auto& arg = arg_of(type);
  return arg.name + " " + artifact::toString(artifact::Logical{arg.logical.dtype, type.shape, {}}) + " held as " +
      artifact::toString(arg.physical);
}

uint64_t nbytes(const Type& type) {
  const auto& arg = arg_of(type);
  return std::max(artifact::minNbytes(arg, arg.shards.front(), type.shape), logical_nbytes(arg, type.shape));
}

void decode(const Type& type, const void* bytes, void* values, size_t count) {
  convert(type, false, bytes, values, count);
}

void encode(const Type& type, const void* values, void* bytes, size_t count) {
  convert(type, true, values, bytes, count);
}

bool alike(const void* a, const void* b) {
  if (!caching::any_typed()) {
    return false;
  }
  auto x = caching::try_locate_typed(a);
  auto y = caching::try_locate_typed(b);
  return x && y && x->type && y->type && x->type->id == y->type->id;
}

AsTyped::AsTyped() : before_(caching::locate_as_typed(true)) {}

AsTyped::~AsTyped() {
  caching::locate_as_typed(before_);
}

} // namespace c10::rbln::typed
