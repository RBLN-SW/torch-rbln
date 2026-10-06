#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHeld.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <rbln/artifact/function.h>

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

} // namespace

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
