#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/rbln/RBLNRuntime.h>
#include <rbln/runtime/function.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace c10::rbln::held {

/**
 * @brief How a compiled program holds the value of the tensor over a whole allocation: in the
 * type of arg `arg` of `fn`, as a tensor of `shape`, rather than as torch holds a contiguous tensor.
 *
 * A program whose arg is of that type binds the allocation in place. Whatever else reaches its
 * bytes, through the caching allocator's `locate`, gets them back as torch holds the tensor first.
 */
struct Type {
  std::shared_ptr<const rt::Function> fn;
  size_t arg = 0;
  std::vector<int64_t> shape;
  // artifact::typeId of the arg, which two programs holding the allocation alike share.
  std::string id;
};

/**
 * @brief Puts the allocation starting at `data`, the bytes of a contiguous tensor of `type.shape`
 * as torch holds it, in `type`, in the order of the current stream of its device.
 */
C10_RBLN_API void hold(void* data, std::shared_ptr<const Type> type);

/**
 * @brief Puts the bytes of the allocation starting at `data`, which `type` holds, back as torch
 * holds the tensor, in the order of the current stream of its device. The caching allocator
 * calls it when something locates the bytes, and then counts the allocation as torch holds it.
 */
void release(void* data, const Type& type);

} // namespace c10::rbln::held
