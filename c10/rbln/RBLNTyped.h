#pragma once

#include <c10/core/Device.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/rbln/RBLNRuntime.h>
#include <rebel/v2/runtime/function.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace c10::rbln::typed {

/**
 * @brief The type of arg `arg` of `fn`, which a tensor of `shape` holds its bytes in over a whole
 * allocation, rather than as torch holds a contiguous tensor.
 *
 * An allocation gets its type when it is made and keeps it until it is freed. A program whose arg
 * is of the type binds it in place; copies move its elements to and from tensors of other types
 * on the host. Anything else that reaches its bytes through the caching allocator's `locate` is
 * refused, as it would read them as torch holds a tensor. The type is elementwise (see
 * artifact::elementwise): each element lies where torch holds it, in a dtype of its size.
 */
struct Type {
  std::shared_ptr<const rt::Function> fn;
  size_t arg = 0;
  std::vector<int64_t> shape;
  // artifact::typeId of the arg, which two programs holding the allocation alike share.
  std::string id;
  // A zero element is all zero bytes, so zeroing the bytes zeroes the elements.
  bool zero_is_zero_bytes = false;
};

/**
 * @brief The type of arg `arg` of `fn` for a tensor of `shape`; throws unless the arg holds its
 * value in one shard, takes `shape`, and is elementwise, which the copies of its elements need.
 */
C10_RBLN_API std::shared_ptr<const Type> of(
    std::shared_ptr<const rt::Function> fn,
    size_t arg,
    std::vector<int64_t> shape);

/**
 * @brief The type in brief, for messages: the arg's name, its logical value and its layout.
 */
C10_RBLN_API std::string describe(const Type& type);

/**
 * @brief The bytes an allocation of `type` takes: those of the arg's shard for the type's shape,
 * and at least those torch reads of a contiguous tensor of the shape.
 */
C10_RBLN_API uint64_t nbytes(const Type& type);

/**
 * @brief Whether the device bytes at `a` and at `b` lie in allocations of one type, so that bytes
 * moved between them keep their values wherever they lie.
 */
C10_RBLN_API bool alike(const void* a, const void* b);

/**
 * @brief The values of `count` elements from `bytes`, held in `type`, into `values`, as the logical
 * dtype holds them; `encode` is the inverse. Any elements, wherever they lie, are elements of the
 * type in a row.
 */
C10_RBLN_API void decode(const Type& type, const void* bytes, void* values, size_t count);
C10_RBLN_API void encode(const Type& type, const void* values, void* bytes, size_t count);

/**
 * @brief While alive, this thread locates typed allocations as they are, to move their bytes as
 * the type holds them: the copies between tensors of one type, and those that decode or encode
 * the elements on the host.
 */
class C10_RBLN_API AsTyped {
 public:
  AsTyped();
  ~AsTyped();
  AsTyped(const AsTyped&) = delete;
  AsTyped& operator=(const AsTyped&) = delete;

 private:
  bool before_;
};

} // namespace c10::rbln::typed
