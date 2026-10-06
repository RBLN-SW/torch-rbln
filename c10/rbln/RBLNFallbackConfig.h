#pragma once

#include <c10/rbln/RBLNMacros.h>

#include <stdexcept>
#include <string>

namespace c10::rbln {

/**
 * @brief The error a check raises where `TORCH_RBLN_DISABLE_FALLBACK` disables the fallback it
 * would take; no other fallback, such as running a compiled graph on CPU, takes its place.
 */
class C10_RBLN_API FallbackDisabled : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Checks if a specific fallback category is disabled.
 *
 * Parses the `TORCH_RBLN_DISABLE_FALLBACK` environment variable (comma-separated)
 * and returns true if the given category or 'all' is present.
 *
 * Valid categories: all, compile_error, host_round_trip, non_blocking_copy, strided_copy_error,
 * unsupported_op
 *
 * @param category The fallback category to check.
 * @return true if the category is disabled, false otherwise.
 */
C10_RBLN_API bool is_fallback_disabled(const std::string& category);

} // namespace c10::rbln
