#pragma once

#include <c10/rbln/RBLNMacros.h>

#include <string>

namespace c10::rbln {

/**
 * @brief Checks if a specific fallback category is disabled.
 *
 * Parses the `TORCH_RBLN_DISABLE_FALLBACK` environment variable (comma-separated)
 * and returns true if the given category or 'all' is present.
 *
 * Valid categories: all, compile_error, non_blocking_copy, unsupported_op
 *
 * @param category The fallback category to check.
 * @return true if the category is disabled, false otherwise.
 */
C10_RBLN_API bool is_fallback_disabled(const std::string& category);

} // namespace c10::rbln
