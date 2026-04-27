#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <string>
#include <string_view>
#include <utility>

namespace c10::rbln {

namespace detail {

struct LogSourceLocation {
  const char* file;
  int line;
  const char* function;
};

C10_RBLN_API void log_debug_message(LogSourceLocation location, std::string_view message);
C10_RBLN_API void log_info_message(LogSourceLocation location, std::string_view message);
C10_RBLN_API void log_warn_message(LogSourceLocation location, std::string_view message);
C10_RBLN_API void log_error_message(LogSourceLocation location, std::string_view message);

inline std::string format_log_message() {
  return {};
}

template <typename... Args>
std::string format_log_message(fmt::format_string<Args...> format_string, Args&&... args) {
  return fmt::format(format_string, std::forward<Args>(args)...);
}

template <typename... Args>
void log_debug(LogSourceLocation location, fmt::format_string<Args...> format_string, Args&&... args) {
  log_debug_message(location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_info(LogSourceLocation location, fmt::format_string<Args...> format_string, Args&&... args) {
  log_info_message(location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_warn(LogSourceLocation location, fmt::format_string<Args...> format_string, Args&&... args) {
  log_warn_message(location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_error(LogSourceLocation location, fmt::format_string<Args...> format_string, Args&&... args) {
  log_error_message(location, format_log_message(format_string, std::forward<Args>(args)...));
}

} // namespace detail

/**
 * @brief RAII scope guard for hierarchical function entry/exit logging.
 *
 * Logs [ENTER] on construction and [EXIT] on destruction with box-drawing tree prefixes.
 * Exception-safe: destructor runs on early return or exception.
 */
class C10_RBLN_API RBLNScopeGuard {
 public:
  explicit RBLNScopeGuard(const char* file, int line, const char* func);
  ~RBLNScopeGuard() noexcept;

  RBLNScopeGuard(const RBLNScopeGuard&) = delete;
  RBLNScopeGuard& operator=(const RBLNScopeGuard&) = delete;
  RBLNScopeGuard(RBLNScopeGuard&&) = delete;
  RBLNScopeGuard& operator=(RBLNScopeGuard&&) = delete;

 private:
  const char* file_;
  int line_;
  const char* func_;
};

/**
 * @brief Logs a message indicating that a specified operation ran on CPU instead of RBLN.
 *
 * This function logs an info-level message that includes the name of the operation that ran on CPU.
 * It also generates a UserWarning that includes the file location where the warning is issued. The
 * warning is formatted to indicate that a fallback to CPU execution is being used for the specified
 * operation. The message is logged when `TORCH_RBLN_LOG_LEVEL` is set to `INFO` or lower.
 *
 * @param full_op_name The full name of the operation that ran on CPU.
 *
 * @code
 * c10::rbln::log_cpu_fallback("aten::mul");
 * // Output:
 * // [2026-01-01 00:00:00.000][I] `aten::mul` op ran on CPU instead of RBLN
 * // /llama/modeling_llama.py:73: UserWarning: TRACE
 * //   result, result_shape = mul_rbln(*args, **kwargs)
 * @endcode
 */
C10_RBLN_API void log_cpu_fallback(std::string_view full_op_name);

// Lightweight dispatch tracer. Activated by `TORCH_RBLN_DISPATCH_TRACE=path`.
// Each call appends one TSV line `<pid>\t<bucket>\t<op_name>\n` to that path,
// best-effort, line-atomic via O_APPEND. `bucket` is one of:
//   "wc_hit"        — warm-cache hit (no Python, no fallback)
//   "shim_miss"     — shim native path missed warm-cache → pybind to Python
//   "shim_fallback" — shim quick-precheck triggered C++ cpu_fallback_rbln
//   "generic_fb"    — ATen routed to C++ generic fallback_rbln (non-shim)
C10_RBLN_API void dispatch_trace_emit(const char* bucket, std::string_view op_name);

/**
 * @brief Returns the current scope depth for hierarchical logging.
 *
 * This function returns the current depth of the function call stack for the purpose of hierarchical logging with tree
 * prefixes. The scope depth is incremented on function entry and decremented on function exit by the RBLNScopeGuard.
 *
 * @return The current scope depth as an integer. A value of 0 indicates the top-level scope, 1 indicates one level of
 * nesting, and so on.
 */
C10_RBLN_API int get_scope_depth();

} // namespace c10::rbln

/**
 * @brief Macro for debug-level logging.
 *
 * This macro is enabled only in debug builds.
 *
 * @code
 * RBLN_LOG_DEBUG("Debug message: {}", value);
 * @endcode
 */
#ifdef NDEBUG
#define RBLN_LOG_DEBUG(format_string, ...) ((void)0)
#else
#define RBLN_LOG_DEBUG(format_string, ...)                                                                 \
  do {                                                                                                     \
    c10::rbln::detail::log_debug(                                                                          \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, format_string, ##__VA_ARGS__); \
  } while (0)
#endif

/**
 * @brief Macro for info-level logging.
 *
 * @code
 * RBLN_LOG_INFO("Info message: {}", value);
 * @endcode
 */
#define RBLN_LOG_INFO(format_string, ...)                                                                  \
  do {                                                                                                     \
    c10::rbln::detail::log_info(                                                                           \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, format_string, ##__VA_ARGS__); \
  } while (0)

/**
 * @brief Macro for warning-level logging.
 *
 * @code
 * RBLN_LOG_WARN("Warning message: {}", value);
 * @endcode
 */
#define RBLN_LOG_WARN(format_string, ...)                                                                  \
  do {                                                                                                     \
    c10::rbln::detail::log_warn(                                                                           \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, format_string, ##__VA_ARGS__); \
  } while (0)

/**
 * @brief Macro for error-level logging.
 *
 * @code
 * RBLN_LOG_ERROR("Error message: {}", value);
 * @endcode
 */
#define RBLN_LOG_ERROR(format_string, ...)                                                                 \
  do {                                                                                                     \
    c10::rbln::detail::log_error(                                                                          \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, format_string, ##__VA_ARGS__); \
  } while (0)

/**
 * @brief Macro for checking conditions specific to RBLN.
 *
 * This macro implementation is adapted from the TORCH_CHECK macro defined in pytorch/c10/util/Exception.h to include
 * logging functionality. The functionality of this macro is the same as TORCH_CHECK except that it logs the error
 * message before throwing the exception.
 *
 * @code
 * RBLN_CHECK(condition, "Error message with value: {}", value);
 * @endcode
 */
#define RBLN_CHECK(condition, ...)                                                                                    \
  do {                                                                                                                \
    if (C10_UNLIKELY_OR_CONST(!(condition))) {                                                                        \
      const auto format_ = c10::rbln::detail::format_log_message(__VA_ARGS__);                                        \
      const auto error_ =                                                                                             \
          c10::Error({__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, TORCH_CHECK_MSG(condition, "", format_)); \
      c10::rbln::detail::log_error_message(                                                                           \
          c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, error_.what());                         \
      throw error_;                                                                                                   \
    }                                                                                                                 \
  } while (0)

/**
 * @brief Macro for debug-level checking specific to RBLN.
 *
 * This macro is enabled only in debug builds. It is useful for checks that are too expensive to perform in release
 * builds.
 *
 * @code
 * RBLN_CHECK_DEBUG(condition, "Debug error message with value: {}", value);
 * @endcode
 */
#ifdef NDEBUG
#define RBLN_CHECK_DEBUG(condition, ...) ((void)0)
#else
#define RBLN_CHECK_DEBUG(condition, ...) RBLN_CHECK(condition, ##__VA_ARGS__)
#endif

/**
 * @brief Macro for issuing warnings specific to RBLN.
 *
 * This implementation is adapted from the TORCH_WARN macro defined in pytorch/c10/util/Exception.h to include logging
 * functionality. The functionality of this macro is the same as TORCH_WARN except that it logs the warning message
 * before issuing the warning.
 *
 * @code
 * RBLN_WARN("Warning message with value: {}", value);
 * @endcode
 */
#define RBLN_WARN(...)                                                                       \
  do {                                                                                       \
    const auto format_ = c10::rbln::detail::format_log_message(__VA_ARGS__);                 \
    const auto warning_ = c10::Warning(                                                      \
        c10::UserWarning(),                                                                  \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},                               \
        WARNING_MESSAGE_STRING(format_),                                                     \
        false);                                                                              \
    c10::rbln::detail::log_warn_message(                                                     \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, warning_.msg()); \
    c10::warn(warning_);                                                                     \
  } while (0)

/**
 * @brief Like RBLN_WARN but safe to call from a `noexcept` context.
 *
 * RBLN_WARN expands into calls that can throw (string formatting, c10::warn).
 * Using it inside a catch handler of a `noexcept` function therefore risks
 * std::terminate. Wrap such sites with this macro to swallow any secondary
 * throw originating from the warning path itself.
 */
#define RBLN_WARN_NOTHROW(...) \
  do {                         \
    try {                      \
      RBLN_WARN(__VA_ARGS__);  \
    } catch (...) {            \
    }                          \
  } while (0)

/**
 * @brief Macro for issuing warnings specific to RBLN only once.
 *
 * This macro ensures that the warning is logged and issued only once, regardless of how many times the macro is
 * invoked.
 *
 * @code
 * RBLN_WARN_ONCE("Warning message with value: {}", value);
 * @endcode
 */
#define RBLN_WARN_ONCE(...)                   \
  do {                                        \
    static const auto rbln_warn_once_ = [&] { \
      RBLN_WARN(__VA_ARGS__);                 \
      return true;                            \
    }();                                      \
  } while (0)

/**
 * @brief Macro for automatic function scope logging with hierarchical tree prefix.
 *
 * Creates an RAII scope guard that logs function entry ([ENTER]) and exit ([EXIT])
 * automatically with box-drawing characters for hierarchical visualization.
 * Only active in debug builds.
 *
 * @code
 * void func() {
 *   RBLN_SCOPE_GUARD();
 *   RBLN_LOG_DEBUG("message");
 * }
 *
 * // Output (debug build with TORCH_RBLN_LOG_LEVEL=DEBUG):
 * // [2026-01-01 00:00:00.000][D][12345]┌─[file.cpp:1 (func)] [ENTER]
 * // [2026-01-01 00:00:00.000][D][12345]├─[file.cpp:1 (func)] message
 * // [2026-01-01 00:00:00.000][D][12345]└─[file.cpp:1 (func)] [EXIT]
 * @endcode
 */
#ifdef NDEBUG
#define RBLN_SCOPE_GUARD() ((void)0)
#else
#define RBLN_SCOPE_GUARD() c10::rbln::RBLNScopeGuard rbln_scope_guard_(__FILE__, __LINE__, __func__)
#endif
