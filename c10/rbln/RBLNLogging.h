#pragma once

#include <c10/rbln/RBLNMacros.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace c10::rbln {

namespace detail {

struct LogSourceLocation {
  const char* file;
  int line;
  const char* function;
};

C10_RBLN_API void log_debug_message(
    LogSourceLocation location,
    std::string_view message);
C10_RBLN_API void log_info_message(
    LogSourceLocation location,
    std::string_view message);
C10_RBLN_API void log_warn_message(
    LogSourceLocation location,
    std::string_view message);
C10_RBLN_API void log_error_message(
    LogSourceLocation location,
    std::string_view message);

inline std::string format_log_message() {
  return {};
}

template <typename... Args>
std::string format_log_message(
    fmt::format_string<Args...> format_string,
    Args&&... args) {
  return fmt::format(format_string, std::forward<Args>(args)...);
}

template <typename... Args>
void log_debug(
    LogSourceLocation location,
    fmt::format_string<Args...> format_string,
    Args&&... args) {
  log_debug_message(
      location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_info(
    LogSourceLocation location,
    fmt::format_string<Args...> format_string,
    Args&&... args) {
  log_info_message(
      location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_warn(
    LogSourceLocation location,
    fmt::format_string<Args...> format_string,
    Args&&... args) {
  log_warn_message(
      location, format_log_message(format_string, std::forward<Args>(args)...));
}

template <typename... Args>
void log_error(
    LogSourceLocation location,
    fmt::format_string<Args...> format_string,
    Args&&... args) {
  log_error_message(
      location, format_log_message(format_string, std::forward<Args>(args)...));
}

} // namespace detail

/**
 * @brief RAII scope guard for hierarchical function entry/exit logging.
 *
 * Logs [ENTER] on construction and [EXIT] on destruction with box-drawing tree
 * prefixes. Exception-safe: destructor runs on early return or exception.
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

// Where a CPU fallback ran. kBoxed is cpu_fallback_rbln, the boxed CPU
// fallback and the only path that consults CPU fast-path handlers. kPython is
// the eager Python path, which runs the op through PyTorch's own CPU kernels.
// kStridedCopy is submit_or_fallback, which redoes a batched strided copy the
// device rejected with the kernel's own CPU version of the op.
enum class CpuFallbackPath : uint8_t { kBoxed, kPython, kStridedCopy };

/**
 * @brief Marks one CPU fallback: an operation running on CPU instead of RBLN.
 *
 * Every torch-rbln op-level CPU fallback runs inside one of these:
 * cpu_fallback_rbln (which the C++ dispatch shim, the boxed fallback and
 * advanced indexing all call), the eager Python path through the
 * `_CpuFallbackScope` context manager, and the CPU retry of a batched strided
 * copy the device rejected (submit_or_fallback). Construction counts the
 * fallback per op (cpu_fallback_counts, read by torch.rbln.explain()) and, when
 * `TORCH_RBLN_LOG_LEVEL` is `INFO` or lower, logs it with a UserWarning that
 * carries the caller's location. Destruction adds the scope's wall time to
 * cpu_fallback_ns() when it is the outermost scope on the thread: a fallback
 * that runs inside another (an SDPA fallback converting its mask on the
 * device, say) is counted, but its time is already part of the outer one.
 *
 * @code
 * {
 *   const CpuFallbackScope fallback(
 *       "aten::mul.out", CpuFallbackPath::kBoxed, "nan_inf");
 *   // ... run the op on CPU ...
 * }
 * // Output (at INFO):
 * // [2026-01-01 00:00:00.000][I] `aten::mul.out` op ran on CPU instead of RBLN
 * // /llama/modeling_llama.py:73: UserWarning: TRACE
 * //   result, result_shape = mul_rbln(*args, **kwargs)
 * @endcode
 */
class C10_RBLN_API CpuFallbackScope {
 public:
  // `reason` names the check that sent the op to CPU, such as the
  // `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` case or
  // `TORCH_RBLN_DISABLE_FALLBACK` category that decided it.
  CpuFallbackScope(
      std::string_view full_op_name,
      CpuFallbackPath path,
      std::string_view reason);
  ~CpuFallbackScope();
  CpuFallbackScope(const CpuFallbackScope&) = delete;
  CpuFallbackScope& operator=(const CpuFallbackScope&) = delete;
  CpuFallbackScope(CpuFallbackScope&&) = delete;
  CpuFallbackScope& operator=(CpuFallbackScope&&) = delete;

 private:
  bool outermost_;
  uint64_t start_ns_;
};

// Per-op counts of CPU fallbacks, non-zero only.
C10_RBLN_API std::vector<std::pair<std::string, uint64_t>> cpu_fallback_counts();
// The same counts split by reason, as (op, reason, count); an op's entries sum
// to its cpu_fallback_counts entry.
C10_RBLN_API std::vector<std::tuple<std::string, std::string, uint64_t>>
cpu_fallback_reason_counts();
// Wall time of the CPU fallbacks those counts cover; nested fallbacks are
// timed once, as part of the outermost.
C10_RBLN_API uint64_t cpu_fallback_ns();
// Clears the counts and the wall time.
C10_RBLN_API void reset_cpu_fallback_counts();
// Names among cpu_fallback_counts counted at least once through
// CpuFallbackPath::kBoxed: the fallbacks a CPU fast-path handler could serve.
C10_RBLN_API std::vector<std::string> boxed_cpu_fallback_ops();

/**
 * @brief Returns the current scope depth for hierarchical logging.
 *
 * This function returns the current depth of the function call stack for the
 * purpose of hierarchical logging with tree prefixes. The scope depth is
 * incremented on function entry and decremented on function exit by the
 * RBLNScopeGuard.
 *
 * @return The current scope depth as an integer. A value of 0 indicates the
 *     top-level scope, 1 indicates one level of nesting, and so on.
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
#define RBLN_LOG_DEBUG(format_string, ...)                                  \
  do {                                                                      \
    c10::rbln::detail::log_debug(                                           \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, \
        format_string,                                                      \
        ##__VA_ARGS__);                                                     \
  } while (0)
#endif

/**
 * @brief Macro for info-level logging.
 *
 * @code
 * RBLN_LOG_INFO("Info message: {}", value);
 * @endcode
 */
#define RBLN_LOG_INFO(format_string, ...)                                   \
  do {                                                                      \
    c10::rbln::detail::log_info(                                            \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, \
        format_string,                                                      \
        ##__VA_ARGS__);                                                     \
  } while (0)

/**
 * @brief Macro for warning-level logging.
 *
 * @code
 * RBLN_LOG_WARN("Warning message: {}", value);
 * @endcode
 */
#define RBLN_LOG_WARN(format_string, ...)                                   \
  do {                                                                      \
    c10::rbln::detail::log_warn(                                            \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, \
        format_string,                                                      \
        ##__VA_ARGS__);                                                     \
  } while (0)

/**
 * @brief Macro for error-level logging.
 *
 * @code
 * RBLN_LOG_ERROR("Error message: {}", value);
 * @endcode
 */
#define RBLN_LOG_ERROR(format_string, ...)                                  \
  do {                                                                      \
    c10::rbln::detail::log_error(                                           \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__}, \
        format_string,                                                      \
        ##__VA_ARGS__);                                                     \
  } while (0)

/**
 * @brief Macro for checking conditions specific to RBLN.
 *
 * This macro implementation is adapted from the TORCH_CHECK macro defined in
 * pytorch/c10/util/Exception.h to include logging functionality. The
 * functionality of this macro is the same as TORCH_CHECK except that it logs
 * the error message before throwing the exception.
 *
 * @code
 * RBLN_CHECK(condition, "Error message with value: {}", value);
 * @endcode
 */
#define RBLN_CHECK(condition, ...)                                             \
  do {                                                                         \
    if (C10_UNLIKELY_OR_CONST(!(condition))) {                                 \
      const auto format_ = c10::rbln::detail::format_log_message(__VA_ARGS__); \
      auto error_ = c10::Error(                                                \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},               \
          TORCH_CHECK_MSG(condition, "", format_));                            \
      c10::rbln::detail::log_error_message(                                    \
          c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__},  \
          error_.what());                                                      \
      throw std::move(error_);                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief RBLN_CHECK that throws without logging. Same message, no console
 * output.
 *
 * For failures a probe may legitimately reach -- device-mapping validation,
 * called from is_available()/device_count(). RBLN_CHECK logs c10::Error::what()
 * (stack trace included) to stdout before throwing, ungated by RBLN_LOG_LEVEL,
 * so catching the exception does not suppress it. The message rides the
 * exception instead.
 *
 * Consequence: a malformed config stays quiet at the point of use too, since
 * that rethrows the same stored plan error. get_device_count_nothrow() warns
 * once for callers that swallow it. Use errors (unassigned device index, bad
 * allocation) keep RBLN_CHECK.
 *
 * @code
 * RBLN_CHECK_QUIET(condition, "Error message with value: {}", value);
 * @endcode
 */
#define RBLN_CHECK_QUIET(condition, ...)                                       \
  do {                                                                         \
    if (C10_UNLIKELY_OR_CONST(!(condition))) {                                 \
      const auto format_ = c10::rbln::detail::format_log_message(__VA_ARGS__); \
      throw c10::Error(                                                        \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},               \
          TORCH_CHECK_MSG(condition, "", format_));                            \
    }                                                                          \
  } while (0)

/**
 * @brief Macro for debug-level checking specific to RBLN.
 *
 * This macro is enabled only in debug builds. It is useful for checks that are
 * too expensive to perform in release builds.
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
 * This implementation is adapted from the TORCH_WARN macro defined in
 * pytorch/c10/util/Exception.h to include logging functionality. The
 * functionality of this macro is the same as TORCH_WARN except that it logs the
 * warning message before issuing the warning.
 *
 * @code
 * RBLN_WARN("Warning message with value: {}", value);
 * @endcode
 */
#define RBLN_WARN(...)                                                       \
  do {                                                                       \
    const auto format_ = c10::rbln::detail::format_log_message(__VA_ARGS__); \
    const auto warning_ = c10::Warning(                                      \
        c10::UserWarning(),                                                  \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},               \
        WARNING_MESSAGE_STRING(format_),                                     \
        false);                                                              \
    c10::rbln::detail::log_warn_message(                                     \
        c10::rbln::detail::LogSourceLocation{__FILE__, __LINE__, __func__},  \
        warning_.msg());                                                     \
    c10::warn(warning_);                                                     \
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
 * This macro ensures that the warning is logged and issued only once,
 * regardless of how many times the macro is invoked.
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
 * @brief Macro for automatic function scope logging with hierarchical tree
 * prefix.
 *
 * Creates an RAII scope guard that logs function entry ([ENTER]) and exit
 * ([EXIT]) automatically with box-drawing characters for hierarchical
 * visualization. Only active in debug builds.
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
#define RBLN_SCOPE_GUARD() \
  c10::rbln::RBLNScopeGuard rbln_scope_guard_(__FILE__, __LINE__, __func__)
#endif
