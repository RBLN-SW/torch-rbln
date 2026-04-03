#include <c10/rbln/RBLNLogging.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace c10::rbln {

namespace {

enum class ScopeLogType : uint8_t { NONE, ENTER, EXIT };
thread_local ScopeLogType scope_log_type_ = ScopeLogType::NONE;
thread_local int scope_depth_ = 0;

/**
 * @brief Custom spdlog flag formatter that generates hierarchical tree prefixes
 * based on thread-local scope depth and log type.
 *
 * Tree prefix rules:
 *   ENTER: scope_depth_ x │ + ┌─   (depth is the value BEFORE increment)
 *   EXIT:  scope_depth_ x │ + └─   (depth is the value AFTER decrement)
 *   NONE (inside scope): (scope_depth_ - 1) x │ + ├─
 *   NONE (outside scope): no prefix
 */
class TreePrefixFormatter : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override {
    int pipes = 0;
    std::string_view connector;

    switch (scope_log_type_) {
      case ScopeLogType::ENTER:
        pipes = scope_depth_;
        connector = "\xe2\x94\x8c\xe2\x94\x80"; // ┌─
        break;
      case ScopeLogType::EXIT:
        pipes = scope_depth_;
        connector = "\xe2\x94\x94\xe2\x94\x80"; // └─
        break;
      case ScopeLogType::NONE:
        if (scope_depth_ > 0) {
          pipes = scope_depth_ - 1;
          connector = "\xe2\x94\x9c\xe2\x94\x80"; // ├─
        }
        break;
    }

    if (pipes == 0 && connector.empty()) {
      return;
    }

    static constexpr std::string_view pipe{"\xe2\x94\x82"}; // │
    for (int i = 0; i < pipes; ++i) {
      dest.append(pipe.data(), pipe.data() + pipe.size());
    }
    dest.append(connector.data(), connector.data() + connector.size());
  }

  std::unique_ptr<custom_flag_formatter> clone() const override {
    return spdlog::details::make_unique<TreePrefixFormatter>();
  }
};

spdlog::level::level_enum get_log_level() {
  static const auto log_level = []() {
    const auto* env_level = std::getenv("TORCH_RBLN_LOG_LEVEL");
    const auto* env_legacy = std::getenv("TORCH_RBLN_LOG");

    // `TORCH_RBLN_LOG_LEVEL` takes precedence over deprecated `TORCH_RBLN_LOG`.
    std::string level = "WARNING";
    if (env_level != nullptr) {
      level = std::string(env_level);
    } else if (env_legacy != nullptr) {
      TORCH_WARN_ONCE(
          "The environment variable `TORCH_RBLN_LOG` is deprecated, please use `TORCH_RBLN_LOG_LEVEL` instead");
      level = std::string(env_legacy);
    }

    if (level == "DEBUG") {
      return spdlog::level::debug;
    } else if (level == "INFO") {
      return spdlog::level::info;
    } else if (level == "WARNING") {
      return spdlog::level::warn;
    } else if (level == "ERROR") {
      return spdlog::level::err;
    }
    TORCH_CHECK(false, "Invalid TORCH_RBLN_LOG_LEVEL `", level, "`, expected one of: DEBUG, INFO, WARNING, ERROR");
  }();
  return log_level;
}

std::filesystem::path get_log_file_path() {
  static const auto log_file_path = []() {
    const auto* env = std::getenv("TORCH_RBLN_LOG_PATH");
    return (env != nullptr) ? std::filesystem::path(std::string(env)) : std::filesystem::path("./torch_rbln.log");
  }();
  return log_file_path;
}

std::vector<spdlog::sink_ptr> get_user_sinks() {
  std::vector<spdlog::sink_ptr> user_sinks;

  const auto user_log_level = spdlog::level::info;
  // [date time][level] message
  const std::string user_log_pattern = "[%Y-%m-%d %H:%M:%S.%e][%^%L%$] %v";

  auto user_stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  user_stdout_sink->set_level(user_log_level);
  user_stdout_sink->set_pattern(user_log_pattern);
  user_sinks.push_back(user_stdout_sink);

  return user_sinks;
}

std::unique_ptr<spdlog::pattern_formatter> create_developer_formatter() {
  // [datetime][level][thread]<tree_prefix>[file:line (func)] message
  const std::string developer_log_pattern = "[%Y-%m-%d %H:%M:%S.%e][%^%L%$][%t]%*[%s:%# (%!)] %v";

  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<TreePrefixFormatter>('*');
  formatter->set_pattern(developer_log_pattern);
  return formatter;
}

std::vector<spdlog::sink_ptr> get_developer_sinks() {
  std::vector<spdlog::sink_ptr> developer_sinks;

  const auto developer_log_level = spdlog::level::debug;

  auto developer_stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  developer_stdout_sink->set_level(developer_log_level);
  developer_stdout_sink->set_formatter(create_developer_formatter());
  developer_sinks.push_back(developer_stdout_sink);

  // Log file is always created in debug builds.
  // Its path can be configured via `TORCH_RBLN_LOG_PATH` environment variable.
  const auto log_file_path = get_log_file_path();
  auto developer_file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);
  developer_file_sink->set_level(developer_log_level);
  developer_file_sink->set_formatter(create_developer_formatter());
  developer_sinks.push_back(developer_file_sink);

  return developer_sinks;
}

} // namespace

class RBLNLogger final {
 public:
  static RBLNLogger& get_instance() {
    static RBLNLogger instance;
    return instance;
  }

  void debug(detail::LogSourceLocation location, std::string_view message) {
    log(spdlog::level::debug, location, message);
  }

  void info(detail::LogSourceLocation location, std::string_view message) {
    log(spdlog::level::info, location, message);
  }

  void warn(detail::LogSourceLocation location, std::string_view message) {
    log(spdlog::level::warn, location, message);
  }

  void error(detail::LogSourceLocation location, std::string_view message) {
    log(spdlog::level::err, location, message);
  }

 private:
  RBLNLogger() {
    std::vector<spdlog::sink_ptr> sinks;
#ifdef NDEBUG
    sinks = get_user_sinks();
#else
    sinks = get_developer_sinks();
#endif

    logger_ = std::make_shared<spdlog::logger>("torch-rbln", sinks.begin(), sinks.end());
    const auto log_level = get_log_level();
    logger_->set_level(log_level);
    logger_->flush_on(spdlog::level::err);
  }

  void log(spdlog::level::level_enum level, detail::LogSourceLocation location, std::string_view message) {
    logger_->log(spdlog::source_loc{location.file, location.line, location.function}, level, "{}", message);
  }

  std::shared_ptr<spdlog::logger> logger_;
};

// NOLINTBEGIN(misc-use-internal-linkage) -- definitions for C10_RBLN_API declarations in RBLNLogging.h
namespace detail {

void log_debug_message(LogSourceLocation location, std::string_view message) {
#ifdef NDEBUG
  (void)location;
  (void)message;
#else
  RBLNLogger::get_instance().debug(location, message);
#endif
}

void log_info_message(LogSourceLocation location, std::string_view message) {
  RBLNLogger::get_instance().info(location, message);
}

void log_warn_message(LogSourceLocation location, std::string_view message) {
  RBLNLogger::get_instance().warn(location, message);
}

void log_error_message(LogSourceLocation location, std::string_view message) {
  RBLNLogger::get_instance().error(location, message);
}

} // namespace detail
// NOLINTEND(misc-use-internal-linkage)

namespace {
void log_scope_marker(detail::LogSourceLocation location, std::string_view marker) {
  detail::log_debug_message(location, marker);
}
} // namespace

RBLNScopeGuard::RBLNScopeGuard(const char* file, int line, const char* func) : file_(file), line_(line), func_(func) {
  scope_log_type_ = ScopeLogType::ENTER;
  try {
    log_scope_marker(detail::LogSourceLocation{file_, line_, func_}, "[ENTER]");
  } catch (...) {
    scope_log_type_ = ScopeLogType::NONE;
    throw;
  }
  scope_log_type_ = ScopeLogType::NONE;
  scope_depth_++;
}

RBLNScopeGuard::~RBLNScopeGuard() noexcept {
  scope_depth_--;
  try {
    scope_log_type_ = ScopeLogType::EXIT;
    log_scope_marker(detail::LogSourceLocation{file_, line_, func_}, "[EXIT]");
  } catch (...) {
    (void)0;
  }
  scope_log_type_ = ScopeLogType::NONE;
}

// NOLINTNEXTLINE(misc-use-internal-linkage) -- C10_RBLN_API in RBLNLogging.h
void log_cpu_fallback(std::string_view full_op_name) {
  detail::log_info_message(
      detail::LogSourceLocation{__FILE__, __LINE__, __func__},
      std::string("`") + std::string(full_op_name) + "` op ran on CPU instead of RBLN");

  const auto log_level = get_log_level();
  if (log_level <= spdlog::level::info) {
    // Emit a trace marker via Python warnings.
    const auto warning_ = c10::Warning(
        c10::UserWarning(),
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
        WARNING_MESSAGE_STRING("TRACE"),
        false);
    detail::log_warn_message(detail::LogSourceLocation{__FILE__, __LINE__, __func__}, warning_.msg());
    c10::warn(warning_);
  }
}

int get_scope_depth() {
  return scope_depth_;
}

} // namespace c10::rbln
