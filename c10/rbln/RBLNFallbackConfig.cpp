#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/StringUtil.h>
#include <c10/util/env.h>

#include <set>
#include <unordered_set>

namespace c10::rbln {

namespace {

const std::set<std::string> kValidOptions = {"all", "compile_error", "non_blocking_copy", "unsupported_op"};

} // namespace

// TODO: Introduce fallback manager to optimize lookup instead of parsing env var on every check.
bool is_fallback_disabled(const std::string& category) {
  const auto env_opt = c10::utils::get_env("TORCH_RBLN_DISABLE_FALLBACK");
  if (!env_opt.has_value()) {
    return false;
  }

  const auto& env = env_opt.value();
  std::unordered_set<std::string> disabled_fallbacks;
  for (auto token_view : c10::split(env, ',')) {
    auto start = token_view.find_first_not_of(" \t");
    if (start == std::string_view::npos) {
      continue;
    }
    auto end = token_view.find_last_not_of(" \t");
    auto token = std::string(token_view.substr(start, end - start + 1));

    RBLN_CHECK(
        kValidOptions.count(token) > 0,
        "Invalid TORCH_RBLN_DISABLE_FALLBACK option `{}`, expected one of: {}",
        token,
        fmt::join(kValidOptions, ","));

    disabled_fallbacks.insert(std::move(token));
  }

  const bool fallback_disabled = ((disabled_fallbacks.count("all") > 0) || (disabled_fallbacks.count(category) > 0));
  RBLN_LOG_DEBUG(
      "'{}' fallback: {} (TORCH_RBLN_DISABLE_FALLBACK='{}')",
      category,
      (fallback_disabled ? "disabled" : "enabled"),
      env);
  return fallback_disabled;
}

} // namespace c10::rbln
