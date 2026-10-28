# Note: spdlog is licensed under the MIT License.
# https://github.com/gabime/spdlog/blob/v1.x/LICENSE

include_guard(GLOBAL)
include(FetchContent)

# Use internal {fmt} library to avoid version conflicts between spdlog and
# external {fmt}.
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)

# Disable unnecessary tests, examples, and benchmarks.
set(SPDLOG_BUILD_EXAMPLES OFF)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_BUILD_BENCH OFF)

fetchcontent_declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH
    SHA256=d8862955c6d74e5846b3f580b1605d2428b11d97a410d86e2fb13e857cd3a744
  EXCLUDE_FROM_ALL
)
fetchcontent_makeavailable(spdlog)
