# Note: spdlog is licensed under the MIT License.
# https://github.com/gabime/spdlog/blob/v1.x/LICENSE

set(DOWNLOAD_EXTRACT_TIMESTAMP ON)
include(FetchContent)

# Use internal {fmt} library to avoid version conflicts between spdlog and
# external {fmt}.
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)

# Disable unnecessary tests, examples, and benchmarks.
set(SPDLOG_BUILD_EXAMPLES OFF)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_BUILD_BENCH OFF)

# Fetch and make spdlog available.
fetchcontent_declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE
)
fetchcontent_getproperties(spdlog)
if(NOT spdlog_POPULATED)
  fetchcontent_populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
