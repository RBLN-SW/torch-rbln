cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(DOWNLOAD_EXTRACT_TIMESTAMP ON)
include(FetchContent)

set(old_cxx_flags ${CMAKE_CXX_FLAGS})

include(TorchSetup)

set(googletest_commit 03597a01ee50ed33e9dfd640b249b4be3799d395)
fetchcontent_declare(
  googletest
  URL https://github.com/google/googletest/archive/${googletest_commit}.zip
)
unset(googletest_commit)
set(INSTALL_GTEST OFF)
fetchcontent_makeavailable(googletest)

set(CMAKE_CXX_FLAGS ${old_cxx_flags})
unset(old_cxx_flags)

include(GoogleTest)
