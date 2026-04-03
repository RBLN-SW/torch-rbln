cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(DOWNLOAD_EXTRACT_TIMESTAMP ON)
include(FetchContent)

set(old_cxx_flags ${CMAKE_CXX_FLAGS})

include(FindTorch)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_FLAGS ${old_cxx_flags})
unset(old_cxx_flags)

include(GoogleTest)
