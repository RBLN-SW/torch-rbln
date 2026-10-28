include_guard(GLOBAL)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR
      "GCC ${CMAKE_CXX_COMPILER_VERSION} is older than GCC 13, the version "
      "upstream PyTorch builds its wheels with. Build torch-rbln with GCC 13: "
      "CC=gcc-13 CXX=g++-13 on Debian or Ubuntu, or gcc-toolset-13 on RHEL.")
  endif()
  message(STATUS
    "GCC version ${CMAKE_CXX_COMPILER_VERSION} meets requirement (>= 13.0)")
else()
  message(FATAL_ERROR
    "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} is not GCC, the "
    "compiler upstream PyTorch builds its wheels with. Build torch-rbln with "
    "GCC 13: CC=gcc-13 CXX=g++-13 on Debian or Ubuntu, or gcc-toolset-13 "
    "on RHEL.")
endif()
