include_guard(GLOBAL)

find_package(Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED
)
if(NOT DEFINED PYTHON_SITE_PACKAGES)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PYTHON_SITE_PACKAGES)
    message(FATAL_ERROR "Failed to get Python site packages")
  endif()
endif()

find_package(Torch REQUIRED
  PATHS
    ${PYTHON_SITE_PACKAGES}/torch/share/cmake/Torch
    ${PYTHON_SITE_PACKAGES}/torch/share/cmake
  NO_DEFAULT_PATH
)

# Set runtime paths for `torch`
list(APPEND CMAKE_BUILD_RPATH ${PYTHON_SITE_PACKAGES}/torch/lib)
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../torch/lib")

# Create a symlink to the `torch` directory next to the install prefix
install(CODE "execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${PYTHON_SITE_PACKAGES}/torch torch
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/..
)")
