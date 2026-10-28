include_guard(GLOBAL)
include(FetchContent)

fetchcontent_declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
  URL_HASH
    SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
)
set(INSTALL_GTEST OFF)
fetchcontent_makeavailable(googletest)

include(GoogleTest)
