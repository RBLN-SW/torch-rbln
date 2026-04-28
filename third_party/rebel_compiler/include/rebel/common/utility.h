#ifndef REBEL_COMMON_UTILITY_H
#define REBEL_COMMON_UTILITY_H

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace rbln {

template <typename E>
constexpr auto to_underlying(E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}

template <typename T>
inline std::string ToString(std::vector<T> v) {
  std::ostringstream oss;
  if (!v.empty()) {
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(oss, ","));
    oss << v.back();
  }
  return oss.str();
}

template <typename T>
inline std::string GetShapeStr(std::vector<T> v) {
  std::stringstream oss;
  if (!v.empty()) {
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(oss, ","));
    oss << v.back();
  }
  return oss.str();
}

template <typename T>
std::string HexStringFrom(const std::vector<T>& vector) {
  static_assert(std::is_integral_v<T>);
  static_assert(sizeof(T) == 1);
  std::string hex(vector.size() * 2, ' ');
  char hexchars[] = "0123456789abcdef";
  for (size_t i = 0; i < vector.size(); i++) {
    hex[i * 2] = hexchars[(vector[i] >> 4) & 0xf];
    hex[i * 2 + 1] = hexchars[vector[i] & 0xf];
  }
  return hex;
}

}  // namespace rbln

#endif  // REBEL_COMMON_UTILITY_H
