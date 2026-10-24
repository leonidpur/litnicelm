#pragma once

#include "string_utils.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// Values from text (config files, command-line flags), surrounding
// whitespace ignored. `what` names the value in errors, e.g.
// "key training.batch_size" or "--epochs".
namespace parse_utils {

// 1/true/True/TRUE/yes/on, 0/false/False/FALSE/no/off.
inline bool parse_bool_or_throw(const std::string &v, const std::string &what) {
  const std::string s = string_utils::trim_copy(v);
  if (s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes" || s == "on") {
    return true;
  }
  if (s == "0" || s == "false" || s == "False" || s == "FALSE" || s == "no" || s == "off") {
    return false;
  }
  throw std::runtime_error("invalid bool for " + what + ": " + v);
}

inline uint64_t parse_u64_or_throw(const std::string &v, const std::string &what) {
  try {
    return static_cast<uint64_t>(std::stoull(string_utils::trim_copy(v)));
  } catch (...) {
    throw std::runtime_error("invalid unsigned integer for " + what + ": " + v);
  }
}

inline uint32_t parse_u32_or_throw(const std::string &v, const std::string &what) {
  const uint64_t x = parse_u64_or_throw(v, what);
  if (x > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("u32 overflow for " + what + ": " + v);
  }
  return static_cast<uint32_t>(x);
}

inline int32_t parse_i32_or_throw(const std::string &v, const std::string &what) {
  long x = 0;
  try {
    x = std::stol(string_utils::trim_copy(v));
  } catch (...) {
    throw std::runtime_error("invalid integer for " + what + ": " + v);
  }
  if (x < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
      x > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("i32 overflow for " + what + ": " + v);
  }
  return static_cast<int32_t>(x);
}

inline float parse_f32_or_throw(const std::string &v, const std::string &what) {
  try {
    return std::stof(string_utils::trim_copy(v));
  } catch (...) {
    throw std::runtime_error("invalid float for " + what + ": " + v);
  }
}

} // namespace parse_utils
