#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

// Numbers as the shortest text that reads back as the same value.
namespace format_utils {

// %g with 6 to 9 significant digits, the fewest that read back as v:
// 0.0006f prints as 0.0006, not its double expansion 0.000600000028.
inline std::string float_text(float v) {
  char buf[64];
  for (int precision = 6; precision <= 9; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, static_cast<double>(v));
    if (std::strtof(buf, nullptr) == v) {
      break;
    }
  }
  return buf;
}

// %g with 15 to 17 significant digits, the fewest that read back as v.
inline std::string double_text(double v) {
  char buf[64];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    if (std::strtod(buf, nullptr) == v) {
      break;
    }
  }
  return buf;
}

} // namespace format_utils
