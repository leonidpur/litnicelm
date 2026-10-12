#pragma once

#include <string>

namespace string_utils {
inline std::string trim_copy(const std::string &s) {
  size_t a = 0;
  while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) {
    ++a;
  }
  size_t b = s.size();
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' ||
                   s[b - 1] == '\r')) {
    --b;
  }
  return s.substr(a, b - a);
}

inline std::string unquote_copy(const std::string &raw) {
  const std::string v = trim_copy(raw);
  if (v.size() >= 2) {
    const char a = v.front();
    const char b = v.back();
    if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
      return v.substr(1, v.size() - 2);
    }
  }
  return v;
}
} // namespace string_utils
