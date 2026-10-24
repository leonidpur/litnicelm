#pragma once

#include <string>
#include <string_view>

// JSON string literals.
namespace json_utils {

enum class NonAscii {
  Keep,    // bytes >= 0x7f pass through (UTF-8 text stays readable)
  Escape,  // bytes >= 0x7f become \u00XX (raw bytes, not UTF-8; the file stays ASCII)
};

// The characters of a JSON string literal for s, without the quotes:
// quote and backslash escaped, \n \r \t as short escapes, other control
// characters as \u00XX (lowercase hex).
inline std::string escape(std::string_view s, NonAscii non_ascii = NonAscii::Keep) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(s.size());
  for (const unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20 || (c >= 0x7f && non_ascii == NonAscii::Escape)) {
        out += "\\u00";
        out += kHex[c >> 4];
        out += kHex[c & 0xF];
      } else {
        out += static_cast<char>(c);
      }
    }
  }
  return out;
}

// s as a quoted JSON string literal.
inline std::string quoted(std::string_view s, NonAscii non_ascii = NonAscii::Keep) {
  return "\"" + escape(s, non_ascii) + "\"";
}

} // namespace json_utils
