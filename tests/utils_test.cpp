// Shared helpers in liblitnicelm_utils.
//
// hash_utils: FNV-1a 32/64 against the published test vectors, and hashing
// in pieces equals hashing at once.

#include <litnicelm_utils/hash_utils.hpp>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  failures += !ok;
}

std::string hex(uint64_t v) {
  std::ostringstream out;
  out << "0x" << std::hex << v;
  return out.str();
}

void hash_tests() {
  using namespace hash_utils;
  // Test vectors from the FNV reference (isthe.com/chongo/tech/comp/fnv).
  const struct {
    const char *text;
    uint32_t h32;
    uint64_t h64;
  } vectors[] = {
      {"", 0x811c9dc5u, 0xcbf29ce484222325ull},
      {"a", 0xe40c292cu, 0xaf63dc4c8601ec8cull},
      {"foobar", 0xbf9cf968u, 0x85944171f73967e8ull},
  };
  for (const auto &v : vectors) {
    const size_t n = std::strlen(v.text);
    const uint32_t h32 = fnv1a32(kFnv1a32Offset, v.text, n);
    const uint64_t h64 = fnv1a64(kFnv1a64Offset, v.text, n);
    expect(h32 == v.h32, "fnv1a32(\"" + std::string(v.text) + "\") = " + hex(h32));
    expect(h64 == v.h64, "fnv1a64(\"" + std::string(v.text) + "\") = " + hex(h64));
  }
  const std::string text = "the quick brown fox";
  const uint64_t whole = fnv1a64(kFnv1a64Offset, text.data(), text.size());
  const uint64_t pieces = fnv1a64(fnv1a64(kFnv1a64Offset, text.data(), 7), text.data() + 7,
                                  text.size() - 7);
  expect(whole == pieces, "fnv1a64 in two pieces equals at once");
  const uint32_t whole32 = fnv1a32(kFnv1a32Offset, text.data(), text.size());
  const uint32_t pieces32 = fnv1a32(fnv1a32(kFnv1a32Offset, text.data(), 7), text.data() + 7,
                                    text.size() - 7);
  expect(whole32 == pieces32, "fnv1a32 in two pieces equals at once");
}

} // namespace

int main() {
  hash_tests();
  std::cout << (failures == 0 ? "All utils tests passed.\n" : "Utils tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
