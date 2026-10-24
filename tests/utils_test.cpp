// Shared helpers in liblitnicelm_utils.
//
// hash_utils: FNV-1a 32/64 against the published test vectors, and hashing
// in pieces equals hashing at once.
// view_geometry: block counts and byte offsets for every suffix rank of
// contiguous, column-sliced, column-major and size-1-axis views against a
// brute-force walk of the index tuples; rank 0; errors.

#include <litnicelm_utils/hash_utils.hpp>
#include <litnicelm_utils/view_geometry.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

TensorViewStruct make_view(std::vector<int64_t> dims, std::vector<int64_t> strides) {
  TensorViewStruct v{};
  v.ndim = static_cast<uint32_t>(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    v.dims[i] = dims[i];
    v.strides_bytes[i] = strides[i];
  }
  return v;
}

// Byte offsets of every block of the trailing suffix_rank axes, in
// row-major order of the leading axes, by enumerating the index tuples.
std::vector<int64_t> brute_force_offsets(const TensorViewStruct &v, size_t suffix_rank) {
  const size_t lead = v.ndim - suffix_rank;
  std::vector<int64_t> idx(lead, 0);
  std::vector<int64_t> out;
  while (true) {
    int64_t offset = 0;
    for (size_t a = 0; a < lead; ++a) {
      offset += idx[a] * v.strides_bytes[a];
    }
    out.push_back(offset);
    size_t a = lead;
    while (a > 0) {
      --a;
      if (++idx[a] < v.dims[a]) {
        break;
      }
      idx[a] = 0;
      if (a == 0) {
        return out;
      }
    }
    if (lead == 0) {
      return out;
    }
  }
}

bool throws(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

void view_geometry_tests() {
  using namespace view_geometry;
  const struct {
    const char *name;
    TensorViewStruct view;
  } cases[] = {
      {"contiguous F32 [2,3,4]", make_view({2, 3, 4}, {48, 16, 4})},
      {"column slice [2,3,4] of [2,3,8]", make_view({2, 3, 4}, {96, 32, 4})},
      {"column-major [3,2]", make_view({3, 2}, {4, 12})},
      {"size-1 axes [1,5,1,2]", make_view({1, 5, 1, 2}, {999, 16, 777, 4})},
      {"rank 1 [7]", make_view({7}, {2})},
  };
  for (const auto &c : cases) {
    bool ok = true;
    for (size_t suffix = 0; suffix <= c.view.ndim; ++suffix) {
      const std::vector<int64_t> want = brute_force_offsets(c.view, suffix);
      ok = ok && prefix_count(c.view, suffix) == want.size();
      for (uint64_t p = 0; p < want.size(); ++p) {
        ok = ok && prefix_byte_offset(c.view, p, suffix) == want[p];
      }
    }
    expect(ok, std::string("view_geometry: ") + c.name +
                   ": counts and offsets match for every suffix rank");
  }
  const TensorViewStruct scalar = make_view({}, {});
  expect(prefix_count(scalar, 0) == 1 && prefix_byte_offset(scalar, 0, 0) == 0,
         "view_geometry: rank 0 is one block at offset 0");
  const TensorViewStruct m = make_view({2, 3}, {12, 4});
  expect(throws([&] { prefix_count(m, 3); }) &&
             throws([&] { prefix_byte_offset(m, 0, 3); }),
         "view_geometry: suffix rank above the view's rank is rejected");
  const TensorViewStruct empty = make_view({0, 3}, {12, 4});
  expect(prefix_count(empty, 1) == 0 && throws([&] { prefix_byte_offset(empty, 0, 1); }),
         "view_geometry: a zero dimension holds no blocks, and its offsets are rejected");
}

} // namespace

int main() {
  hash_tests();
  view_geometry_tests();
  std::cout << (failures == 0 ? "All utils tests passed.\n" : "Utils tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
