#pragma once

#include <cstddef>
#include <cstdint>

// FNV-1a hashes. Each call continues from `hash` (start from the offset
// basis, or from a previous result to hash data in pieces).
namespace hash_utils {

inline constexpr uint32_t kFnv1a32Offset = 2166136261u;
inline constexpr uint64_t kFnv1a64Offset = 14695981039346656037ull;

inline uint32_t fnv1a32(uint32_t hash, const void *data, size_t len) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

inline uint64_t fnv1a64(uint64_t hash, const void *data, size_t len) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

} // namespace hash_utils
