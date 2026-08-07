#pragma once

#include <cstdint>
#include <vector>

class StagingMemory {
public:
  void ensure_bytes(uint64_t bytes) {
    if (bytes > buffer_.size()) {
      buffer_.resize(static_cast<size_t>(bytes));
    }
  }

  void *data() { return buffer_.empty() ? nullptr : buffer_.data(); }
  const void *data() const { return buffer_.empty() ? nullptr : buffer_.data(); }
  uint64_t size_bytes() const { return static_cast<uint64_t>(buffer_.size()); }

private:
  std::vector<uint8_t> buffer_;
};
