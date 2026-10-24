#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// Writes a .safetensors file: the header length as a little-endian u64, a
// JSON header mapping each tensor name to its dtype, shape and byte range
// (plus "__metadata__"), then the tensor bytes in the order added.
// F32 tensors only.
class SafetensorsWriter {
public:
  // Adds a tensor of shape `shape` read from `data` (row-major, contiguous).
  // `data` must stay valid until write() returns.
  void add_f32(std::string name, std::vector<int64_t> shape, const float *data);
  // A "__metadata__" entry; values are strings, as the format requires.
  void set_metadata(std::string key, std::string value);
  // Writes the file; throws on an I/O error or a duplicate tensor name.
  void write(const std::string &path) const;

private:
  struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    const float *data;
  };
  std::vector<Entry> entries_;
  std::vector<std::pair<std::string, std::string>> metadata_;

  // The JSON header, space-padded to a multiple of 8 bytes; throws on a
  // duplicate tensor name.
  std::string header_json() const;
  // The tensor bytes, in the order added.
  void write_tensor_data(std::ostream &out) const;
};
