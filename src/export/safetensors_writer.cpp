#include "safetensors_writer.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

// JSON string literal: quotes, backslashes and control characters escaped.
std::string json_string(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (c < 0x20) {
      const char *hex = "0123456789abcdef";
      out << "\\u00" << hex[c >> 4] << hex[c & 0xF];
    } else {
      out << c;
    }
  }
  out << '"';
  return out.str();
}

uint64_t byte_count(const std::vector<int64_t> &shape) {
  uint64_t n = sizeof(float);
  for (const int64_t d : shape) {
    n *= static_cast<uint64_t>(d);
  }
  return n;
}

// "__metadata__":{"key":"value",...}
std::string metadata_json(const std::vector<std::pair<std::string, std::string>> &metadata) {
  std::string out = json_string("__metadata__") + ":{";
  for (size_t i = 0; i < metadata.size(); ++i) {
    out += (i ? "," : "") + json_string(metadata[i].first) + ':' +
           json_string(metadata[i].second);
  }
  return out + '}';
}

// "name":{"dtype":"F32","shape":[...],"data_offsets":[begin,end]}
std::string entry_json(const std::string &name, const std::vector<int64_t> &shape,
                       uint64_t begin, uint64_t end) {
  std::ostringstream out;
  out << json_string(name) << ":{\"dtype\":\"F32\",\"shape\":[";
  for (size_t d = 0; d < shape.size(); ++d) {
    out << (d ? "," : "") << shape[d];
  }
  out << "],\"data_offsets\":[" << begin << ',' << end << "]}";
  return out.str();
}

void write_u64_le(std::ostream &out, uint64_t value) {
  unsigned char bytes[8];
  for (int b = 0; b < 8; ++b) {
    bytes[b] = static_cast<unsigned char>(value >> (8 * b));
  }
  out.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

} // namespace

void SafetensorsWriter::add_f32(std::string name, std::vector<int64_t> shape,
                                const float *data) {
  entries_.push_back(Entry{std::move(name), std::move(shape), data});
}

void SafetensorsWriter::set_metadata(std::string key, std::string value) {
  metadata_.emplace_back(std::move(key), std::move(value));
}

std::string SafetensorsWriter::header_json() const {
  std::vector<std::string> fields;
  if (!metadata_.empty()) {
    fields.push_back(metadata_json(metadata_));
  }
  std::set<std::string> names;
  uint64_t offset = 0;
  for (const Entry &e : entries_) {
    if (!names.insert(e.name).second) {
      throw std::runtime_error("SafetensorsWriter: duplicate tensor name " + e.name);
    }
    const uint64_t end = offset + byte_count(e.shape);
    fields.push_back(entry_json(e.name, e.shape, offset, end));
    offset = end;
  }
  std::string text = "{";
  for (size_t i = 0; i < fields.size(); ++i) {
    text += (i ? "," : "") + fields[i];
  }
  text += '}';
  // The format allows trailing spaces in the header; padding it to a
  // multiple of 8 bytes keeps the tensor data 8-byte aligned in the file.
  text.append((8 - text.size() % 8) % 8, ' ');
  return text;
}

void SafetensorsWriter::write_tensor_data(std::ostream &out) const {
  for (const Entry &e : entries_) {
    // F32 little-endian: the host layout on every supported platform.
    out.write(reinterpret_cast<const char *>(e.data),
              static_cast<std::streamsize>(byte_count(e.shape)));
  }
}

void SafetensorsWriter::write(const std::string &path) const {
  const std::string header = header_json();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("SafetensorsWriter: cannot open " + path);
  }
  write_u64_le(out, header.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  write_tensor_data(out);
  if (!out) {
    throw std::runtime_error("SafetensorsWriter: write failed: " + path);
  }
}
