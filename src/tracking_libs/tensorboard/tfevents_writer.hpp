#pragma once

// Minimal TensorBoard event-file writer (scalars only), with no TensorFlow or
// protobuf dependency.
//
// File: a sequence of records
//   uint64 length (little endian)
//   uint32 masked_crc32c(length bytes)
//   byte   data[length]            serialized tensorflow.Event
//   uint32 masked_crc32c(data)
//
// Event fields used: wall_time (1, double), step (2, int64),
// file_version (3, string), summary (5, Summary). Summary.value (1) holds
// Value{tag (1, string), simple_value (2, float)}.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "tfevents writes fixed-width fields in host order");

namespace tfevents {

// CRC-32C (Castagnoli, reflected polynomial 0x82F63B78).
inline uint32_t crc32c(const void *data, size_t len) {
  static const struct Table {
    uint32_t t[256];
    Table() {
      for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
          c = (c & 1u) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        }
        t[i] = c;
      }
    }
  } table;
  const auto *p = static_cast<const uint8_t *>(data);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc = table.t[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

inline uint32_t masked_crc32c(const void *data, size_t len) {
  const uint32_t crc = crc32c(data, len);
  return ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
}

// Protobuf wire encoding helpers.
inline void put_varint(std::string &out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}

inline void put_tag(std::string &out, uint32_t field, uint32_t wire_type) {
  put_varint(out, (static_cast<uint64_t>(field) << 3) | wire_type);
}

inline void put_fixed(std::string &out, const void *value, size_t size) {
  // Protobuf fixed32/fixed64 and the record header are little endian.
  out.append(static_cast<const char *>(value), size);
}

inline void put_bytes(std::string &out, uint32_t field,
                      const std::string &bytes) {
  put_tag(out, field, 2);
  put_varint(out, bytes.size());
  out += bytes;
}

inline std::string encode_event_header(double wall_time, int64_t step) {
  std::string e;
  put_tag(e, 1, 1);
  put_fixed(e, &wall_time, sizeof(wall_time));
  if (step != 0) {
    put_tag(e, 2, 0);
    put_varint(e, static_cast<uint64_t>(step));
  }
  return e;
}

// Event{wall_time, file_version: "brain.Event:2"}; must open every file.
inline std::string encode_file_version_event(double wall_time) {
  std::string e = encode_event_header(wall_time, 0);
  put_bytes(e, 3, "brain.Event:2");
  return e;
}

// Event{wall_time, step, summary{value{tag, simple_value}}}.
inline std::string encode_scalar_event(double wall_time, int64_t step,
                                       const std::string &tag, float value) {
  std::string v;
  put_bytes(v, 1, tag);
  put_tag(v, 2, 5);
  put_fixed(v, &value, sizeof(value));
  std::string summary;
  put_bytes(summary, 1, v);
  std::string e = encode_event_header(wall_time, step);
  put_bytes(e, 5, summary);
  return e;
}

inline std::string frame_record(const std::string &data) {
  std::string out;
  const uint64_t len = data.size();
  put_fixed(out, &len, sizeof(len));
  const uint32_t len_crc = masked_crc32c(&len, sizeof(len));
  put_fixed(out, &len_crc, sizeof(len_crc));
  out += data;
  const uint32_t data_crc = masked_crc32c(data.data(), data.size());
  put_fixed(out, &data_crc, sizeof(data_crc));
  return out;
}

// Appends framed events to one file, flushing each so TensorBoard can read a
// run while it trains.
class Writer {
public:
  Writer(const std::string &path, double wall_time)
      : out_(path, std::ios::binary | std::ios::app) {
    if (!out_) {
      throw std::runtime_error("failed to open " + path);
    }
    write(encode_file_version_event(wall_time));
  }

  void write(const std::string &event) {
    out_ << frame_record(event);
    out_.flush();
    if (!out_) {
      throw std::runtime_error("failed to write event file");
    }
  }

private:
  std::ofstream out_;
};

} // namespace tfevents
