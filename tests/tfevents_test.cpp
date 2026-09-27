// Checks the TensorBoard event encoder against published test vectors.

#include "tfevents_writer.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

std::string hex(const std::string &bytes) {
  static const char *digits = "0123456789abcdef";
  std::string out;
  for (unsigned char c : bytes) {
    out.push_back(digits[c >> 4]);
    out.push_back(digits[c & 0xF]);
  }
  return out;
}

} // namespace

int main() {
  // CRC-32C check value, and RFC 3720 (iSCSI) appendix B.4 vectors.
  const std::string digits = "123456789";
  check(tfevents::crc32c(digits.data(), digits.size()) == 0xE3069283u,
        "crc32c(\"123456789\") == 0xE3069283");
  const std::string zeros(32, '\x00');
  check(tfevents::crc32c(zeros.data(), zeros.size()) == 0x8A9136AAu,
        "crc32c(32 x 0x00) == 0x8A9136AA");
  const std::string ones(32, '\xFF');
  check(tfevents::crc32c(ones.data(), ones.size()) == 0x62A8AB43u,
        "crc32c(32 x 0xFF) == 0x62A8AB43");

  // Protobuf encoding guide: 150 -> 96 01, 300 -> ac 02.
  std::string v150;
  tfevents::put_varint(v150, 150);
  std::string v300;
  tfevents::put_varint(v300, 300);
  check(hex(v150) == "9601" && hex(v300) == "ac02", "varint encoding");

  // A framed record is 8 (length) + 4 + data + 4 bytes.
  const std::string event =
      tfevents::encode_scalar_event(1.5, 42, "epoch/train_loss", 0.25f);
  check(tfevents::frame_record(event).size() == event.size() + 16,
        "record framing size");

  std::cout << (g_failures == 0 ? "All tfevents tests passed.\n"
                                : "tfevents tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
