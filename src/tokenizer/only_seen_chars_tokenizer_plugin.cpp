#include "only_seen_chars_tokenizer_plugin.hpp"

#include <report_interface.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;

fs::path artifact_path(const std::string &artifacts_dir) {
  return fs::path(artifacts_dir) / "char_seen.json";
}

std::string json_escape(const std::string &s) {
  std::ostringstream out;
  for (unsigned char ch : s) {
    switch (ch) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20 || ch >= 0x7f) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  return out.str();
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return 10 + (c - 'a');
  }
  if (c >= 'A' && c <= 'F') {
    return 10 + (c - 'A');
  }
  return -1;
}

std::string parse_json_string(const std::string &text, size_t &pos) {
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("OnlySeenCharsTokenizer: expected JSON string");
  }
  ++pos;
  std::string out;
  while (pos < text.size()) {
    const char c = text[pos++];
    if (c == '"') {
      return out;
    }
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("OnlySeenCharsTokenizer: bad JSON escape");
    }
    const char e = text[pos++];
    switch (e) {
    case '"':
    case '\\':
    case '/':
      out.push_back(e);
      break;
    case 'n':
      out.push_back('\n');
      break;
    case 'r':
      out.push_back('\r');
      break;
    case 't':
      out.push_back('\t');
      break;
    case 'u': {
      if (pos + 4 > text.size()) {
        throw std::runtime_error("OnlySeenCharsTokenizer: bad JSON unicode escape");
      }
      int value = 0;
      for (int i = 0; i < 4; ++i) {
        const int d = hex_digit(text[pos++]);
        if (d < 0) {
          throw std::runtime_error("OnlySeenCharsTokenizer: bad JSON unicode escape");
        }
        value = (value << 4) | d;
      }
      if (value > 255) {
        throw std::runtime_error(
            "OnlySeenCharsTokenizer: only byte-sized chars are supported");
      }
      out.push_back(static_cast<char>(static_cast<unsigned char>(value)));
      break;
    }
    default:
      throw std::runtime_error("OnlySeenCharsTokenizer: unsupported JSON escape");
    }
  }
  throw std::runtime_error("OnlySeenCharsTokenizer: unterminated JSON string");
}

std::vector<std::string> parse_vocab_array(const std::string &text) {
  const size_t key = text.find("\"vocab\"");
  if (key == std::string::npos) {
    throw std::runtime_error("OnlySeenCharsTokenizer: missing vocab in artifact");
  }
  size_t pos = text.find('[', key);
  if (pos == std::string::npos) {
    throw std::runtime_error("OnlySeenCharsTokenizer: missing vocab array");
  }
  ++pos;

  std::vector<std::string> vocab;
  while (pos < text.size()) {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (pos < text.size() && text[pos] == ']') {
      return vocab;
    }
    vocab.push_back(parse_json_string(text, pos));
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (pos < text.size() && text[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < text.size() && text[pos] == ']') {
      return vocab;
    }
    throw std::runtime_error("OnlySeenCharsTokenizer: malformed vocab array");
  }
  throw std::runtime_error("OnlySeenCharsTokenizer: unterminated vocab array");
}

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TOKENIZER, event, step, value,
                          message);
}

void warn_about_suspicious_byte_vocab(const std::vector<std::string> &vocab,
                                      const std::string &context) {
  size_t non_ascii = 0;
  size_t c1_controls = 0;
  bool has_cr = false;
  std::vector<unsigned int> examples;
  for (const std::string &token : vocab) {
    if (token.size() != 1) {
      continue;
    }
    const unsigned int byte =
        static_cast<unsigned int>(static_cast<unsigned char>(token[0]));
    const bool suspicious = byte >= 0x80 || byte == '\r';
    if (byte >= 0x80) {
      ++non_ascii;
    }
    if (byte >= 0x80 && byte <= 0x9f) {
      ++c1_controls;
    }
    if (byte == '\r') {
      has_cr = true;
    }
    if (suspicious && examples.size() < 8) {
      examples.push_back(byte);
    }
  }

  if (non_ascii == 0 && c1_controls == 0 && !has_cr) {
    return;
  }

  std::ostringstream oss;
  oss << "[OnlySeenCharsTokenizer][WARN] " << context
      << " byte vocabulary contains non_ascii=" << non_ascii
      << ", c1_controls=" << c1_controls
      << ", carriage_return=" << (has_cr ? "yes" : "no")
      << ". only_seen_chars is byte-level; UTF-8 smart quotes or mojibake can "
         "be learned as separate bytes and later decode as replacement "
         "characters. Prefer an ASCII-normalized corpus or a real Unicode/BPE "
         "tokenizer for non-ASCII text. examples=";
  for (size_t i = 0; i < examples.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << "0x" << std::hex << std::uppercase << std::setw(2)
        << std::setfill('0') << examples[i] << std::dec
        << std::nouppercase << std::setfill(' ');
  }
  std::cerr << oss.str() << "\n";
}
} // namespace

OnlySeenCharsTokenizerPlugin::OnlySeenCharsTokenizerPlugin(
    uint32_t vocab_size_limit)
    : vocab_size_limit_(vocab_size_limit) {}

void OnlySeenCharsTokenizerPlugin::train(const std::string &corpus_path,
                                         const std::string &artifacts_dir,
                                         uint32_t target_vocab_size,
                                         ReportSink *sink) {
  if (corpus_path.empty()) {
    throw std::runtime_error(
        "OnlySeenCharsTokenizer: tokenizer.training_corpus is required");
  }
  if (artifacts_dir.empty()) {
    throw std::runtime_error(
        "OnlySeenCharsTokenizer: tokenizer.artifacts_dir is required");
  }

  std::ifstream in(corpus_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("OnlySeenCharsTokenizer: failed to open corpus: " +
                             corpus_path);
  }

  std::vector<std::string> seen;
  bool present[256] = {};
  char ch = 0;
  while (in.get(ch)) {
    const unsigned char b = static_cast<unsigned char>(ch);
    if (!present[b]) {
      present[b] = true;
      seen.emplace_back(1, static_cast<char>(b));
    }
  }

  const uint32_t limit =
      (target_vocab_size == 0) ? vocab_size_limit_ : target_vocab_size;
  if (limit != 0 && seen.size() > limit) {
    throw std::runtime_error(
        "OnlySeenCharsTokenizer: seen chars exceed target_vocab_size");
  }

  vocab_ = seen;
  rebuild_index_();
  warn_about_suspicious_byte_vocab(vocab_, "trained");

  fs::create_directories(artifacts_dir);
  std::ofstream out(artifact_path(artifacts_dir), std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error(
        "OnlySeenCharsTokenizer: failed to write char_seen.json");
  }

  out << "{\n";
  out << "  \"type\": \"char_seen\",\n";
  out << "  \"version\": 1,\n";
  out << "  \"vocab\": [";
  for (size_t i = 0; i < vocab_.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "\"" << json_escape(vocab_[i]) << "\"";
  }
  out << "],\n";
  out << "  \"char_to_id\": {\n";
  bool first = true;
  for (size_t i = 0; i < vocab_.size(); ++i) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    \"" << json_escape(vocab_[i]) << "\": " << i;
  }
  out << "\n  }\n";
  out << "}\n";

  report_if(sink, ReportEvent::END, static_cast<uint32_t>(vocab_.size()), 100.0f,
            "OnlySeenCharsTokenizer trained: chars=" +
                std::to_string(vocab_.size()) +
                ", artifact=" + artifact_path(artifacts_dir).string());
}

bool OnlySeenCharsTokenizerPlugin::load(const std::string &artifacts_dir) {
  std::ifstream in(artifact_path(artifacts_dir), std::ios::binary);
  if (!in) {
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  vocab_ = parse_vocab_array(ss.str());
  if (vocab_size_limit_ != 0 && vocab_.size() > vocab_size_limit_) {
    throw std::runtime_error(
        "OnlySeenCharsTokenizer: artifact vocab exceeds model.target_vocab_size");
  }
  rebuild_index_();
  warn_about_suspicious_byte_vocab(vocab_, "loaded artifact");
  return true;
}

std::vector<int32_t>
OnlySeenCharsTokenizerPlugin::encode(const std::string &text) const {
  std::vector<int32_t> out;
  out.reserve(text.size());
  for (unsigned char b : text) {
    const std::string key(1, static_cast<char>(b));
    const auto it = char_to_id_.find(key);
    if (it == char_to_id_.end()) {
      throw std::runtime_error(
          "OnlySeenCharsTokenizer::encode: unseen char in input");
    }
    out.push_back(it->second);
  }
  return out;
}

std::string
OnlySeenCharsTokenizerPlugin::decode(const std::vector<int32_t> &ids) const {
  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
      throw std::runtime_error(
          "OnlySeenCharsTokenizer::decode: id out of range");
    }
    out += vocab_[static_cast<size_t>(id)];
  }
  return out;
}

int32_t OnlySeenCharsTokenizerPlugin::vocab_size() const {
  return static_cast<int32_t>(vocab_.size());
}

const char *OnlySeenCharsTokenizerPlugin::name() const {
  return "OnlySeenCharsTokenizer";
}

void OnlySeenCharsTokenizerPlugin::rebuild_index_() {
  char_to_id_.clear();
  for (size_t i = 0; i < vocab_.size(); ++i) {
    char_to_id_[vocab_[i]] = static_cast<int32_t>(i);
  }
}
