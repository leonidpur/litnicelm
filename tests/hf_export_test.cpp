// Checks an `export` directory against the checkpoint it came from:
// model.safetensors, read here with its own parser (not the writer's code),
// holds exactly the expected GPT-2-style tensor names and shapes, and every
// tensor is bit-identical to its checkpoint parameter (lm_head.weight
// against the transposed parameter); __metadata__ has format "pt";
// config.json, README.md and the tokenizer files are present.
//
// Usage: litnice_hf_export_test <config.yaml> <export dir>

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "memory/inference_memory_manager.hpp"
#include "memory/precision_policy.hpp"
#include "model_algo_config.hpp"
#include "model_algo_factory.hpp"
#include "runtime_config.hpp"
#include <tokenizer_factory.hpp>

#include <config.hpp>

#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void expect(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  failures += !ok;
}

struct StEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

// A minimal reader for the safetensors header: an object of objects whose
// values are strings, arrays of integers, or (in __metadata__) strings.
class HeaderParser {
public:
  explicit HeaderParser(const std::string &text) : s_(text) {}

  void parse(std::map<std::string, StEntry> &entries,
             std::map<std::string, std::string> &metadata) {
    expect_char('{');
    if (peek() == '}') {
      return;
    }
    do {
      const std::string name = string();
      expect_char(':');
      expect_char('{');
      StEntry e;
      do {
        const std::string key = string();
        expect_char(':');
        if (name == "__metadata__") {
          metadata[key] = string();
        } else if (key == "dtype") {
          e.dtype = string();
        } else if (key == "shape") {
          for (const uint64_t v : integers()) {
            e.shape.push_back(static_cast<int64_t>(v));
          }
        } else if (key == "data_offsets") {
          const std::vector<uint64_t> o = integers();
          if (o.size() != 2) {
            throw std::runtime_error("data_offsets needs two values");
          }
          e.begin = o[0];
          e.end = o[1];
        } else {
          throw std::runtime_error("unexpected key " + key);
        }
      } while (accept(','));
      expect_char('}');
      if (name != "__metadata__") {
        entries[name] = e;
      }
    } while (accept(','));
    expect_char('}');
  }

private:
  const std::string &s_;
  size_t i_ = 0;

  void skip() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      ++i_;
    }
  }
  char peek() {
    skip();
    return i_ < s_.size() ? s_[i_] : '\0';
  }
  bool accept(char c) {
    if (peek() == c) {
      ++i_;
      return true;
    }
    return false;
  }
  void expect_char(char c) {
    if (!accept(c)) {
      throw std::runtime_error(std::string("expected '") + c + "' at " + std::to_string(i_));
    }
  }
  std::string string() {
    expect_char('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\') {
        ++i_;
      }
      out += s_[i_++];
    }
    ++i_;
    return out;
  }
  std::vector<uint64_t> integers() {
    expect_char('[');
    std::vector<uint64_t> out;
    if (accept(']')) {
      return out;
    }
    do {
      skip();
      size_t used = 0;
      out.push_back(std::stoull(s_.substr(i_), &used));
      i_ += used;
    } while (accept(','));
    expect_char(']');
    return out;
  }
};

std::vector<int64_t> dims_of(const TensorView &t) {
  if (t.rank() == 2 && t.dim(0) == 1) {
    return {t.dim(1)};
  }
  std::vector<int64_t> d;
  for (size_t i = 0; i < t.rank(); ++i) {
    d.push_back(t.dim(i));
  }
  return d;
}

std::string shape_text(const std::vector<int64_t> &s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    out += (i ? "," : "") + std::to_string(s[i]);
  }
  return out + "]";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <config.yaml> <export dir>\n";
    return 2;
  }
  const fs::path dir = argv[2];

  // The checkpoint, loaded the way inference loads it, on the CPU.
  Config cfg = Config::load_from_file(argv[1]);
  cfg.backend.library.clear();
  cfg.precision = Precision::Fp32;
  const std::unique_ptr<Tokenizer> tokenizer = TokenizerFactory::create(cfg);
  cfg = with_tokenizer_vocab_size(cfg, *tokenizer);
  const std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  const ModelAlgoFactory algo(ModelAlgoConfig::from_config(cfg));
  const PrecisionPolicy precision = PrecisionPolicy::create(cfg, backend->precision_variants());
  InferenceMemoryManager memory(cfg, algo, *backend, precision);
  uint64_t step = 0;
  uint32_t epoch = 0;
  std::string error;
  if (!load_checkpoint(cfg.paths.model_file_best, cfg.model, *algo.create_position_encoding(cfg),
                       cfg.conf_version, cfg.memory.alignment_bytes, *backend,
                       memory.data_arena(), memory.adam_state(), step, epoch, nullptr,
                       &error)) {
    std::cerr << "cannot load " << cfg.paths.model_file_best << ": " << error << "\n";
    return 1;
  }
  const TensorStore &store = memory.tensor_store();

  // Expected tensors, named independently of the exporter.
  std::vector<std::pair<std::string, const TensorView *>> want;
  want.emplace_back("transformer.wte.weight", &store.param_tok_embedding());
  const bool learned = cfg.model_algo.position_encoding == "learned";
  if (learned) {
    want.emplace_back("transformer.wpe.weight", &store.param_pos_embedding());
  }
  for (int l = 0; l < static_cast<int>(cfg.model.n_layers); ++l) {
    const std::string h = "transformer.h." + std::to_string(l) + ".";
    want.emplace_back(h + "ln_1.weight", &store.param_ln1_gamma(l));
    want.emplace_back(h + "ln_1.bias", &store.param_ln1_beta(l));
    want.emplace_back(h + "attn.c_attn.weight", &store.param_attn_qkv_w(l));
    want.emplace_back(h + "attn.c_attn.bias", &store.param_attn_qkv_b(l));
    want.emplace_back(h + "attn.c_proj.weight", &store.param_attn_out_w(l));
    want.emplace_back(h + "attn.c_proj.bias", &store.param_attn_out_b(l));
    want.emplace_back(h + "ln_2.weight", &store.param_ln2_gamma(l));
    want.emplace_back(h + "ln_2.bias", &store.param_ln2_beta(l));
    want.emplace_back(h + "mlp.c_fc.weight", &store.param_ffn_w1(l));
    want.emplace_back(h + "mlp.c_fc.bias", &store.param_ffn_b1(l));
    want.emplace_back(h + "mlp.c_proj.weight", &store.param_ffn_w2(l));
    want.emplace_back(h + "mlp.c_proj.bias", &store.param_ffn_b2(l));
  }
  want.emplace_back("transformer.ln_f.weight", &store.param_lnf_gamma());
  want.emplace_back("transformer.ln_f.bias", &store.param_lnf_beta());

  // The file.
  std::ifstream in(dir / "model.safetensors", std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 8) {
    std::cerr << "model.safetensors missing or truncated\n";
    return 1;
  }
  uint64_t header_len = 0;
  for (int b = 7; b >= 0; --b) {
    header_len = (header_len << 8) | static_cast<unsigned char>(bytes[b]);
  }
  const std::string header(bytes.data() + 8, header_len);
  const char *data = bytes.data() + 8 + header_len;
  const uint64_t data_bytes = bytes.size() - 8 - header_len;
  std::map<std::string, StEntry> entries;
  std::map<std::string, std::string> metadata;
  HeaderParser(header).parse(entries, metadata);

  expect(metadata["format"] == "pt", "__metadata__ format is \"pt\"");
  expect(entries.size() == want.size() + 1,
         std::to_string(entries.size()) + " tensors, expected " +
             std::to_string(want.size() + 1) + (learned ? " (with wpe)" : " (no wpe)"));

  uint64_t covered = 0;
  bool all_ok = true;
  auto check = [&](const std::string &name, const std::vector<int64_t> &shape,
                   const std::vector<float> &values) {
    const auto it = entries.find(name);
    if (it == entries.end()) {
      expect(false, name + " present");
      all_ok = false;
      return;
    }
    const StEntry &e = it->second;
    const uint64_t n = values.size() * sizeof(float);
    const bool ok = e.dtype == "F32" && e.shape == shape && e.end - e.begin == n &&
                    e.end <= data_bytes &&
                    std::memcmp(data + e.begin, values.data(), n) == 0;
    if (!ok) {
      expect(false, name + ": dtype " + e.dtype + ", shape " + shape_text(e.shape) +
                        " (want " + shape_text(shape) + "), bytes differ or out of range");
      all_ok = false;
    }
    covered += e.end - e.begin;
  };
  for (const auto &[name, view] : want) {
    const float *p = static_cast<const float *>(view->data());
    check(name, dims_of(*view), std::vector<float>(p, p + view->numel()));
  }
  const TensorView &lm = store.param_lm_head_w();
  const int64_t rows = lm.dim(0);
  const int64_t cols = lm.dim(1);
  std::vector<float> lm_t(static_cast<size_t>(rows * cols));
  const float *lp = static_cast<const float *>(lm.data());
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      lm_t[static_cast<size_t>(c * rows + r)] = lp[r * cols + c];
    }
  }
  check("lm_head.weight", {cols, rows}, lm_t);
  expect(all_ok, "every tensor has the expected name, F32 dtype and shape, and the "
                 "checkpoint's exact bytes (lm_head transposed)");
  expect(covered == data_bytes, "the tensors cover the data section exactly (" +
                                    std::to_string(covered) + " of " +
                                    std::to_string(data_bytes) + " bytes)");

  std::ifstream config_json(dir / "config.json");
  std::stringstream config_text;
  config_text << config_json.rdbuf();
  const std::string cj = config_text.str();
  expect(cj.find("\"n_layer\": " + std::to_string(cfg.model.n_layers)) != std::string::npos &&
             cj.find("\"position_encoding\": \"" + cfg.model_algo.position_encoding + "\"") !=
                 std::string::npos,
         "config.json has n_layer and position_encoding");
  std::ifstream card(dir / "README.md");
  std::stringstream card_text;
  card_text << card.rdbuf();
  const std::string ct = card_text.str();
  expect(ct.rfind("---\nlicense: ", 0) == 0 && ct.find(fs::current_path().string()) == std::string::npos &&
             ct.find(cfg.tokenization.input_corpus) == std::string::npos,
         "README.md has a license and no absolute local paths");
  expect(fs::is_directory(dir / "tokenizer") && !fs::is_empty(dir / "tokenizer"),
         "tokenizer files copied");

  std::cout << (failures == 0 ? "All export tests passed.\n" : "Export tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
