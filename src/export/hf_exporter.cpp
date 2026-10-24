#include "hf_exporter.hpp"

#include "algo/position_encoding.hpp"
#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "export_events.hpp"
#include "memory/inference_memory_manager.hpp"
#include "memory/precision_policy.hpp"
#include "model_algo_config.hpp"
#include "model_algo_factory.hpp"
#include "runtime_config.hpp"
#include "safetensors_writer.hpp"
#include <tokenizer_factory.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

// DefaultCpuBackend and the CUDA layernorm kernels use this epsilon.
constexpr double kLayerNormEpsilon = 1e-5;

// ---------------------------------------------------------------------------
// Loading: the checkpoint's parameters on the in-process CPU backend.
// ---------------------------------------------------------------------------

// The run's config adjusted for loading: the in-process CPU backend under
// fp32 (parameters are F32 in every preset), whatever the run used.
Config cpu_fp32_config(const Config &run_cfg) {
  Config cfg = run_cfg;
  cfg.backend.library.clear();
  cfg.precision = Precision::Fp32;
  return cfg;
}

// A checkpoint loaded for reading its parameters, with the training state
// stored next to them.
class LoadedCheckpoint {
public:
  LoadedCheckpoint(const Config &run_cfg, const std::string &path)
      : tokenizer_(TokenizerFactory::create(cpu_fp32_config(run_cfg))),
        cfg_(with_tokenizer_vocab_size(cpu_fp32_config(run_cfg), *tokenizer_)),
        backend_(DeviceBackend::create_instance(cfg_)),
        algo_(ModelAlgoConfig::from_config(cfg_)),
        precision_(PrecisionPolicy::create(cfg_, backend_->precision_variants())),
        memory_(cfg_, algo_, *backend_, precision_) {
    std::string error;
    if (!load_checkpoint(path, cfg_.model, *algo_.create_position_encoding(cfg_),
                         cfg_.conf_version, cfg_.memory.alignment_bytes, *backend_,
                         memory_.data_arena(), memory_.adam_state(), step, epoch,
                         &convergence, &error)) {
      throw std::runtime_error("export: failed to load checkpoint " + path + ": " + error);
    }
    if (!verify_checkpoint_tokenizer_fingerprint(
            path, checkpoint_tokenizer_fingerprint(cfg_), &error)) {
      throw std::runtime_error("export: " + path + ": " + error);
    }
  }

  // The loading config: the run's, with the tokenizer's vocabulary size.
  const Config &cfg() const { return cfg_; }
  const TensorStore &store() const { return memory_.tensor_store(); }
  bool has_position_table() const {
    return algo_.create_position_encoding(cfg_)->needs_position_table();
  }

  uint64_t step = 0;
  uint32_t epoch = 0;
  CheckpointConvergenceState convergence;

private:
  std::unique_ptr<Tokenizer> tokenizer_;
  Config cfg_;
  std::unique_ptr<DeviceBackend> backend_;
  ModelAlgoFactory algo_;
  PrecisionPolicy precision_;
  InferenceMemoryManager memory_;
};

// ---------------------------------------------------------------------------
// model.safetensors
// ---------------------------------------------------------------------------

// Shape of a parameter as exported: a [1, n] row vector becomes [n].
std::vector<int64_t> export_shape(const TensorView &param) {
  if (param.rank() == 2 && param.dim(0) == 1) {
    return {param.dim(1)};
  }
  std::vector<int64_t> shape;
  for (size_t i = 0; i < param.rank(); ++i) {
    shape.push_back(param.dim(i));
  }
  return shape;
}

const float *f32_data(const TensorView &param) {
  return static_cast<const float *>(param.data());
}

// A [rows, cols] parameter as a [cols, rows] row-major copy.
std::vector<float> transposed(const TensorView &param) {
  const int64_t rows = param.dim(0);
  const int64_t cols = param.dim(1);
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c * rows + r)] = f32_data(param)[r * cols + c];
    }
  }
  return out;
}

// Adds block l's parameters under transformer.h.<l>.*.
void add_block(SafetensorsWriter &writer, const TensorStore &store, int l) {
  const std::string h = "transformer.h." + std::to_string(l) + ".";
  auto add = [&](const std::string &name, const TensorView &param) {
    writer.add_f32(h + name, export_shape(param), f32_data(param));
  };
  add("ln_1.weight", store.param_ln1_gamma(l));
  add("ln_1.bias", store.param_ln1_beta(l));
  add("attn.c_attn.weight", store.param_attn_qkv_w(l));
  add("attn.c_attn.bias", store.param_attn_qkv_b(l));
  add("attn.c_proj.weight", store.param_attn_out_w(l));
  add("attn.c_proj.bias", store.param_attn_out_b(l));
  add("ln_2.weight", store.param_ln2_gamma(l));
  add("ln_2.bias", store.param_ln2_beta(l));
  add("mlp.c_fc.weight", store.param_ffn_w1(l));
  add("mlp.c_fc.bias", store.param_ffn_b1(l));
  add("mlp.c_proj.weight", store.param_ffn_w2(l));
  add("mlp.c_proj.bias", store.param_ffn_b2(l));
}

// Every parameter under its GPT-2-style name; lm_head.weight transposed to
// [vocab, d_model] as nn.Linear (and GPT-2's lm_head) store it.
void write_weights(const LoadedCheckpoint &model, const fs::path &file) {
  const TensorStore &store = model.store();
  SafetensorsWriter writer;
  writer.set_metadata("format", "pt");
  auto add = [&](const std::string &name, const TensorView &param) {
    writer.add_f32(name, export_shape(param), f32_data(param));
  };
  add("transformer.wte.weight", store.param_tok_embedding());
  if (model.has_position_table()) {
    add("transformer.wpe.weight", store.param_pos_embedding());
  }
  for (int l = 0; l < static_cast<int>(model.cfg().model.n_layers); ++l) {
    add_block(writer, store, l);
  }
  add("transformer.ln_f.weight", store.param_lnf_gamma());
  add("transformer.ln_f.bias", store.param_lnf_beta());
  const TensorView &lm_head = store.param_lm_head_w();
  const std::vector<float> lm_head_t = transposed(lm_head);
  writer.add_f32("lm_head.weight", {lm_head.dim(1), lm_head.dim(0)}, lm_head_t.data());
  writer.write(file.string());
}

// ---------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------

// Text for a value that was a float: 6 significant digits, so 0.0006f
// prints as 0.0006, not its double expansion.
std::string number(double v) {
  std::ostringstream out;
  out << std::setprecision(6) << v;
  return out.str();
}

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::trunc);
  out << text;
  if (!out) {
    throw std::runtime_error("HfExporter: cannot write " + path.string());
  }
}

std::string config_json(const Config &cfg) {
  const std::string &pos = cfg.model_algo.position_encoding;
  std::ostringstream j;
  j << "{\n"
    << "  \"model_type\": \"litnicelm\",\n"
    << "  \"architectures\": [\"LitniceLMForCausalLM\"],\n"
    << "  \"vocab_size\": " << cfg.model.vocab_size << ",\n"
    << "  \"n_positions\": " << cfg.model.max_seq_len << ",\n"
    << "  \"n_embd\": " << cfg.model.d_model << ",\n"
    << "  \"n_layer\": " << cfg.model.n_layers << ",\n"
    << "  \"n_head\": " << cfg.model.n_heads << ",\n"
    << "  \"n_inner\": " << cfg.model.d_ff << ",\n"
    << "  \"layer_norm_epsilon\": " << number(kLayerNormEpsilon) << ",\n"
    << "  \"activation_function\": \"relu\",\n"
    << "  \"position_encoding\": \"" << pos << "\",\n";
  if (pos == "rope") {
    j << "  \"rope_theta\": " << number(RopePositionEncoding::kBase) << ",\n"
      << "  \"rope_layout\": \"half_split\",\n";
  }
  j << "  \"tie_word_embeddings\": false,\n"
    << "  \"lm_head_bias\": false,\n"
    << "  \"tokenizer_type\": \"" << cfg.tokenizer.type << "\",\n"
    << "  \"torch_dtype\": \"float32\"\n"
    << "}\n";
  return j.str();
}

} // namespace

int HfExporter::run(const Config &cfg, const HfExportOptions &options) {
  if (options.out_dir.empty()) {
    throw std::runtime_error("export: --out <dir> is required");
  }
  const std::string checkpoint = options.checkpoint_path.empty()
                                     ? cfg.paths.model_file_best
                                     : options.checkpoint_path;
  logger().log(export_events::kExportStart, checkpoint + " -> " + options.out_dir);
  const LoadedCheckpoint model(cfg, checkpoint);

  const fs::path out = options.out_dir;
  fs::create_directories(out);
  write_weights(model, out / "model.safetensors");
  write_text(out / "config.json", config_json(model.cfg()));

  logger().log(export_events::kExportEnd,
               "wrote model.safetensors, config.json to " + options.out_dir);
  return 0;
}
