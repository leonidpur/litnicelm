#pragma once

#include <config.hpp>

#include <string>

struct HfExportOptions {
  std::string out_dir;
  // Checkpoint to export; empty: paths.model_file_best.
  std::string checkpoint_path;
};

// Exports a trained model as a Hugging Face model directory:
//   model.safetensors  the F32 parameters under GPT-2-style names
//                      (transformer.h.N.attn.c_attn.weight, ...; lm_head.weight
//                      transposed to [vocab, d_model])
//   config.json        architecture hyperparameters, GPT-2 keys plus ours
//                      (activation_function, position_encoding, rope_*)
//   tokenizer/         the tokenizer's artifact files, copied as they are
//   README.md          a model card: architecture, training summary, format
// Runs on the in-process CPU backend (parameters are F32 in every precision
// preset), so it needs no GPU.
class HfExporter {
public:
  // Returns the process exit code; throws on errors.
  static int run(const Config &cfg, const HfExportOptions &options);
};
