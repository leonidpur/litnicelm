#include "trainer_validation_utils.hpp"

#include "dataset.hpp"

#include <config.hpp>
#include <tokenizer_factory.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace TrainerValidationUtils {

void validate_training_context(const Config &cfg) {
  auto require_ctx = [](bool cond, const std::string &msg) {
    if (!cond) {
      throw std::runtime_error("Trainer Expertise Error: " + msg);
    }
  };

  require_ctx(cfg.training.batch_size > 0, "batch_size must be positive");
  require_ctx(cfg.training.train_seq_len > 0, "train_seq_len must be positive");
  require_ctx(cfg.training.window_stride > 0, "window_stride must be positive");
  require_ctx(cfg.training.train_seq_len <= cfg.model.max_seq_len,
              "train_seq_len exceeds max_seq_len");
}

void validate_vocab_contract_or_throw(const Config &cfg) {
  auto tokenizer = TokenizerFactory::create(cfg, nullptr);
  const uint32_t cfg_vocab = cfg.model.target_vocab_size;
  const uint32_t tok_vocab = static_cast<uint32_t>(tokenizer->vocab_size());
  const DatasetHeader header =
      TextDataset::read_header_or_throw(cfg.tokenization.output_binary);
  const uint32_t ds_vocab = header.vocab_size;
  if (cfg_vocab != tok_vocab || cfg_vocab != ds_vocab || tok_vocab != ds_vocab) {
    throw std::runtime_error("Mismatched Vocab: Config(" +
                             std::to_string(cfg_vocab) + ") != File(" +
                             std::to_string(tok_vocab) + ") != Header(" +
                             std::to_string(ds_vocab) +
                             "). Tokenizer was loaded to validate the shared "
                             "vocab contract between model config, tokenizer "
                             "artifacts, and dataset header.");
  }

  std::cout << "[TrainerValidationUtils] Loaded tokenizer '" << tokenizer->name()
            << "' to validate vocab contract: config=" << cfg_vocab
            << ", tokenizer=" << tok_vocab << ", dataset=" << ds_vocab
            << ". Check passed.\n";
}

void print_dataset_stats(const Config &cfg, const TextDataset &loader) {
  const uint64_t total_tokens = loader.num_tokens();
  const uint32_t tokens_per_step =
      cfg.training.batch_size * cfg.training.train_seq_len;

  if (tokens_per_step == 0) {
    return;
  }
  const uint64_t total_steps = loader.steps_per_epoch();
  const double data_mb =
      (total_tokens * sizeof(uint32_t)) / (1024.0 * 1024.0);

  std::cout << "\n[Dataset Stats]\n"
            << "  -> Source File:      " << cfg.tokenization.output_binary
            << "\n"
            << "  -> Total Tokens:     " << total_tokens << " (" << std::fixed
            << std::setprecision(2) << data_mb << " MB)\n"
            << "  -> Step Geometry:    " << cfg.training.batch_size
            << " (batch) x " << cfg.training.train_seq_len << " (seq_len)"
            << " stride=" << cfg.training.window_stride << "\n"
            << "  -> Steps per Epoch:  " << total_steps << "\n"
            << "  -> Total Iterations: "
            << (total_steps * cfg.training.num_epochs_train) << " (over "
            << cfg.training.num_epochs_train << " epochs)\n";

  if (loader.max_token_id() >= cfg.model.target_vocab_size) {
    throw std::runtime_error(
        "DATA ERROR: Dataset contains token ID " +
        std::to_string(loader.max_token_id()) +
        " which exceeds model.target_vocab_size (" +
        std::to_string(cfg.model.target_vocab_size) +
        "). Did you use the wrong tokenizer for this model?");
  }

  std::cout << "  -> Vocab Check:      Passed (Max ID: " << loader.max_token_id()
            << " < " << cfg.model.target_vocab_size << ")\n\n";
}

} // namespace TrainerValidationUtils
