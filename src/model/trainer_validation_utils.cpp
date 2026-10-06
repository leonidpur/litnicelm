#include "trainer_validation_utils.hpp"

#include "dataset.hpp"
#include "logger.hpp"

#include <config.hpp>
#include <tokenizer.hpp>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace TrainerValidationUtils {

void validate_config_tokenizer_and_dataset(const Config &cfg,
                                           const Tokenizer &tokenizer) {
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

  const uint32_t tok_vocab = static_cast<uint32_t>(tokenizer.vocab_size());
  const DatasetHeader header =
      TextDataset::read_header_or_throw(cfg.tokenization.output_binary);
  const uint32_t ds_vocab = header.vocab_size;
  if (tok_vocab != ds_vocab) {
    throw std::runtime_error("Mismatched Vocab: tokenizer(" +
                             std::to_string(tok_vocab) + ") != dataset header(" +
                             std::to_string(ds_vocab) + ")");
  }

  logger().log(EventId::ConfigTokenizerDatasetValidated,
               "batch_size=" + std::to_string(cfg.training.batch_size) +
                   ", train_seq_len=" + std::to_string(cfg.training.train_seq_len) +
                   " <= max_seq_len=" + std::to_string(cfg.model.max_seq_len) +
                   ", window_stride=" + std::to_string(cfg.training.window_stride) +
                   "; vocab: tokenizer '" + std::string(tokenizer.name()) + "'=" +
                   std::to_string(tok_vocab) +
                   ", dataset=" + std::to_string(ds_vocab));
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

  std::ostringstream stats;
  stats << "\n"
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
        << cfg.training.num_epochs_train << " epochs)";
  logger().log(EventId::DatasetStats, stats.str());

  if (loader.max_token_id() >= cfg.model.vocab_size) {
    throw std::runtime_error(
        "DATA ERROR: Dataset contains token ID " +
        std::to_string(loader.max_token_id()) +
        " which exceeds model.vocab_size (" +
        std::to_string(cfg.model.vocab_size) +
        "). Did you use the wrong tokenizer for this model?");
  }

  logger().log(EventId::VocabCheckPassed,
               "(Max ID: " + std::to_string(loader.max_token_id()) + " < " +
                   std::to_string(cfg.model.vocab_size) + ")\n");
}

void validate_validation_split(const Config &cfg, const TextDataset &val_loader) {
  if (cfg.training.validation_fraction > 0.0f &&
      val_loader.steps_per_epoch() == 0) {
    throw std::runtime_error(
        "validation split holds no full batch of " +
        std::to_string(cfg.training.batch_size) +
        "; raise training.validation_fraction or lower training.batch_size");
  }
}

} // namespace TrainerValidationUtils
