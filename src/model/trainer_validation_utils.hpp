#pragma once

struct Config;
class TextDataset;
class Tokenizer;

namespace TrainerValidationUtils {

// Checks the training config (batch, sequence length, stride) and that the
// tokenizer and the dataset header (read from tokenization.output_binary)
// agree on the vocab size. Throws on the first
// failure; logs one line when all pass.
void validate_config_tokenizer_and_dataset(const Config &cfg,
                                           const Tokenizer &tokenizer);
void print_dataset_stats(const Config &cfg, const TextDataset &loader);

} // namespace TrainerValidationUtils
