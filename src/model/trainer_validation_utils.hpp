#pragma once

struct Config;
class TextDataset;

namespace TrainerValidationUtils {

void validate_training_context(const Config &cfg);
void validate_vocab_contract_or_throw(const Config &cfg);
void print_dataset_stats(const Config &cfg, const TextDataset &loader);

} // namespace TrainerValidationUtils
