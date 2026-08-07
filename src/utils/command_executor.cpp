#include "command_executor.hpp"

#include <inference.hpp>
#include "report_sink.hpp"
#include "trainer.hpp"
#include <tokenizer_factory.hpp>

int CommandExecutor::run(const Command &cmd, const Config &cfg) {
  ConsoleSink sink(cfg.logging);
  switch (cmd.target) {
  case Command::Target::TOKENIZER_TRAINING:
    return run_tokenizer_training_mode(cmd.config_path, &sink);
  case Command::Target::TOKENIZE:
    return run_tokenization_mode(cmd.config_path, &sink);
  case Command::Target::INFER:
    return run_infer_mode(cfg, &sink);
  case Command::Target::INSPECT:
    return run_inspect_mode(cfg, &sink);
  case Command::Target::INFERLOOP:
    return run_inferloop_mode(cfg, &sink);
  case Command::Target::DRY_RUN:
  case Command::Target::TRAIN:
    return Trainer::train_entry_point(cfg, cmd);
  }

  return 1;
}
