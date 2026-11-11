// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "command_executor.hpp"

#include "hf_exporter.hpp"
#include <inference.hpp>
#include "program_terminal.hpp"
#include "variant_catalog.hpp"
#include "trainer.hpp"
#include <tokenizer_factory.hpp>

int CommandExecutor::run(const Command &cmd, const Config &cfg) {
  ProgramTerminal terminal;
  switch (cmd.target) {
  case Command::Target::TOKENIZER_TRAINING:
    return run_tokenizer_training_mode(cmd.config_path);
  case Command::Target::TOKENIZE:
    return run_tokenization_mode(cmd.config_path);
  case Command::Target::INFER:
    return run_infer_mode(cfg, terminal);
  case Command::Target::INSPECT:
    return run_inspect_mode(cfg, terminal);
  case Command::Target::INFERLOOP:
    return run_inferloop_mode(cfg, terminal);
  case Command::Target::EXPORT:
    return HfExporter::run(cfg, {cmd.export_out_dir, cmd.export_checkpoint});
  case Command::Target::DRY_RUN:
  case Command::Target::TRAIN:
    return Trainer::train_entry_point(cfg, cmd);
  case Command::Target::LIST_VARIANTS:
    return run_without_config(cmd);
  }

  return 1;
}

int CommandExecutor::run_without_config(const Command &cmd) {
  ProgramTerminal terminal;
  if (cmd.target == Command::Target::LIST_VARIANTS) {
    terminal.variant_catalog(render_variant_catalog());
    return 0;
  }
  return 1;
}
