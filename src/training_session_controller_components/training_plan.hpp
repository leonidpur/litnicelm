#pragma once

#include <config.hpp>

#include <cstdint>

enum class RunMode {
  Train,
  DryRun,  // estimates timing; saves no checkpoints
};

// What this invocation will do, derived once from config and command.
struct TrainingPlan {
  RunMode mode = RunMode::Train;
  uint32_t last_epoch = 0;  // training.num_epochs_train or num_epochs_dry_run
  // --stop_after_epoch: end this invocation after that epoch, as if
  // interrupted; the LR schedule still spans last_epoch. 0: no early stop.
  uint32_t stop_after_epoch = 0;

  bool dry_run() const { return mode == RunMode::DryRun; }

  static TrainingPlan from(const Config &cfg, const Command &cmd) {
    TrainingPlan plan;
    plan.mode = (cmd.target == Command::Target::DRY_RUN || cfg.training.dry_run)
                    ? RunMode::DryRun
                    : RunMode::Train;
    plan.last_epoch = plan.dry_run() ? cfg.training.num_epochs_dry_run
                                     : cfg.training.num_epochs_train;
    plan.stop_after_epoch = cmd.stop_after_epoch;
    return plan;
  }
};
