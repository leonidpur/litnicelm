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

  bool dry_run() const { return mode == RunMode::DryRun; }

  static TrainingPlan from(const Config &cfg, const Command &cmd) {
    TrainingPlan plan;
    plan.mode = (cmd.target == Command::Target::DRY_RUN || cfg.training.dry_run)
                    ? RunMode::DryRun
                    : RunMode::Train;
    plan.last_epoch = plan.dry_run() ? cfg.training.num_epochs_dry_run
                                     : cfg.training.num_epochs_train;
    return plan;
  }
};
