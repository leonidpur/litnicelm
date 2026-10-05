#pragma once

#include "checkpoint.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <cstdint>
#include <string>

// Outcome of one epoch for convergence.
struct ConvergenceVerdict {
  bool new_best = false;
  ContinueTrainingDecision decision;  // decided; stop and why, if stopping
};

// Convergence rules on the epoch training loss and their state: best loss,
// improvement by training.min_delta, patience (counted from
// training.min_epochs), training.target_loss and non-finite loss. Keeps no
// files and saves nothing; its state travels in checkpoints through
// state() / restore().
class ConvergenceMonitor {
public:
  explicit ConvergenceMonitor(const Config &cfg);

  ConvergenceVerdict observe(uint32_t epoch, const EpochMetrics &metrics);

  void reset();
  void restore(const CheckpointConvergenceState &state);
  CheckpointConvergenceState state() const;
  // A resumed run continues even if the checkpoint recorded a stop.
  void clear_persisted_stop_for_resume();

  bool has_best() const { return has_best_; }
  float best_loss() const { return best_loss_; }
  uint32_t best_epoch() const { return best_epoch_; }
  float last_epoch_loss() const { return last_epoch_loss_; }
  bool should_stop() const { return stop_requested_; }
  bool early_stopped() const {
    return stop_requested_ && stop_reason_ != StopReason::None;
  }
  std::string stop_reason_text() const;

private:
  enum class StopReason : uint32_t {
    None = 0,
    TargetLoss = 1,
    Patience = 2,
    NonFiniteLoss = 3,
  };

  bool loss_improved(float mean_loss) const;
  void request_stop(StopReason reason, const std::string &message);
  ConvergenceVerdict verdict(bool new_best) const;

  const Config &cfg_;
  float best_loss_ = 0.0f;
  float last_epoch_loss_ = -1.0f;
  uint32_t best_epoch_ = 0;
  uint32_t epochs_without_improvement_ = 0;
  bool has_best_ = false;
  bool stop_requested_ = false;
  StopReason stop_reason_ = StopReason::None;
  std::string stop_message_;
};
