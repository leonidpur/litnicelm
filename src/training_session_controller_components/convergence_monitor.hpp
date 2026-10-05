#pragma once

#include "checkpoint.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <cstdint>
#include <optional>
#include <string>

// Outcome of one epoch for convergence.
struct ConvergenceVerdict {
  bool new_best = false;
  ContinueTrainingDecision decision;  // decided; stop, if stopping
};

// Convergence rules and their state: best loss, improvement by
// training.min_delta, patience in epochs (counted from training.min_epochs)
// and training.target_loss. They judge the val loss when the run has a
// validation split (only on epochs where validation ran), else the train loss.
// A non-finite train loss is checked every epoch. Keeps no files and saves
// nothing; its state travels in checkpoints through state() / restore().
class ConvergenceMonitor {
public:
  explicit ConvergenceMonitor(const Config &cfg);

  ConvergenceVerdict observe(uint32_t epoch, const EpochMetrics &metrics);

  void reset();
  // resumed_epoch: the checkpoint's epoch, where patience counting resumes.
  void restore(const CheckpointConvergenceState &state,
               uint32_t resumed_epoch = 0);
  CheckpointConvergenceState state() const;
  // A resumed run continues even if the checkpoint recorded a stop.
  void clear_persisted_stop_for_resume();

  bool has_best() const { return has_best_; }
  float best_loss() const { return best_loss_; }
  uint32_t best_epoch() const { return best_epoch_; }
  float last_epoch_loss() const { return last_epoch_loss_; }
  std::optional<float> last_val_loss() const { return last_val_loss_; }
  // "val" or "train": the loss the rules judge.
  const char *judged_loss_name() const { return judge_by_val_ ? "val" : "train"; }
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
  const bool judge_by_val_;
  float best_loss_ = 0.0f;
  std::optional<float> last_val_loss_;
  uint32_t last_judged_epoch_ = 0;
  float last_epoch_loss_ = -1.0f;
  uint32_t best_epoch_ = 0;
  uint32_t epochs_without_improvement_ = 0;
  bool has_best_ = false;
  bool stop_requested_ = false;
  StopReason stop_reason_ = StopReason::None;
  std::string stop_message_;
};
