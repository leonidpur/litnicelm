#include "convergence_monitor.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
constexpr const char *kAnsiGreen = "\033[32m";
constexpr const char *kAnsiYellow = "\033[33m";
constexpr const char *kAnsiReset = "\033[0m";
} // namespace

ConvergenceMonitor::ConvergenceMonitor(const Config &cfg,
                                       const Reporters &reporters)
    : cfg_(cfg), reporters_(reporters) {}

ConvergenceVerdict ConvergenceMonitor::observe(uint32_t epoch,
                                               const EpochMetrics &metrics) {
  const float mean_loss = metrics.train_loss;
  last_epoch_loss_ = mean_loss;

  if (!std::isfinite(mean_loss)) {
    if (cfg_.training.stop_on_nonfinite_loss) {
      request_stop(StopReason::NonFiniteLoss,
                   "epoch loss became non-finite at epoch " +
                       std::to_string(epoch));
    }
    return verdict(/*new_best=*/false);
  }

  bool new_best = false;
  if (!has_best_ || loss_improved(mean_loss)) {
    has_best_ = true;
    best_loss_ = mean_loss;
    best_epoch_ = epoch;
    epochs_without_improvement_ = 0;
    new_best = true;
    // Formatted like std::cout, which this line was written to directly.
    std::ostringstream oss;
    oss.copyfmt(std::cout);
    oss << "[MC&CListener] New best loss " << best_loss_ << " at epoch "
        << best_epoch_ << ". Best checkpoint path: "
        << cfg_.paths.model_file_best;
    reporters_.message(oss.str());
  } else {
    if (epoch >= cfg_.training.min_epochs) {
      epochs_without_improvement_ += 1;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "[MC&CListener] No new best at epoch " << epoch
        << ": mean_loss=" << mean_loss
        << ", best=" << best_loss_
        << ", min_delta=" << cfg_.training.min_delta;
    if (cfg_.training.patience_epochs > 0) {
      if (epoch >= cfg_.training.min_epochs) {
        oss << " | patience " << epochs_without_improvement_ << "/"
            << cfg_.training.patience_epochs;
      } else {
        oss << " | patience starts at epoch "
            << cfg_.training.min_epochs;
      }
    }
    reporters_.message(kAnsiYellow + oss.str() + kAnsiReset);
  }

  if (cfg_.training.target_loss >= 0.0f &&
      mean_loss <= cfg_.training.target_loss) {
    request_stop(StopReason::TargetLoss,
                 "target loss reached at epoch " + std::to_string(epoch) +
                     " (loss=" + std::to_string(mean_loss) + ")");
  }

  if (cfg_.training.patience_epochs > 0 &&
      epoch >= cfg_.training.min_epochs &&
      epochs_without_improvement_ >= cfg_.training.patience_epochs) {
    request_stop(StopReason::Patience,
                 "no improvement for " +
                     std::to_string(epochs_without_improvement_) +
                     " epoch(s); best loss=" + std::to_string(best_loss_) +
                     " at epoch " + std::to_string(best_epoch_));
  }
  return verdict(new_best);
}

ConvergenceVerdict ConvergenceMonitor::verdict(bool new_best) const {
  ConvergenceVerdict v;
  v.new_best = new_best;
  v.decision.decided = true;
  v.decision.continue_training = !stop_requested_;
  if (stop_requested_) {
    v.decision.early_stop_message = stop_reason_text();
  }
  return v;
}

void ConvergenceMonitor::reset() {
  best_loss_ = 0.0f;
  last_epoch_loss_ = -1.0f;
  best_epoch_ = 0;
  epochs_without_improvement_ = 0;
  has_best_ = false;
  stop_requested_ = false;
  stop_reason_ = StopReason::None;
  stop_message_.clear();
}

void ConvergenceMonitor::restore(const CheckpointConvergenceState &state) {
  best_loss_ = state.best_loss;
  last_epoch_loss_ = state.last_epoch_loss;
  best_epoch_ = state.best_epoch;
  epochs_without_improvement_ = state.epochs_without_improvement;
  has_best_ = (state.flags & CheckpointConvergenceState::kHasBest) != 0;
  stop_requested_ =
      (state.flags & CheckpointConvergenceState::kStopRequested) != 0;
  stop_reason_ = static_cast<StopReason>(state.stop_reason);
  stop_message_.clear();
}

CheckpointConvergenceState ConvergenceMonitor::state() const {
  CheckpointConvergenceState state;
  state.best_loss = best_loss_;
  state.last_epoch_loss = last_epoch_loss_;
  state.best_epoch = best_epoch_;
  state.epochs_without_improvement = epochs_without_improvement_;
  if (has_best_) {
    state.flags |= CheckpointConvergenceState::kHasBest;
  }
  if (stop_requested_) {
    state.flags |= CheckpointConvergenceState::kStopRequested;
  }
  state.stop_reason = static_cast<uint32_t>(stop_reason_);
  return state;
}

void ConvergenceMonitor::clear_persisted_stop_for_resume() {
  const std::string restored_stop_reason = stop_reason_text();
  stop_requested_ = false;
  stop_reason_ = StopReason::None;
  stop_message_.clear();
  epochs_without_improvement_ = 0;
  reporters_.message(
      std::string(kAnsiGreen) +
      "[MC&CListener] Resumed checkpoint carries a prior early-stop state (" +
      restored_stop_reason +
      "). Current run explicitly requested incremental resume, so the "
      "persisted stop state and no-improvement counter are cleared "
      "and training will continue." +
      kAnsiReset);
}

std::string ConvergenceMonitor::stop_reason_text() const {
  if (!stop_message_.empty()) {
    return stop_message_;
  }
  switch (stop_reason_) {
  case StopReason::TargetLoss:
    return "target loss reached";
  case StopReason::Patience:
    return "patience exhausted";
  case StopReason::NonFiniteLoss:
    return "non-finite loss";
  case StopReason::None:
  default:
    return "none";
  }
}

bool ConvergenceMonitor::loss_improved(float mean_loss) const {
  return (best_loss_ - mean_loss) > cfg_.training.min_delta;
}

void ConvergenceMonitor::request_stop(StopReason reason,
                                      const std::string &message) {
  if (stop_requested_) {
    return;
  }
  stop_requested_ = true;
  stop_reason_ = reason;
  stop_message_ = message;
  reporters_.message("[MC&CListener] Early stop requested: " + stop_message_);
}
