#include "convergence_monitor.hpp"
#include "logger.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace {

std::string format_loss(float loss) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << loss;
  return oss.str();
}

} // namespace

ConvergenceMonitor::ConvergenceMonitor(const Config &cfg) : cfg_(cfg) {}

ConvergenceVerdict ConvergenceMonitor::observe(uint32_t epoch,
                                               const EpochMetrics &metrics) {
  const float mean_loss = metrics.train_loss;
  last_epoch_loss_ = mean_loss;

  if (!std::isfinite(mean_loss)) {
    if (cfg_.training.stop_on_nonfinite_loss) {
      request_stop(StopReason::NonFiniteLoss,
                   "epoch train loss became non-finite at epoch " +
                       std::to_string(epoch));
    }
    return verdict(/*new_best=*/false);
  }

  bool new_best = false;
  if (!has_best_ || loss_improved(mean_loss)) {
    const bool had_best = has_best_;
    const float previous_best = best_loss_;
    has_best_ = true;
    best_loss_ = mean_loss;
    best_epoch_ = epoch;
    epochs_without_improvement_ = 0;
    new_best = true;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << "train loss " << best_loss_ << " at epoch " << best_epoch_;
    if (had_best) {
      oss << " (previous " << previous_best << ")";
    }
    logger().log(EventId::NewBest, oss.str(),
                LogFields()
                    .u64("epoch", best_epoch_)
                    .f64("loss", best_loss_)
                    .str("path", cfg_.paths.model_file_best));
  } else {
    if (epoch >= cfg_.training.min_epochs) {
      epochs_without_improvement_ += 1;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << epoch << ": train " << mean_loss
        << ", best train " << best_loss_
        << ", min_delta " << cfg_.training.min_delta;
    if (cfg_.training.patience_epochs > 0) {
      if (epoch >= cfg_.training.min_epochs) {
        oss << " | patience " << epochs_without_improvement_ << "/"
            << cfg_.training.patience_epochs;
      } else {
        oss << " | patience starts at epoch "
            << cfg_.training.min_epochs;
      }
    }
    logger().log(EventId::NoImprovement, oss.str(),
                LogFields()
                    .u64("epoch", epoch)
                    .f64("loss", mean_loss)
                    .f64("best_loss", best_loss_)
                    .u64("epochs_without_improvement",
                         epochs_without_improvement_));
  }

  if (cfg_.training.target_loss >= 0.0f &&
      mean_loss <= cfg_.training.target_loss) {
    request_stop(StopReason::TargetLoss,
                 "target loss reached at epoch " + std::to_string(epoch) +
                     " (train loss " + format_loss(mean_loss) + ")");
  }

  if (cfg_.training.patience_epochs > 0 &&
      epoch >= cfg_.training.min_epochs &&
      epochs_without_improvement_ >= cfg_.training.patience_epochs) {
    request_stop(StopReason::Patience,
                 "no improvement for " +
                     std::to_string(epochs_without_improvement_) +
                     " epoch(s); best train loss " + format_loss(best_loss_) +
                     " at epoch " + std::to_string(best_epoch_));
  }
  return verdict(new_best);
}

ConvergenceVerdict ConvergenceMonitor::verdict(bool new_best) const {
  ConvergenceVerdict v;
  v.new_best = new_best;
  v.decision.decided = true;
  v.decision.continue_training = !stop_requested_;
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
  logger().log(EventId::PersistedStopCleared,
              "(" + restored_stop_reason +
                  "). Current run explicitly requested incremental resume, so "
                  "the persisted stop state and no-improvement counter are "
                  "cleared and training will continue.");
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
    return "non-finite train loss";
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
  logger().log(EventId::EarlyStopRequested, stop_message_,
              LogFields().str("reason", stop_message_));
}
