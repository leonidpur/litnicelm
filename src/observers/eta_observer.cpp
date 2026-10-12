#include "eta_observer.hpp"
#include <litnicelm_utils/log.hpp>
#include "observers_events.hpp"

#include <algorithm>
#include <sstream>

EtaObserver::EtaObserver(const Config &cfg, const TrainingPlan &plan,
                         uint32_t epoch_report_every)
    : cfg_(cfg),
      plan_(plan),
      epoch_report_every_(std::max<uint32_t>(1, epoch_report_every)) {}

void EtaObserver::on_training_start(const TrainingPosition &training_position,
                                    uint64_t batches_per_epoch) {
  (void)training_position;
  (void)batches_per_epoch;
  start_time_ = std::chrono::steady_clock::now();
  epoch_start_time_ = start_time_;
  ms_per_epoch_avg_ = 0;
}

void EtaObserver::on_epoch_start(uint32_t epoch) {
  (void)epoch;
  epoch_start_time_ = std::chrono::steady_clock::now();
}

ContinueTrainingDecision EtaObserver::on_epoch_end(uint32_t epoch,
                                                   const EpochMetrics &metrics,
                               TrainingPosition &training_position,
                               DeviceBackend &device_backend,
                               const ArenaView &data_arena,
                               const AdamStateView &adam_state) {
  (void)metrics;
  (void)device_backend;
  (void)data_arena;
  (void)adam_state;
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
  const auto epoch_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now - epoch_start_time_)
          .count();
  ms_per_epoch_avg_ = (epoch > 0) ? (elapsed / epoch) : 0;
  if ((epoch % epoch_report_every_) == 0) {
    std::ostringstream oss;
    oss << "Epoch " << epoch << "/" << plan_.last_epoch << ", step "
        << training_position.optimizer_steps << " | epoch "
        << format_duration(epoch_elapsed) << ", total "
        << format_duration(elapsed) << ", " << get_eta_report(epoch);
    logger().log(observers_events::kEpochProgress, oss.str());
  }
  return {};
}

void EtaObserver::on_training_end(const TrainingPosition &training_position) {
  if (!plan_.dry_run()) {
    return;
  }
  logger().log(observers_events::kEtaSummary,
               get_dry_run_estimate(training_position.epoch));
}

std::string EtaObserver::format_duration(int64_t ms) const {
  if (ms < 0) {
    ms = 0;
  }

  const int64_t total_seconds = ms / 1000;
  const int64_t hours = total_seconds / 3600;
  const int64_t minutes = (total_seconds % 3600) / 60;
  const int64_t seconds = total_seconds % 60;

  std::ostringstream oss;
  if (hours > 0) {
    oss << hours << "h " << minutes << "m " << seconds << "s";
  } else if (minutes > 0) {
    oss << minutes << "m " << seconds << "s";
  } else {
    oss << seconds << "s";
  }
  return oss.str();
}

std::string EtaObserver::get_eta_report(uint32_t current_epoch) const {
  if (ms_per_epoch_avg_ == 0) {
    return "remaining: calculating...";
  }
  const int64_t remaining_epochs =
      std::max<int64_t>(0, static_cast<int64_t>(plan_.last_epoch) -
                               static_cast<int64_t>(current_epoch));
  return "remaining " + format_duration(remaining_epochs * ms_per_epoch_avg_);
}

std::string EtaObserver::get_dry_run_estimate(uint32_t current_epoch) const {
  if (ms_per_epoch_avg_ == 0) {
    return "Dry-run estimate: no epoch measured";
  }
  const uint32_t train_total = cfg_.training.num_epochs_train;
  const int64_t projected_total =
      static_cast<int64_t>(train_total) * ms_per_epoch_avg_;
  const int64_t projected_remaining =
      std::max<int64_t>(0, static_cast<int64_t>(train_total) -
                               static_cast<int64_t>(current_epoch)) *
      ms_per_epoch_avg_;
  return "Dry-run estimate from " + std::to_string(current_epoch) +
         " epoch(s): full training (" + std::to_string(train_total) +
         " epochs) " + format_duration(projected_total) +
         ", remaining if continued " + format_duration(projected_remaining);
}
