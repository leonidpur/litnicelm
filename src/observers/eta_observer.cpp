#include "eta_observer.hpp"

#include <algorithm>
#include <sstream>

namespace {
void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  if (sink == nullptr) {
    return;
  }
  const char *event_name =
      event == ReportEvent::STEP_COMPLETE ? "epoch complete"
                                          : report_utils::event_name(event);
  sink->report(event, std::string("[EtaObserver][") +
                          event_name + "] step=" +
                          std::to_string(step) + " val=" +
                          std::to_string(value) + " " + message);
}
} // namespace

EtaObserver::EtaObserver(const Config &cfg, const Command &cmd, ReportSink *sink,
                         uint32_t epoch_report_every)
    : cfg_(cfg),
      cmd_(cmd),
      sink_(sink),
      epoch_report_every_(std::max<uint32_t>(1, epoch_report_every)) {}

void EtaObserver::on_training_start(const TrainingPosition &training_position,
                                    uint64_t steps_per_epoch,
                                    ReportSink *sink) {
  (void)training_position;
  (void)steps_per_epoch;
  (void)sink;
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
                               ReportSink *sink,
                               const ArenaView &data_arena,
                               const AdamStateView &adam_state) {
  (void)device_backend;
  (void)sink;
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
    oss << "Epoch " << epoch << " mean_loss=" << metrics.train_loss;
    if (metrics.val_loss) {
      oss << " val_loss=" << *metrics.val_loss;
    }
    oss << " | Epoch time: " << format_duration(epoch_elapsed)
        << " | Total time: " << format_duration(elapsed)
        << " " << get_eta_report(epoch);
    report_if(sink_, ReportEvent::STEP_COMPLETE,
              static_cast<uint32_t>(training_position.optimizer_steps), metrics.train_loss,
              oss.str());
  }
  return {};
}

void EtaObserver::on_training_end(const TrainingPosition &training_position, ReportSink *sink) {
  (void)sink;
  if (!is_estimation_mode()) {
    return;
  }
  report_if(sink_, ReportEvent::END,
            static_cast<uint32_t>(training_position.optimizer_steps), 0.0f,
            get_eta_report(training_position.epoch));
}

bool EtaObserver::is_estimation_mode() const {
  return cmd_.target == Command::Target::DRY_RUN || cfg_.training.dry_run;
}

uint32_t EtaObserver::last_epoch() const {
  return is_estimation_mode() ? cfg_.training.num_epochs_dry_run
                              : cfg_.training.num_epochs_train;
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
    return "Calculating...";
  }

  if (is_estimation_mode()) {
    const uint32_t dry_total = cfg_.training.num_epochs_dry_run;
    const uint32_t train_total = cfg_.training.num_epochs_train;
    const int64_t projected_total =
        static_cast<int64_t>(train_total) * ms_per_epoch_avg_;
    const int64_t projected_remaining =
        std::max<int64_t>(
            0, static_cast<int64_t>(train_total) -
                   static_cast<int64_t>(current_epoch)) *
        ms_per_epoch_avg_;

    return "[ESTIMATE] Dry-run " + std::to_string(current_epoch) + "/" +
           std::to_string(dry_total) + " | Projected full training: " +
           format_duration(projected_total) + " | Remaining if continued: " +
           format_duration(projected_remaining);
  }

  const uint32_t train_total = last_epoch();
  const int64_t remaining_epochs =
      std::max<int64_t>(0, static_cast<int64_t>(train_total) -
                               static_cast<int64_t>(current_epoch));
  const int64_t remaining = remaining_epochs * ms_per_epoch_avg_;
  return "Epoch " + std::to_string(current_epoch) + "/" +
         std::to_string(train_total) + " | Rem: " + format_duration(remaining);
}
