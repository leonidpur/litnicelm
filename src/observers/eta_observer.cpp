#include "eta_observer.hpp"

#include <algorithm>
#include <sstream>

namespace {
void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TRAINING, event, step, value, message);
}
} // namespace

EtaObserver::EtaObserver(const Config &cfg, const Command &cmd, ReportSink *sink,
                         uint32_t epoch_report_every)
    : cfg_(cfg),
      cmd_(cmd),
      sink_(sink),
      epoch_report_every_(std::max<uint32_t>(1, epoch_report_every)) {}

void EtaObserver::on_training_start() {
  start_time_ = std::chrono::steady_clock::now();
  ms_per_epoch_avg_ = 0;
}

void EtaObserver::on_epoch_end(uint32_t epoch, float mean_loss,
                               uint64_t global_step) {
  (void)global_step;
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
  ms_per_epoch_avg_ = (epoch > 0) ? (elapsed / epoch) : 0;
  if ((epoch % epoch_report_every_) == 0) {
    std::ostringstream oss;
    oss << "Epoch " << epoch << " mean_loss=" << mean_loss
        << " " << get_eta_report(epoch);
    report_if(sink_, ReportEvent::STEP_COMPLETE, epoch, mean_loss, oss.str());
  }
}

void EtaObserver::finalize(uint64_t global_step, uint32_t epoch) {
  if (!is_estimation_mode()) {
    return;
  }
  report_if(sink_, ReportEvent::END, static_cast<uint32_t>(global_step), 0.0f,
            get_eta_report(epoch));
}

bool EtaObserver::is_estimation_mode() const {
  return cmd_.target == Command::Target::DRY_RUN || cfg_.training.dry_run;
}

uint32_t EtaObserver::total_epochs() const {
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

  const uint32_t train_total = total_epochs();
  const int64_t remaining_epochs =
      std::max<int64_t>(0, static_cast<int64_t>(train_total) -
                               static_cast<int64_t>(current_epoch));
  const int64_t remaining = remaining_epochs * ms_per_epoch_avg_;
  return "[TRAIN] Epoch " + std::to_string(current_epoch) + "/" +
         std::to_string(train_total) + " | Rem: " + format_duration(remaining);
}
