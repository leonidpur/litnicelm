#pragma once

#include "training_observer.hpp"

#include <config.hpp>
#include <report_interface.hpp>
#include <types.hpp>

#include <chrono>
#include <cstdint>
#include <string>

class EtaObserver final : public ITrainingObserver {
public:
  EtaObserver(const Config &cfg, const Command &cmd, ReportSink *sink,
              uint32_t epoch_report_every);

  void on_training_start() override;
  void on_epoch_end(uint32_t epoch, float mean_loss,
                    uint64_t global_step) override;
  void finalize(uint64_t global_step, uint32_t epoch) override;

private:
  bool is_estimation_mode() const;
  uint32_t total_epochs() const;
  std::string format_duration(int64_t ms) const;
  std::string get_eta_report(uint32_t current_epoch) const;

  const Config &cfg_;
  const Command &cmd_;
  ReportSink *sink_ = nullptr;
  uint32_t epoch_report_every_ = 1;
  std::chrono::steady_clock::time_point start_time_{};
  int64_t ms_per_epoch_avg_ = 0;
};
