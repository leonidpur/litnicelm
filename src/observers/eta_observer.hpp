#pragma once

#include "training_observer.hpp"
#include "training_plan.hpp"

#include <config.hpp>
#include <types.hpp>

#include <chrono>
#include <cstdint>
#include <string>

class EtaObserver final : public ITrainingObserver {
public:
  EtaObserver(const Config &cfg, const TrainingPlan &plan,
              uint32_t epoch_report_every);

  void on_training_start(const TrainingPosition &training_position,
                         uint64_t steps_per_epoch) override;
  void on_epoch_start(uint32_t epoch) override;
  ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                        const EpochMetrics &metrics,
                    TrainingPosition &training_position,
                    DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
  void on_training_end(const TrainingPosition &training_position) override;

private:
  std::string format_duration(int64_t ms) const;
  // Time left in this run (training or dry run).
  std::string get_eta_report(uint32_t current_epoch) const;
  // Dry run only: full-training time projected from the measured epochs.
  std::string get_dry_run_estimate(uint32_t current_epoch) const;

  const Config &cfg_;
  const TrainingPlan plan_;
  uint32_t epoch_report_every_ = 1;
  std::chrono::steady_clock::time_point start_time_{};
  std::chrono::steady_clock::time_point epoch_start_time_{};
  int64_t ms_per_epoch_avg_ = 0;
};
