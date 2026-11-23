// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "training_observer.hpp"

#include <chrono>
#include <cstdint>

// Steady-state batch time: the sum of batch_start..batch_end wall times for
// every batch after the first warmup_steps optimizer steps of this process,
// so one-time setup (allocation, cuDNN graph build and plan selection) is
// excluded. Inactive when warmup_steps is 0.
class SteadyStateObserver final : public ITrainingObserver {
public:
  explicit SteadyStateObserver(uint32_t warmup_steps);

  void on_batch_start(uint64_t batch_idx) override;
  void on_batch_end(uint64_t batch_idx, double loss) override;
  void on_train_step_end(uint64_t optimizer_steps, double loss) override;
  void on_training_end(const TrainingPosition &training_position) override;

private:
  using Clock = std::chrono::steady_clock;

  const uint32_t warmup_steps_;
  uint64_t steps_seen_ = 0;
  uint64_t measured_steps_ = 0;
  bool measuring_ = false;
  bool batch_timed_ = false;
  Clock::time_point batch_start_{};
  Clock::duration total_{};
};
