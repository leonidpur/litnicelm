#pragma once

#include <config.hpp>

#include <cstdint>

// The run's learning rate per optimizer step, from training.lr_schedule:
// "constant" keeps training.learning_rate; "cosine" rises linearly over
// training.warmup_steps, then decays along a cosine to
// learning_rate * min_lr_ratio at total_steps, and stays there. Fixed once
// built; the optimizer only receives the rate.
class LearningRateSchedule {
public:
  LearningRateSchedule(const TrainingConfig &tc, uint64_t total_steps);

  // Rate for the step that follows completed_steps optimizer steps.
  float at(uint64_t completed_steps) const;
  uint64_t total_steps() const { return total_steps_; }
  // Logs the schedule, and a warning when warmup outlasts the run.
  void report() const;

private:
  const TrainingConfig &tc_;
  const uint64_t total_steps_;
};
