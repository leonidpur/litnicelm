// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "steady_state_observer.hpp"
#include "observers_events.hpp"

#include <litnicelm_utils/log.hpp>

#include <iomanip>
#include <sstream>

SteadyStateObserver::SteadyStateObserver(uint32_t warmup_steps)
    : warmup_steps_(warmup_steps) {}

void SteadyStateObserver::on_batch_start(uint64_t batch_idx) {
  (void)batch_idx;
  batch_timed_ = measuring_;
  if (batch_timed_) batch_start_ = Clock::now();
}

void SteadyStateObserver::on_batch_end(uint64_t batch_idx, double loss) {
  (void)batch_idx;
  (void)loss;
  if (batch_timed_) total_ += Clock::now() - batch_start_;
  batch_timed_ = false;
}

void SteadyStateObserver::on_train_step_end(uint64_t optimizer_steps, double loss) {
  (void)optimizer_steps;
  (void)loss;
  if (warmup_steps_ == 0) return;
  steps_seen_ += 1;
  // The step that ends the warm-up finishes inside a batch that is not timed.
  if (measuring_) measured_steps_ += 1;
  if (steps_seen_ == warmup_steps_) measuring_ = true;
}

void SteadyStateObserver::on_training_end(const TrainingPosition &training_position) {
  (void)training_position;
  if (warmup_steps_ == 0) return;
  const double total_ms =
      std::chrono::duration<double, std::milli>(total_).count();
  std::ostringstream text;
  text << std::fixed << std::setprecision(3) << "after " << warmup_steps_
       << " warm-up optimizer steps: total=" << total_ms / 1000.0
       << " s steps=" << measured_steps_ << " avg="
       << (measured_steps_ == 0 ? 0.0 : total_ms / measured_steps_) << " ms";
  logger().log(observers_events::kSteadyStateSummary, text.str());
}
