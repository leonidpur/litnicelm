#include "learning_rate_schedule.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

LearningRateSchedule::LearningRateSchedule(const TrainingConfig &tc,
                                           uint64_t total_steps)
    : tc_(tc), total_steps_(total_steps) {}

float LearningRateSchedule::at(uint64_t completed_steps) const {
  const double peak = tc_.learning_rate;
  if (tc_.lr_schedule != "cosine") {
    return static_cast<float>(peak);
  }
  const uint64_t step = completed_steps + 1;  // 1-based step being taken
  if (step <= tc_.warmup_steps) {
    return static_cast<float>(peak * static_cast<double>(step) /
                              static_cast<double>(tc_.warmup_steps));
  }
  if (total_steps_ <= tc_.warmup_steps) {
    return static_cast<float>(peak);
  }
  const double floor = peak * tc_.min_lr_ratio;
  const double progress = std::min(
      1.0, static_cast<double>(step - tc_.warmup_steps) /
               static_cast<double>(total_steps_ - tc_.warmup_steps));
  const double pi = std::acos(-1.0);
  return static_cast<float>(floor + (peak - floor) * 0.5 *
                                        (1.0 + std::cos(pi * progress)));
}

void LearningRateSchedule::report() const {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(2);
  if (tc_.lr_schedule != "cosine") {
    oss << "constant " << tc_.learning_rate;
  } else {
    oss << "cosine, warmup " << tc_.warmup_steps << " steps to "
        << tc_.learning_rate << ", then down to "
        << tc_.learning_rate * tc_.min_lr_ratio << " at step " << total_steps_;
  }
  logger().log(EventId::LrSchedule, oss.str(),
               LogFields()
                   .str("schedule", tc_.lr_schedule)
                   .f64("learning_rate", tc_.learning_rate)
                   .u64("warmup_steps", tc_.warmup_steps)
                   .f64("min_lr_ratio", tc_.min_lr_ratio)
                   .u64("total_steps", total_steps_));
  if (tc_.lr_schedule == "cosine" && tc_.warmup_steps >= total_steps_) {
    logger().log(EventId::LrWarmupTooLong,
                 "warmup_steps " + std::to_string(tc_.warmup_steps) +
                     " >= planned steps " + std::to_string(total_steps_) +
                     "; the rate never reaches its peak");
  }
}
