// Checks LearningRateSchedule: constant, cosine warmup, decay and floor.

#include "learning_rate_schedule.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

bool near(float a, double b) { return std::fabs(a - b) <= 1e-6 * std::fabs(b) + 1e-12; }

TrainingConfig cosine(float peak, uint64_t warmup, float min_ratio) {
  TrainingConfig tc{};
  tc.learning_rate = peak;
  tc.lr_schedule = "cosine";
  tc.warmup_steps = warmup;
  tc.min_lr_ratio = min_ratio;
  return tc;
}

} // namespace

int main() {
  {
    TrainingConfig tc{};
    tc.learning_rate = 3e-4f;
    check(near(LearningRateSchedule(tc, 1000).at(0), 3e-4) &&
              near(LearningRateSchedule(tc, 1000).at(999), 3e-4),
          "constant: same rate at every step");
  }
  {
    const TrainingConfig tc = cosine(1e-3f, 100, 0.1f);
    check(near(LearningRateSchedule(tc, 1100).at(0), 1e-5), "warmup: first step is peak / warmup");
    check(near(LearningRateSchedule(tc, 1100).at(49), 5e-4), "warmup: halfway is half the peak");
    check(near(LearningRateSchedule(tc, 1100).at(99), 1e-3), "warmup: last warmup step reaches the peak");
    // Decay covers steps 101..1100: the midpoint is step 600.
    check(near(LearningRateSchedule(tc, 1100).at(599), 1e-3 * (0.1 + 0.9 * 0.5)),
          "cosine: midpoint is halfway between peak and floor");
    check(near(LearningRateSchedule(tc, 1100).at(1099), 1e-4), "cosine: last planned step is the floor");
    check(near(LearningRateSchedule(tc, 1100).at(5000), 1e-4), "cosine: stays at the floor past the plan");
    check(LearningRateSchedule(tc, 1100).at(300) > LearningRateSchedule(tc, 1100).at(301),
          "cosine: decreasing after warmup");
  }
  {
    const TrainingConfig tc = cosine(1e-3f, 0, 0.0f);
    check(near(LearningRateSchedule(tc, 1000).at(0), 1e-3 * 0.5 * (1.0 + std::cos(std::acos(-1.0) / 1000))),
          "no warmup: decay starts at the first step");
    check(near(LearningRateSchedule(tc, 1000).at(999), 0.0), "min_lr_ratio 0 decays to zero");
  }
  {
    const TrainingConfig tc = cosine(1e-3f, 500, 0.1f);
    check(near(LearningRateSchedule(tc, 200).at(299), 6e-4) &&
              near(LearningRateSchedule(tc, 200).at(799), 1e-3),
          "warmup longer than the run: still warms up, then holds the peak");
  }

  std::cout << (g_failures == 0 ? "All lr schedule tests passed.\n"
                                : "LR schedule tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
