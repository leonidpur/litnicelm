#pragma once

#include "training_observer.hpp"

#include <cstdint>
#include <memory>
#include <vector>

// Report-only access to the training observers, shared by the session
// controller and the components it runs (checkpointing, convergence) for
// their own reports. Lifecycle calls (training/epoch start and end) stay with
// the controller.
class Reporters {
public:
  explicit Reporters(
      const std::vector<std::unique_ptr<ITrainingObserver>> &observers)
      : observers_(observers) {}

  void on_checkpoint_load_start() const;
  void on_checkpoint_load_end(bool ok) const;
  void on_checkpoint_save_start(uint64_t optimizer_steps, uint32_t epoch) const;
  void on_checkpoint_save_end(bool ok) const;

private:
  const std::vector<std::unique_ptr<ITrainingObserver>> &observers_;
};
