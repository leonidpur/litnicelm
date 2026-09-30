#include "reporters.hpp"

void Reporters::on_checkpoint_load_start() const {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_load_start();
  }
}

void Reporters::on_checkpoint_load_end(bool ok) const {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_load_end(ok);
  }
}

void Reporters::on_checkpoint_save_start(uint64_t optimizer_steps,
                                         uint32_t epoch) const {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_save_start(optimizer_steps, epoch);
  }
}

void Reporters::on_checkpoint_save_end(bool ok) const {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_save_end(ok);
  }
}
