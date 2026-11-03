// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "backend/backend_observer.hpp"
#include "backend/device_backend.hpp"

#include <string>

class GpuOpTimer final : public IBackendObserver {
public:
  GpuOpTimer() = default;
  GpuOpTimer(uint32_t skip_batches, uint32_t requested_steps,
             std::string attention_mode);
  void on_backend_ready(DeviceBackend &backend) override;
  void start();
  void stop();
  void on_batch_end();
  void on_optimizer_step_start();
  void on_optimizer_step_end();
  void on_op_begin(OpId op) override;
  void on_op_end(OpId op) override;
  std::string summary();

private:
  DeviceBackend *backend_ = nullptr;
  uint32_t skip_batches_ = 0;
  uint32_t requested_steps_ = 1;
  uint32_t completed_batches_ = 0;
  uint32_t recorded_steps_ = 0;
  uint64_t mark_capacity_ = 4096;
  std::string attention_mode_ = "unknown";
  bool enabled_ = false;
};
