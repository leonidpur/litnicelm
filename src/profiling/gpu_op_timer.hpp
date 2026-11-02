#pragma once

#include "backend/backend_observer.hpp"
#include "backend/device_backend.hpp"

#include <string>

class GpuOpTimer final : public IBackendObserver {
public:
  void on_backend_ready(DeviceBackend &backend) override;
  void start();
  void stop();
  void on_op_begin(OpId op) override;
  void on_op_end(OpId op) override;
  std::string summary();

private:
  DeviceBackend *backend_ = nullptr;
  bool enabled_ = false;
};
