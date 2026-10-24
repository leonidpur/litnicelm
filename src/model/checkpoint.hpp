#pragma once

#include "backend/device_backend.hpp"
#include <config.hpp>
#include "tensor.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

class IPositionEncoding;
class ModelAlgoFactory;

struct ArenaView {
  void *base = nullptr;
  uint64_t bytes = 0;
  Device device = Device::CPU;
};

struct AdamStateView {
  void *base = nullptr;
  uint64_t bytes = 0;
  Device device = Device::CPU;
};

struct CheckpointConvergenceState {
  static constexpr uint32_t kHasBest = 1u << 0;
  static constexpr uint32_t kStopRequested = 1u << 1;
  static constexpr uint32_t kBestIsVal = 1u << 2;  // best_loss is a val loss

  float best_loss = 0.0f;
  float last_epoch_loss = -1.0f;
  uint32_t best_epoch = 0;
  uint32_t epochs_without_improvement = 0;
  uint32_t flags = 0;
  uint32_t stop_reason = 0;
};

// Thrown when a checkpoint cannot be saved or loaded; the message names the
// checkpoint and the reason.
class CheckpointError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// What a checkpoint restores besides the arenas.
struct LoadedCheckpointState {
  uint64_t step = 0;   // optimizer steps taken
  uint32_t epoch = 0;  // epochs completed
  CheckpointConvergenceState convergence;
};

// Saves and loads checkpoints (format v7) of one model: the model
// dimensions, position encoding, conf version and alignment from cfg; the
// tokenizer, whose fingerprint goes in the path + ".tokfp" sidecar; and
// the parameter and optimizer arenas the payloads come from and go to.
// cfg, backend and the arenas must outlive it.
class CheckpointIO {
public:
  CheckpointIO(const Config &cfg, const ModelAlgoFactory &algo, DeviceBackend &backend,
               const ArenaView &params, const AdamStateView &adam);
  ~CheckpointIO();

  // Writes the checkpoint and its fingerprint sidecar, creating the
  // directory; throws CheckpointError.
  void save(const std::string &path, uint64_t step, uint32_t epoch,
            const CheckpointConvergenceState &convergence) const;
  // Checks the header against this model and the sidecar against this
  // tokenizer, then fills the arenas; throws CheckpointError.
  LoadedCheckpointState load(const std::string &path) const;

private:
  const Config &cfg_;
  std::unique_ptr<IPositionEncoding> position_encoding_;
  DeviceBackend &backend_;
  ArenaView params_;
  AdamStateView adam_;
};
