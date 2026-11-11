// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "backend/device_backend.hpp"
#include "i_self_attention.hpp"
#include "slice_request.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Debug-mode wrapper (model_algo.debug_guards) around one layer's attention:
//   - guard regions: each requested slice is followed by "<name>.guard",
//     filled with a canary at construction; a write past the end of a slice
//     that reaches its guard fails the next check (before and after every
//     forward and backward);
//   - saved-state check: forward ends by fingerprinting the input x and every
//     Step-lifetime slice; backward first checks that none of them changed in
//     between (a content check, standing in for version counters).
// Failures throw std::runtime_error naming the layer and the slice. Copies
// every checked byte to the host, so it is for debugging only.
class GuardedSelfAttention final : public ISelfAttention {
public:
  // requests: without guards, as the inner attention declared them; slices:
  // bound from with_guards(requests).
  GuardedSelfAttention(int layer_index, std::unique_ptr<ISelfAttention> inner,
                       const std::vector<SliceRequest> &requests,
                       const NodeSlices &slices, DeviceBackend &backend);

  // requests, each followed by its guard slice.
  static std::vector<SliceRequest>
  with_guards(const std::vector<SliceRequest> &requests);

  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;
  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  struct Watched {
    std::string name;
    TensorView view;
    uint64_t fingerprint = 0;
  };

  void check_guards(const char *when) const;
  uint64_t fingerprint(const TensorView &view) const;
  [[noreturn]] void fail(const std::string &what) const;

  int idx_;
  std::unique_ptr<ISelfAttention> inner_;
  DeviceBackend &backend_;
  std::vector<std::pair<std::string, TensorView>> guards_;
  std::vector<Watched> saved_;  // Step-lifetime slices, then x
  bool has_forward_ = false;
};
