// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "attention_dims.hpp"
#include "i_position_encoding.hpp"
#include "i_self_attention.hpp"
#include "linear_params.hpp"
#include "ops.hpp"
#include "slice_request.hpp"
#include "training_observer.hpp"

#include <vector>

class TrainingDiagnosticsController;

class SelfAttentionFusedInplaceMultistream final : public ISelfAttention {
public:
  SelfAttentionFusedInplaceMultistream(int layer_index, const Config &cfg,
                                       const AttentionParams &params,
                                       const Ops &ops,
                                       IPositionEncoding &position_encoding,
                                       NodeSlices slices);

  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  int idx_;
  const Config &cfg_;
  AttentionParams params_;
  const Ops &ops_;
  IPositionEncoding &positionEncoding_;
  // Full extent; the backward ones are empty in an inference layout.
  struct Slices {
    TensorView qkv;
    TensorView context;
    TensorView weights_cache;
    TensorView dcontext;
    TensorView dqkv;
    TensorView dweights;
    TensorView dscores;
  };
  Slices slices_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_qkv_;
  TensorView cache_context_;
  ITrainingObserver *observer_ = &default_training_observer();
};
