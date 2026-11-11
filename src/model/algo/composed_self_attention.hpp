// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "attention_dims.hpp"
#include "i_attention_core.hpp"
#include "i_output_projection.hpp"
#include "i_position_encoding.hpp"
#include "i_qkv_projection.hpp"
#include "i_self_attention.hpp"
#include "linear_params.hpp"
#include "slice_request.hpp"
#include "training_observer.hpp"

#include <memory>
#include <vector>

class TrainingDiagnosticsController;

// Self-attention as QKV projection -> position encoding on Q/K -> core ->
// output projection. Owns the tensors passed between the nodes (qkv,
// context and their gradients); each node owns only its own state.
class ComposedSelfAttention final : public ISelfAttention {
public:
  ComposedSelfAttention(int layer_index, const Config &cfg,
                        const AttentionParams &params,
                        IPositionEncoding &position_encoding,
                        NodeSlices slices,
                        std::unique_ptr<IQkvProjection> qkv_projection,
                        std::unique_ptr<IAttentionCore> core,
                        std::unique_ptr<IOutputProjection> output_projection);
  // params: checked here; the projections hold their own.
  // The composite's own slices (qkv, context and their gradients); each
  // node declares its own.
  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  void check_contract(const AttentionParams &params) const;

  int idx_;
  const Config &cfg_;
  IPositionEncoding &positionEncoding_;
  std::unique_ptr<IQkvProjection> qkvProjection_;
  std::unique_ptr<IAttentionCore> core_;
  std::unique_ptr<IOutputProjection> outputProjection_;
  // Full extent; dcontext and dqkv are empty in an inference layout.
  struct Slices {
    TensorView qkv;
    TensorView context;
    TensorView dcontext;
    TensorView dqkv;
  };
  Slices slices_;
  TensorView cache_x_;
  ITrainingObserver *observer_ = &default_training_observer();
};
