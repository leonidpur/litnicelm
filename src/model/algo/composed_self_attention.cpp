// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "composed_self_attention.hpp"
#include "slice_views.hpp"
#include "tensor_contracts.hpp"

#include <litnicelm_utils/checks.hpp>

#include <utility>

ComposedSelfAttention::ComposedSelfAttention(
    int layer_index, const Config &cfg, const AttentionParams &params,
    IPositionEncoding &position_encoding, NodeSlices slices,
    std::unique_ptr<IQkvProjection> qkv_projection,
    std::unique_ptr<IAttentionCore> core,
    std::unique_ptr<IOutputProjection> output_projection)
    : idx_(layer_index),
      cfg_(cfg),
      positionEncoding_(position_encoding),
      qkvProjection_(std::move(qkv_projection)),
      core_(std::move(core)),
      outputProjection_(std::move(output_projection)),
      slices_{slices.find("attn.qkv"), slices.find("attn.context"),
              slices.find("attn.dcontext"), slices.find("attn.dqkv")} {
  check_contract(params);
}

std::vector<SliceRequest>
ComposedSelfAttention::slice_requests(const AttentionDims &dims) {
  const int64_t B = dims.batch_size;
  const int64_t S = dims.seq_len;
  const int64_t D = dims.model_dim;
  std::vector<SliceRequest> requests{
      {"attn.qkv", Shape{B, S, 3 * D}, ItemType::Activation, SliceRole::Output,
       SliceLifetime::Step},
      {"attn.context", Shape{B, S, D}, ItemType::Activation, SliceRole::Output,
       SliceLifetime::Step},
  };
  if (dims.with_backward) {
    requests.push_back({"attn.dcontext", Shape{B, S, D}, ItemType::Activation,
                        SliceRole::Output, SliceLifetime::Op});
    requests.push_back({"attn.dqkv", Shape{B, S, 3 * D}, ItemType::Activation,
                        SliceRole::Output, SliceLifetime::Op});
  }
  return requests;
}

void ComposedSelfAttention::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void ComposedSelfAttention::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  qkvProjection_->set_diagnostics(diagnostics);
  core_->set_diagnostics(diagnostics);
  outputProjection_->set_diagnostics(diagnostics);
}

void ComposedSelfAttention::check_contract(const AttentionParams &params) const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  TensorContracts::check_attention_config(
      model_dim, static_cast<int64_t>(cfg_.model.n_heads),
      "ComposedSelfAttention");
  TensorContracts::check_attention_params(
      params.qkv.weight, params.qkv.bias, params.out.weight, params.out.bias,
      model_dim, "ComposedSelfAttention");
}

void ComposedSelfAttention::forward(const TensorView &x, TensorView &out) {
  observer_->on_attention_start(idx_);
  const TensorContracts::BatchSeqDims dims = TensorContracts::check_bsd_io(
      x, out, static_cast<int64_t>(cfg_.model.d_model), "ComposedSelfAttention");

  TensorView qkv = slice_views::batch_seq(
      slices_.qkv, dims.batch_size, dims.seq_len, "ComposedSelfAttention qkv");
  qkvProjection_->forward(x, qkv);
  positionEncoding_.apply_to_qk(qkv);

  TensorView context = slice_views::batch_seq(
      slices_.context, dims.batch_size, dims.seq_len, "ComposedSelfAttention context");
  core_->forward(qkv, context);
  outputProjection_->forward(context, out);

  cache_x_ = x;
  observer_->on_attention_end(idx_);
}

void ComposedSelfAttention::backward(const TensorView &dout, TensorView &dx) {
  observer_->on_attention_start(idx_);
  checker().check_same_shape(dout, cache_x_, "ComposedSelfAttention dout");
  checker().check_same_shape(dx, cache_x_, "ComposedSelfAttention dx");
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);

  TensorView dcontext = slice_views::batch_seq(
      slices_.dcontext, batch_size, seq_len, "ComposedSelfAttention dcontext");
  outputProjection_->backward(dout, dcontext);

  TensorView dqkv = slice_views::batch_seq(
      slices_.dqkv, batch_size, seq_len, "ComposedSelfAttention dqkv");
  core_->backward(dcontext, dqkv);
  positionEncoding_.backward_qk(dqkv);
  qkvProjection_->backward(dqkv, dx);
  observer_->on_attention_end(idx_);
}
