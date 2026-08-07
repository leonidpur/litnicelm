#include "self_attention_fused_inplace.hpp"
#include "training_diagnostics_controller.hpp"
#include <utils/assert.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("SelfAttentionFusedInplace: ") + std::string(msg);      \
  })

SelfAttentionFusedInplace::SelfAttentionFusedInplace(
    int layer_index, const Config &cfg, TensorStore &tensor_store,
    GradientStore *gradient_store, Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorStore_(tensor_store),
      gradientStore_(gradient_store),
      ops_(ops) {
  validate_contract();
}

void SelfAttentionFusedInplace::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void SelfAttentionFusedInplace::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  require(diagnostics != nullptr, "diagnostics must be non-null");
  diagnostics_ = diagnostics;
}

void SelfAttentionFusedInplace::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t num_heads = static_cast<int64_t>(cfg_.model.n_heads);
  require(num_heads > 0, "n_heads must be > 0");
  require((model_dim % num_heads) == 0,
          "d_model must be divisible by n_heads");

  const TensorView &Wqkv = tensorStore_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorStore_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorStore_.param_attn_out_w(idx_);
  const TensorView &bo = tensorStore_.param_attn_out_b(idx_);

  require(Wqkv.dim(0) == model_dim && Wqkv.dim(1) == 3 * model_dim,
          "Wqkv must be [D, 3D]");
  require(bqkv.dim(0) == 1 && bqkv.dim(1) == 3 * model_dim,
          "bqkv must be [1, 3D]");
  require(Wo.dim(0) == model_dim && Wo.dim(1) == model_dim,
          "Wo must be [D, D]");
  require(bo.dim(0) == 1 && bo.dim(1) == model_dim,
          "bo must be [1, D]");

  require(Wqkv.device() == bqkv.device() && Wqkv.device() == Wo.device() &&
              Wqkv.device() == bo.device(),
          "attention parameter devices must match");
  require(Wqkv.dtype() == bqkv.dtype() && Wqkv.dtype() == Wo.dtype() &&
              Wqkv.dtype() == bo.dtype(),
          "attention parameter dtypes must match");
}

void SelfAttentionFusedInplace::forward(const TensorView &x, TensorView &out) {
  observer_->on_attention_start(idx_);
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  require(x.rank() == 3, "x must be semantic [B, S, D]");
  const int64_t batch_size = x.dim(0);
  const int64_t seq_len = x.dim(1);
  const int64_t model_dim = x.dim(2);
  const int64_t token_rows =
      static_cast<int64_t>(x.numel() / static_cast<uint64_t>(model_dim));
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model),
          "x.dim(2) != d_model");
  require(out.rank() == 3 && out.dim(0) == batch_size &&
              out.dim(1) == seq_len && out.dim(2) == model_dim,
          "out must be semantic [B, S, D]");
  require(x.dim(0) > 0 && x.dim(1) > 0 && x.dim(2) == model_dim &&
              x.dim(0) * x.dim(1) == token_rows,
          "x must be semantic [B, S, D]");

  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &Wqkv = tensorStore_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorStore_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorStore_.param_attn_out_w(idx_);
  const TensorView &bo = tensorStore_.param_attn_out_b(idx_);

  require(Wqkv.device() == x.device() && Wo.device() == x.device(),
          "param device mismatch");
  require(Wqkv.dtype() == x.dtype() && Wo.dtype() == x.dtype(),
          "param dtype mismatch");

  TensorView qkv = tensorStore_.temp_attn_qkv(idx_, batch_size, seq_len);
  ops_.matmul(x, Wqkv, qkv);
  ops_.add_bias_rowwise(qkv, bqkv, qkv);

  TensorView Q = qkv.subcols(0, model_dim);
  TensorView K = qkv.subcols(model_dim, model_dim);
  TensorView V = qkv.subcols(2 * model_dim, model_dim);

  TensorView context = tensorStore_.temp_attn_context(idx_, batch_size, seq_len);
  ops_.fill(context, 0.0f);

  TensorView scores = tensorStore_.temp_attn_scores(idx_, batch_size, seq_len);
  TensorView cached_weights =
      tensorStore_.temp_attn_cached_weights(idx_, batch_size, seq_len);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;

    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView context_h = context.subcols(col0, dh);
    TensorView weights_h = cached_weights.select(0, h);

    ops_.matmul_right_transposed(Qh, Kh, scores);
    ops_.scaled_causal_softmax_rows(scores, scale, weights_h);
    ops_.matmul(weights_h, Vh, context_h);
  }

  ops_.matmul(context, Wo, out);
  ops_.add_bias_rowwise(out, bo, out);

  cache_x_ = x;
  cache_qkv_ = qkv;
  cache_context_ = context;
  has_cache_ = true;
  observer_->on_attention_end(idx_);
}

void SelfAttentionFusedInplace::backward(const TensorView &dout,
                                         TensorView &dx) {
  observer_->on_attention_start(idx_);
  require(gradientStore_ != nullptr, "backward requires gradient store");
  require(diagnostics_ != nullptr, "backward requires diagnostics controller");
  require(has_cache_, "backward called before forward");
  require(dout.rank() == cache_x_.rank() && dout.dim(0) == cache_x_.dim(0) &&
              dout.dim(1) == cache_x_.dim(1) && dout.dim(2) == cache_x_.dim(2),
          "dout shape mismatch");
  require(dx.rank() == cache_x_.rank() && dx.dim(0) == cache_x_.dim(0) &&
              dx.dim(1) == cache_x_.dim(1) && dx.dim(2) == cache_x_.dim(2),
          "dx shape mismatch");

  const int64_t token_rows = cache_x_.dim(0) * cache_x_.dim(1);
  const int64_t model_dim = cache_x_.dim(2);
  require(cache_x_.rank() == 3 && cache_x_.dim(0) > 0 &&
              cache_x_.dim(1) > 0 && cache_x_.dim(2) == model_dim &&
              cache_x_.dim(0) * cache_x_.dim(1) == token_rows,
          "cached x must be semantic [B, S, D]");
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &Wqkv = tensorStore_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorStore_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorStore_.param_attn_out_w(idx_);
  const TensorView &bo = tensorStore_.param_attn_out_b(idx_);
  TensorView dWo = gradientStore_->grad_for_param(Wo);
  ops_.matmul_left_transposed(cache_context_, dout, dWo);
  diagnostics_->bk_attn_dWo(idx_, dWo);
  TensorView dbo = gradientStore_->grad_for_param(bo);
  ops_.row_sum(dout, dbo);
  diagnostics_->bk_attn_dbo(idx_, dbo);

  TensorView dcontext = tensorStore_.temp_attn_dcontext(idx_, batch_size, seq_len);
  ops_.matmul_right_transposed(dout, Wo, dcontext);
  diagnostics_->bk_attn_dcontext(idx_, dcontext);

  TensorView dqkv = tensorStore_.temp_attn_dqkv(idx_, batch_size, seq_len);
  ops_.fill(dqkv, 0.0f);
  TensorView dQ = dqkv.subcols(0, model_dim);
  TensorView dK = dqkv.subcols(model_dim, model_dim);
  TensorView dV = dqkv.subcols(2 * model_dim, model_dim);

  TensorView Q = cache_qkv_.subcols(0, model_dim);
  TensorView K = cache_qkv_.subcols(model_dim, model_dim);
  TensorView V = cache_qkv_.subcols(2 * model_dim, model_dim);
  TensorView cached_weights =
      tensorStore_.temp_attn_cached_weights(idx_, batch_size, seq_len);

  TensorView dweights =
      tensorStore_.temp_attn_dweights(idx_, batch_size, seq_len);
  TensorView dscores =
      tensorStore_.temp_attn_dscores(idx_, batch_size, seq_len);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;
    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView dQh = dQ.subcols(col0, dh);
    TensorView dKh = dK.subcols(col0, dh);
    TensorView dVh = dV.subcols(col0, dh);
    TensorView dhead = dcontext.subcols(col0, dh);
    TensorView weights_h = cached_weights.select(0, h);

    ops_.matmul_right_transposed(dhead, Vh, dweights);
    diagnostics_->bk_attn_dweights(idx_, h, dweights);
    ops_.matmul_left_transposed(weights_h, dhead, dVh);
    diagnostics_->bk_attn_dVh(idx_, h, dVh);

    ops_.softmax_backward_causal_rows(weights_h, dweights, dscores);
    diagnostics_->bk_attn_dscores_softmax_backward(idx_, h, dscores);
    diagnostics_->bk_attn_dscores_masked(idx_, h, dscores);

    ops_.matmul(dscores, Kh, dQh);
    ops_.mul_scalar(dQh, scale, dQh);
    diagnostics_->bk_attn_dQh(idx_, h, dQh);

    ops_.matmul_left_transposed(dscores, Qh, dKh);
    ops_.mul_scalar(dKh, scale, dKh);
    diagnostics_->bk_attn_dKh(idx_, h, dKh);
  }

  ops_.matmul_right_transposed(dqkv, Wqkv, dx);
  diagnostics_->bk_attn_dx(idx_, dx);

  TensorView dWqkv = gradientStore_->grad_for_param(Wqkv);
  ops_.matmul_left_transposed(cache_x_, dqkv, dWqkv);
  diagnostics_->bk_attn_dWqkv(idx_, dWqkv);
  TensorView dbqkv = gradientStore_->grad_for_param(bqkv);
  ops_.row_sum(dqkv, dbqkv);
  diagnostics_->bk_attn_dbqkv(idx_, dbqkv);
  observer_->on_attention_end(idx_);
}

#undef require
