#include "transformer_layer.hpp"
#include "training_diagnostics_controller.hpp"

#include <utils/assert.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("TransformerLayer: ") + std::string(msg);               \
  })

TransformerLayer::TransformerLayer(int layer_index, const Config &cfg,
                                   TensorFactory &tensor_factory,
                                   GradientFactory *gradient_factory, Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorFactory_(tensor_factory),
      gradientFactory_(gradient_factory),
      ops_(ops),
      attn_(layer_index, cfg, tensor_factory, gradient_factory, ops),
      ffn_(layer_index, cfg, tensor_factory, gradient_factory, ops) {
  validate_contract();
}

void TransformerLayer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  attn_.set_observer(observer);
  ffn_.set_observer(observer);
}

void TransformerLayer::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  require(diagnostics != nullptr, "diagnostics must be non-null");
  diagnostics_ = diagnostics;
  attn_.set_diagnostics(diagnostics);
  ffn_.set_diagnostics(diagnostics);
}

void TransformerLayer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);

  require(ln1_gamma.dim(0) == 1 && ln1_gamma.dim(1) == model_dim,
          "ln1_gamma must be [1, D]");
  require(ln1_beta.dim(0) == 1 && ln1_beta.dim(1) == model_dim,
          "ln1_beta must be [1, D]");
  require(ln2_gamma.dim(0) == 1 && ln2_gamma.dim(1) == model_dim,
          "ln2_gamma must be [1, D]");
  require(ln2_beta.dim(0) == 1 && ln2_beta.dim(1) == model_dim,
          "ln2_beta must be [1, D]");

  require(ln1_gamma.device() == ln1_beta.device() &&
              ln1_gamma.device() == ln2_gamma.device() &&
              ln1_gamma.device() == ln2_beta.device(),
          "layernorm parameter devices must match");
  require(ln1_gamma.dtype() == ln1_beta.dtype() &&
              ln1_gamma.dtype() == ln2_gamma.dtype() &&
              ln1_gamma.dtype() == ln2_beta.dtype(),
          "layernorm parameter dtypes must match");
}

void TransformerLayer::forward(const TensorView &x, TensorView &out) {
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  require(x.rank() == 3, "x must be semantic [B, S, D]");
  const int64_t batch_size = x.dim(0);
  const int64_t seq_len = x.dim(1);
  const int64_t model_dim = x.dim(2);
  const int64_t token_rows = static_cast<int64_t>(x.numel() / static_cast<uint64_t>(model_dim));
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model), "x.dim(2) != d_model");
  require(out.rank() == 3 && out.dim(0) == batch_size && out.dim(1) == seq_len &&
              out.dim(2) == model_dim,
          "out must be semantic [B, S, D]");
  require(x.dim(0) > 0 && x.dim(1) > 0 &&
              x.dim(2) == model_dim && x.dim(0) * x.dim(1) == token_rows,
          "x must be semantic [B, S, D]");

  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);

  require(ln1_gamma.device() == x.device() && ln2_gamma.device() == x.device(),
          "layernorm param device mismatch");
  require(ln1_gamma.dtype() == x.dtype() && ln2_gamma.dtype() == x.dtype(),
          "layernorm param dtype mismatch");

  TensorView ln1 = tensorFactory_.temp_layer_ln1(idx_, batch_size, seq_len);
  TensorView attn_out =
      tensorFactory_.temp_layer_attn_out(idx_, batch_size, seq_len);
  TensorView y = tensorFactory_.temp_layer_resid1(idx_, batch_size, seq_len);

  TensorView ln2 = tensorFactory_.temp_layer_ln2(idx_, batch_size, seq_len);
  TensorView ffn_out =
      tensorFactory_.temp_layer_ffn_out(idx_, batch_size, seq_len);

  ops_.layernorm(x, ln1_gamma, ln1_beta, ln1);
  attn_.forward(ln1, attn_out);
  ops_.add(x, attn_out, y);

  ops_.layernorm(y, ln2_gamma, ln2_beta, ln2);
  ffn_.forward(ln2, ffn_out);

  ops_.add(y, ffn_out, out);

  cache_x_ = x;
  cache_y_ = y;
  cache_ln1_ = ln1;
  cache_ln2_ = ln2;
  has_cache_ = true;
}

void TransformerLayer::backward(const TensorView &dout, TensorView &dx) {
  require(gradientFactory_ != nullptr, "backward requires gradient factory");
  require(diagnostics_ != nullptr, "backward requires diagnostics controller");
  require(has_cache_, "backward called before forward");
  require(dout.rank() == cache_x_.rank() && dout.dim(0) == cache_x_.dim(0) &&
              dout.dim(1) == cache_x_.dim(1) && dout.dim(2) == cache_x_.dim(2),
          "dout shape mismatch");
  require(dx.rank() == cache_x_.rank() && dx.dim(0) == cache_x_.dim(0) &&
              dx.dim(1) == cache_x_.dim(1) && dx.dim(2) == cache_x_.dim(2),
          "dx shape mismatch");

  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t model_dim = cache_x_.dim(2);
  (void)model_dim;

  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);
  TensorView dln2 = tensorFactory_.temp_layer_dln2(idx_, batch_size, seq_len);
  ffn_.backward(dout, dln2);
  diagnostics_->bk_layer_dln2_after_ffn(idx_, dln2);

  TensorView dy_ln2 =
      tensorFactory_.temp_layer_dy_ln2(idx_, batch_size, seq_len);
  TensorView dln2_gamma = gradientFactory_->grad_for_param(ln2_gamma);
  TensorView dln2_beta = gradientFactory_->grad_for_param(ln2_beta);
  ops_.layernorm_backward(cache_y_, ln2_gamma, dln2, dy_ln2, dln2_gamma,
                          dln2_beta);
  diagnostics_->bk_layer_dy_ln2(idx_, dy_ln2);
  diagnostics_->bk_layer_dln2_gamma(idx_, dln2_gamma);
  diagnostics_->bk_layer_dln2_beta(idx_, dln2_beta);

  TensorView dy_total =
      tensorFactory_.temp_layer_dy_total(idx_, batch_size, seq_len);
  ops_.add(dout, dy_ln2, dy_total);
  diagnostics_->bk_layer_dy_total(idx_, dy_total);

  TensorView dln1 = tensorFactory_.temp_layer_dln1(idx_, batch_size, seq_len);
  attn_.backward(dy_total, dln1);
  diagnostics_->bk_layer_dln1_after_attn(idx_, dln1);

  TensorView dx_ln1 =
      tensorFactory_.temp_layer_dx_ln1(idx_, batch_size, seq_len);
  TensorView dln1_gamma = gradientFactory_->grad_for_param(ln1_gamma);
  TensorView dln1_beta = gradientFactory_->grad_for_param(ln1_beta);
  ops_.layernorm_backward(cache_x_, ln1_gamma, dln1, dx_ln1, dln1_gamma,
                          dln1_beta);
  diagnostics_->bk_layer_dx_ln1(idx_, dx_ln1);
  diagnostics_->bk_layer_dln1_gamma(idx_, dln1_gamma);
  diagnostics_->bk_layer_dln1_beta(idx_, dln1_beta);

  ops_.add(dy_total, dx_ln1, dx);
  diagnostics_->bk_layer_dx(idx_, dx);
}

#undef require
