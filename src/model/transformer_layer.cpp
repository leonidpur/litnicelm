#include "transformer_layer.hpp"

#include <utils/assert.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("TransformerLayer: ") + std::string(msg);               \
  })

TransformerLayer::TransformerLayer(int layer_index, const Config &cfg,
                                   TensorFactory &tensor_factory, Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorFactory_(tensor_factory),
      ops_(ops),
      attn_(layer_index, cfg, tensor_factory, ops),
      ffn_(layer_index, cfg, tensor_factory, ops) {
  validate_contract();
}

void TransformerLayer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  attn_.set_observer(observer);
  ffn_.set_observer(observer);
}

void TransformerLayer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);

  require(ln1_gamma.shape().r == 1 && ln1_gamma.shape().c == model_dim,
          "ln1_gamma must be [1, D]");
  require(ln1_beta.shape().r == 1 && ln1_beta.shape().c == model_dim,
          "ln1_beta must be [1, D]");
  require(ln2_gamma.shape().r == 1 && ln2_gamma.shape().c == model_dim,
          "ln2_gamma must be [1, D]");
  require(ln2_beta.shape().r == 1 && ln2_beta.shape().c == model_dim,
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

  const int64_t token_rows = x.shape().r;
  const int64_t model_dim = x.shape().c;
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == token_rows && out.shape().c == model_dim,
          "out must be [T, D]");

  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);

  require(ln1_gamma.device() == x.device() && ln2_gamma.device() == x.device(),
          "layernorm param device mismatch");
  require(ln1_gamma.dtype() == x.dtype() && ln2_gamma.dtype() == x.dtype(),
          "layernorm param dtype mismatch");

  TensorView ln1 = tensorFactory_.temp_layer_ln1(idx_, token_rows);
  TensorView attn_out = tensorFactory_.temp_layer_attn_out(idx_, token_rows);
  TensorView y = tensorFactory_.temp_layer_resid1(idx_, token_rows);

  TensorView ln2 = tensorFactory_.temp_layer_ln2(idx_, token_rows);
  TensorView ffn_out = tensorFactory_.temp_layer_ffn_out(idx_, token_rows);

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

void TransformerLayer::backward(const TensorView &dout, TensorView &dx,
                                const ParamUpdater &update_param) {
  require(has_cache_, "backward called before forward");
  require(dout.shape().r == cache_x_.shape().r && dout.shape().c == cache_x_.shape().c,
          "dout shape mismatch");
  require(dx.shape().r == cache_x_.shape().r && dx.shape().c == cache_x_.shape().c,
          "dx shape mismatch");

  const int64_t token_rows = cache_x_.shape().r;
  const int64_t model_dim = cache_x_.shape().c;
  (void)model_dim;

  const TensorView &ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorFactory_.param_ln2_beta(idx_);
  const std::string p = "layer" + std::to_string(idx_) + ".";

  TensorView dln2 = tensorFactory_.temp_layer_dln2(idx_, token_rows);
  ffn_.backward(dout, dln2, update_param);

  TensorView dy_ln2 = tensorFactory_.temp_layer_dy_ln2(idx_, token_rows);
  TensorView dln2_gamma = tensorFactory_.temp_layer_dln2_gamma(idx_);
  TensorView dln2_beta = tensorFactory_.temp_layer_dln2_beta(idx_);
  ops_.layernorm_backward(cache_y_, ln2_gamma, dln2, dy_ln2, dln2_gamma,
                          dln2_beta);

  TensorView dy_total = tensorFactory_.temp_layer_dy_total(idx_, token_rows);
  ops_.add(dout, dy_ln2, dy_total);

  TensorView dln1 = tensorFactory_.temp_layer_dln1(idx_, token_rows);
  attn_.backward(dy_total, dln1, update_param);

  TensorView dx_ln1 = tensorFactory_.temp_layer_dx_ln1(idx_, token_rows);
  TensorView dln1_gamma = tensorFactory_.temp_layer_dln1_gamma(idx_);
  TensorView dln1_beta = tensorFactory_.temp_layer_dln1_beta(idx_);
  ops_.layernorm_backward(cache_x_, ln1_gamma, dln1, dx_ln1, dln1_gamma,
                          dln1_beta);

  ops_.add(dy_total, dx_ln1, dx);

  update_param(p + "ln1_gamma", const_cast<TensorView &>(ln1_gamma), dln1_gamma, false);
  update_param(p + "ln1_beta", const_cast<TensorView &>(ln1_beta), dln1_beta, false);
  update_param(p + "ln2_gamma", const_cast<TensorView &>(ln2_gamma), dln2_gamma, false);
  update_param(p + "ln2_beta", const_cast<TensorView &>(ln2_beta), dln2_beta, false);
}

#undef require
