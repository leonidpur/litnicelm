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
                                   TensorFactory &tensors, Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorFactory_(tensors),
      ops_(ops),
      attn_(layer_index, cfg, tensors, ops),
      ffn_(layer_index, cfg, tensors, ops) {}

static void layernorm_backward_f32(const TensorView &x, const TensorView &gamma_1xC,
                                   const TensorView &dout, TensorView &dx,
                                   TensorView &dgamma_1xC,
                                   TensorView &dbeta_1xC) {
  require(x.device() == Device::CPU && gamma_1xC.device() == Device::CPU &&
              dout.device() == Device::CPU && dx.device() == Device::CPU &&
              dgamma_1xC.device() == Device::CPU &&
              dbeta_1xC.device() == Device::CPU,
          "layernorm_backward CPU only");
  require(x.dtype() == DType::F32 && gamma_1xC.dtype() == DType::F32 &&
              dout.dtype() == DType::F32 && dx.dtype() == DType::F32 &&
              dgamma_1xC.dtype() == DType::F32 && dbeta_1xC.dtype() == DType::F32,
          "layernorm_backward F32 only");
  require(gamma_1xC.shape().r == 1 && gamma_1xC.shape().c == x.shape().c,
          "gamma shape mismatch");
  require(dout.shape().r == x.shape().r && dout.shape().c == x.shape().c,
          "dout shape mismatch");
  require(dx.shape().r == x.shape().r && dx.shape().c == x.shape().c, "dx shape mismatch");
  require(dgamma_1xC.shape().r == 1 && dgamma_1xC.shape().c == x.shape().c,
          "dgamma shape mismatch");
  require(dbeta_1xC.shape().r == 1 && dbeta_1xC.shape().c == x.shape().c,
          "dbeta shape mismatch");

  const int64_t T = x.shape().r;
  const int64_t C = x.shape().c;
  const float eps = 1e-5f;

  for (int64_t c = 0; c < C; ++c) {
    dgamma_1xC.set_f32(0, c, 0.0f);
    dbeta_1xC.set_f32(0, c, 0.0f);
  }

  for (int64_t r = 0; r < T; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      mean += x.at_f32(r, c);
    }
    mean /= static_cast<double>(C);

    double var = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double d = static_cast<double>(x.at_f32(r, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(C);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    double sum_dxhat = 0.0;
    double sum_dxhat_xhat = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double xhat = (static_cast<double>(x.at_f32(r, c)) - mean) * inv_std;
      const double g = static_cast<double>(gamma_1xC.at_f32(0, c));
      const double dyi = static_cast<double>(dout.at_f32(r, c));
      const double dxhat = dyi * g;
      sum_dxhat += dxhat;
      sum_dxhat_xhat += dxhat * xhat;

      dgamma_1xC.set_f32(0, c, dgamma_1xC.at_f32(0, c) + static_cast<float>(dyi * xhat));
      dbeta_1xC.set_f32(0, c, dbeta_1xC.at_f32(0, c) + static_cast<float>(dyi));
    }

    for (int64_t c = 0; c < C; ++c) {
      const double xhat = (static_cast<double>(x.at_f32(r, c)) - mean) * inv_std;
      const double g = static_cast<double>(gamma_1xC.at_f32(0, c));
      const double dyi = static_cast<double>(dout.at_f32(r, c));
      const double dxhat = dyi * g;
      const double n = static_cast<double>(C);
      const double dxi =
          (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
      dx.set_f32(r, c, static_cast<float>(dxi));
    }
  }
}

void TransformerLayer::forward(const TensorView &x, TensorView &out) {
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  const int64_t T = x.shape().r;
  const int64_t D = x.shape().c;
  require(D == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == T && out.shape().c == D, "out must be [T, D]");

  const TensorView ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView ln2_beta = tensorFactory_.param_ln2_beta(idx_);
  const std::string p = "layer" + std::to_string(idx_) + ".";

  require(ln1_gamma.shape().r == 1 && ln1_gamma.shape().c == D,
          "ln1_gamma must be [1, D]");
  require(ln1_beta.shape().r == 1 && ln1_beta.shape().c == D,
          "ln1_beta must be [1, D]");
  require(ln2_gamma.shape().r == 1 && ln2_gamma.shape().c == D,
          "ln2_gamma must be [1, D]");
  require(ln2_beta.shape().r == 1 && ln2_beta.shape().c == D,
          "ln2_beta must be [1, D]");

  require(ln1_gamma.device() == x.device() && ln2_gamma.device() == x.device(),
          "layernorm param device mismatch");
  require(ln1_gamma.dtype() == x.dtype() && ln2_gamma.dtype() == x.dtype(),
          "layernorm param dtype mismatch");

  TensorView ln1 = tensorFactory_.temp_layer_ln1(idx_, T);
  TensorView attn_out = tensorFactory_.temp_layer_attn_out(idx_, T);
  TensorView y = tensorFactory_.temp_layer_resid1(idx_, T);

  TensorView ln2 = tensorFactory_.temp_layer_ln2(idx_, T);
  TensorView ffn_out = tensorFactory_.temp_layer_ffn_out(idx_, T);

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

  const int64_t T = cache_x_.shape().r;
  const int64_t D = cache_x_.shape().c;

  const TensorView ln1_gamma = tensorFactory_.param_ln1_gamma(idx_);
  const TensorView ln1_beta = tensorFactory_.param_ln1_beta(idx_);
  const TensorView ln2_gamma = tensorFactory_.param_ln2_gamma(idx_);
  const TensorView ln2_beta = tensorFactory_.param_ln2_beta(idx_);
  const std::string p = "layer" + std::to_string(idx_) + ".";

  TensorView dln2 = tensorFactory_.temp_layer_dln2(idx_, T);
  ffn_.backward(dout, dln2, update_param);

  TensorView dy_ln2 = tensorFactory_.temp_layer_dy_ln2(idx_, T);
  TensorView dln2_gamma = tensorFactory_.temp_layer_dln2_gamma(idx_);
  TensorView dln2_beta = tensorFactory_.temp_layer_dln2_beta(idx_);
  layernorm_backward_f32(cache_y_, ln2_gamma, dln2, dy_ln2, dln2_gamma, dln2_beta);

  TensorView dy_total = tensorFactory_.temp_layer_dy_total(idx_, T);
  ops_.add(dout, dy_ln2, dy_total);

  TensorView dln1 = tensorFactory_.temp_layer_dln1(idx_, T);
  attn_.backward(dy_total, dln1, update_param);

  TensorView dx_ln1 = tensorFactory_.temp_layer_dx_ln1(idx_, T);
  TensorView dln1_gamma = tensorFactory_.temp_layer_dln1_gamma(idx_);
  TensorView dln1_beta = tensorFactory_.temp_layer_dln1_beta(idx_);
  layernorm_backward_f32(cache_x_, ln1_gamma, dln1, dx_ln1, dln1_gamma, dln1_beta);

  ops_.add(dy_total, dx_ln1, dx);

  update_param(p + "ln1_gamma", const_cast<TensorView &>(ln1_gamma), dln1_gamma, false);
  update_param(p + "ln1_beta", const_cast<TensorView &>(ln1_beta), dln1_beta, false);
  update_param(p + "ln2_gamma", const_cast<TensorView &>(ln2_gamma), dln2_gamma, false);
  update_param(p + "ln2_beta", const_cast<TensorView &>(ln2_beta), dln2_beta, false);
}

#undef require
