#include "ffn.hpp"

#include <utils/assert.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("FFN: ") + std::string(msg); })

FFN::FFN(int layer_index, const Config &cfg, TensorFactory &tensors, Ops &ops)
    : idx_(layer_index), cfg_(cfg), tensorFactory_(tensors), ops_(ops) {}

static void row_sum_f32(const TensorView &x, TensorView &out_1xC) {
  require(x.device() == Device::CPU && out_1xC.device() == Device::CPU,
          "row_sum_f32 CPU only");
  require(x.dtype() == DType::F32 && out_1xC.dtype() == DType::F32,
          "row_sum_f32 F32 only");
  require(out_1xC.shape().r == 1 && out_1xC.shape().c == x.shape().c,
          "row_sum_f32 out shape mismatch");
  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t c = 0; c < C; ++c) {
    float acc = 0.0f;
    for (int64_t r = 0; r < R; ++r) {
      acc += x.at_f32(r, c);
    }
    out_1xC.set_f32(0, c, acc);
  }
}

static void relu_backward_f32(const TensorView &preact, const TensorView &dout,
                              TensorView &dx) {
  require(preact.device() == Device::CPU && dout.device() == Device::CPU &&
              dx.device() == Device::CPU,
          "relu_backward_f32 CPU only");
  require(preact.dtype() == DType::F32 && dout.dtype() == DType::F32 &&
              dx.dtype() == DType::F32,
          "relu_backward_f32 F32 only");
  require(preact.shape().r == dout.shape().r && preact.shape().c == dout.shape().c,
          "relu_backward_f32 shape mismatch");
  require(dx.shape().r == preact.shape().r && dx.shape().c == preact.shape().c,
          "relu_backward_f32 dx shape mismatch");
  const int64_t R = preact.shape().r;
  const int64_t C = preact.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      const float g = preact.at_f32(r, c) > 0.0f ? dout.at_f32(r, c) : 0.0f;
      dx.set_f32(r, c, g);
    }
  }
}

void FFN::forward(const TensorView &x, TensorView &out) {
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  const int64_t T = x.shape().r;
  const int64_t D = x.shape().c;
  require(D == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == T && out.shape().c == D, "out must be [T, D]");

  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);

  const TensorView W1 = tensorFactory_.param_ffn_w1(idx_);
  const TensorView b1 = tensorFactory_.param_ffn_b1(idx_);
  const TensorView W2 = tensorFactory_.param_ffn_w2(idx_);
  const TensorView b2 = tensorFactory_.param_ffn_b2(idx_);
  require(W1.device() == x.device() && W2.device() == x.device(),
          "param device mismatch");
  require(W1.dtype() == x.dtype() && W2.dtype() == x.dtype(),
          "param dtype mismatch");

  require(W1.shape().r == D && W1.shape().c == F, "W1 must be [D, F]");
  require(b1.shape().r == 1 && b1.shape().c == F, "b1 must be [1, F]");
  require(W2.shape().r == F && W2.shape().c == D, "W2 must be [F, D]");
  require(b2.shape().r == 1 && b2.shape().c == D, "b2 must be [1, D]");

  TensorView h = tensorFactory_.temp_ffn_h(idx_, T);
  TensorView a = tensorFactory_.temp_ffn_a(idx_, T);

  ops_.matmul(x, W1, h);
  ops_.add_bias_rowwise(h, b1, h);
  ops_.relu(h, a);
  ops_.matmul(a, W2, out);
  ops_.add_bias_rowwise(out, b2, out);

  cache_x_ = x;
  cache_h_ = h;
  cache_a_ = a;
  has_cache_ = true;
}

void FFN::backward(const TensorView &dout, TensorView &dx,
                   const ParamUpdater &update_param) {
  require(has_cache_, "backward called before forward");
  require(dout.device() == cache_x_.device(), "dout device mismatch");
  require(dout.dtype() == cache_x_.dtype(), "dout dtype mismatch");
  require(dx.shape().r == cache_x_.shape().r && dx.shape().c == cache_x_.shape().c,
          "dx shape mismatch");

  const int64_t T = cache_x_.shape().r;
  const int64_t D = cache_x_.shape().c;
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);

  const TensorView W1 = tensorFactory_.param_ffn_w1(idx_);
  const TensorView b1 = tensorFactory_.param_ffn_b1(idx_);
  const TensorView W2 = tensorFactory_.param_ffn_w2(idx_);
  const TensorView b2 = tensorFactory_.param_ffn_b2(idx_);
  TensorView aT = tensorFactory_.temp_ffn_aT(idx_, T);
  ops_.transpose(cache_a_, aT);
  TensorView dW2 = tensorFactory_.temp_ffn_dW2(idx_);
  ops_.matmul(aT, dout, dW2);
  TensorView db2 = tensorFactory_.temp_ffn_db2(idx_);
  row_sum_f32(dout, db2);

  TensorView W2T = tensorFactory_.temp_ffn_W2T(idx_);
  ops_.transpose(W2, W2T);
  TensorView da = tensorFactory_.temp_ffn_da(idx_, T);
  ops_.matmul(dout, W2T, da);

  TensorView dh = tensorFactory_.temp_ffn_dh(idx_, T);
  relu_backward_f32(cache_h_, da, dh);

  TensorView xT = tensorFactory_.temp_ffn_xT(idx_, T);
  ops_.transpose(cache_x_, xT);
  TensorView dW1 = tensorFactory_.temp_ffn_dW1(idx_);
  ops_.matmul(xT, dh, dW1);
  TensorView db1 = tensorFactory_.temp_ffn_db1(idx_);
  row_sum_f32(dh, db1);

  TensorView W1T = tensorFactory_.temp_ffn_W1T(idx_);
  ops_.transpose(W1, W1T);
  ops_.matmul(dh, W1T, dx);

  const std::string layer_prefix = "layer" + std::to_string(idx_) + ".";
  update_param(layer_prefix + "ffn_w1", const_cast<TensorView &>(W1), dW1, true);
  update_param(layer_prefix + "ffn_b1", const_cast<TensorView &>(b1), db1, false);
  update_param(layer_prefix + "ffn_w2", const_cast<TensorView &>(W2), dW2, true);
  update_param(layer_prefix + "ffn_b2", const_cast<TensorView &>(b2), db2, false);
}

#undef require
