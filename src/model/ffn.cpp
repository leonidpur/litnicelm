#include "ffn.hpp"

#include <utils/assert.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("FFN: ") + std::string(msg); })

FFN::FFN(int layer_index, const Config &cfg, TensorFactory &tensor_factory, Ops &ops)
    : idx_(layer_index), cfg_(cfg), tensorFactory_(tensor_factory), ops_(ops) {
  validate_contract();
}

void FFN::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void FFN::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);

  const TensorView &W1 = tensorFactory_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorFactory_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorFactory_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorFactory_.param_ffn_b2(idx_);

  require(W1.shape().r == model_dim && W1.shape().c == ffn_dim,
          "W1 must be [D, F]");
  require(b1.shape().r == 1 && b1.shape().c == ffn_dim, "b1 must be [1, F]");
  require(W2.shape().r == ffn_dim && W2.shape().c == model_dim,
          "W2 must be [F, D]");
  require(b2.shape().r == 1 && b2.shape().c == model_dim,
          "b2 must be [1, D]");

  require(W1.device() == W2.device() && W1.device() == b1.device() &&
              W1.device() == b2.device(),
          "FFN parameter devices must match");
  require(W1.dtype() == W2.dtype() && W1.dtype() == b1.dtype() &&
              W1.dtype() == b2.dtype(),
          "FFN parameter dtypes must match");
}

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
  observer_->on_ffn_start(idx_);
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  const int64_t token_rows = x.shape().r;
  const int64_t model_dim = x.shape().c;
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == token_rows && out.shape().c == model_dim,
          "out must be [T, D]");

  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);

  const TensorView &W1 = tensorFactory_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorFactory_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorFactory_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorFactory_.param_ffn_b2(idx_);
  require(W1.device() == x.device() && W2.device() == x.device(),
          "param device mismatch");
  require(W1.dtype() == x.dtype() && W2.dtype() == x.dtype(),
          "param dtype mismatch");

  TensorView h = tensorFactory_.temp_ffn_h(idx_, token_rows);
  TensorView a = tensorFactory_.temp_ffn_a(idx_, token_rows);

  ops_.matmul(x, W1, h);
  ops_.add_bias_rowwise(h, b1, h);
  ops_.relu(h, a);
  ops_.matmul(a, W2, out);
  ops_.add_bias_rowwise(out, b2, out);

  cache_x_ = x;
  cache_h_ = h;
  cache_a_ = a;
  has_cache_ = true;
  observer_->on_ffn_end(idx_);
}

void FFN::backward(const TensorView &dout, TensorView &dx,
                   const ParamUpdater &update_param) {
  observer_->on_ffn_start(idx_);
  require(has_cache_, "backward called before forward");
  require(dout.device() == cache_x_.device(), "dout device mismatch");
  require(dout.dtype() == cache_x_.dtype(), "dout dtype mismatch");
  require(dx.shape().r == cache_x_.shape().r && dx.shape().c == cache_x_.shape().c,
          "dx shape mismatch");

  const int64_t token_rows = cache_x_.shape().r;
  const int64_t model_dim = cache_x_.shape().c;
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);

  const TensorView &W1 = tensorFactory_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorFactory_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorFactory_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorFactory_.param_ffn_b2(idx_);
  TensorView dW2 = tensorFactory_.temp_ffn_dW2(idx_);
  ops_.matmul_left_transposed(cache_a_, dout, dW2);
  TensorView db2 = tensorFactory_.temp_ffn_db2(idx_);
  row_sum_f32(dout, db2);

  TensorView da = tensorFactory_.temp_ffn_da(idx_, token_rows);
  ops_.matmul_right_transposed(dout, W2, da);

  TensorView dh = tensorFactory_.temp_ffn_dh(idx_, token_rows);
  relu_backward_f32(cache_h_, da, dh);

  TensorView dW1 = tensorFactory_.temp_ffn_dW1(idx_);
  ops_.matmul_left_transposed(cache_x_, dh, dW1);
  TensorView db1 = tensorFactory_.temp_ffn_db1(idx_);
  row_sum_f32(dh, db1);

  ops_.matmul_right_transposed(dh, W1, dx);

  const std::string layer_prefix = "layer" + std::to_string(idx_) + ".";
  update_param(layer_prefix + "ffn_w1", const_cast<TensorView &>(W1), dW1, true);
  update_param(layer_prefix + "ffn_b1", const_cast<TensorView &>(b1), db1, false);
  update_param(layer_prefix + "ffn_w2", const_cast<TensorView &>(W2), dW2, true);
  update_param(layer_prefix + "ffn_b2", const_cast<TensorView &>(b2), db2, false);
  observer_->on_ffn_end(idx_);
}

#undef require
