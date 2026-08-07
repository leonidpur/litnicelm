#include "fused_bias_relu_ffn.hpp"
#include "training_diagnostics_controller.hpp"

#include <utils/assert.hpp>

#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("FusedBiasReluFFN: ") +             \
                              std::string(msg); })

void FusedBiasReluFFN::forward(const TensorView &x, TensorView &out) {
  observer_->on_ffn_start(idx_);
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

  const TensorView &W1 = tensorStore_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorStore_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorStore_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorStore_.param_ffn_b2(idx_);
  require(W1.device() == x.device() && W2.device() == x.device(),
          "param device mismatch");
  require(W1.dtype() == x.dtype() && W2.dtype() == x.dtype(),
          "param dtype mismatch");

  TensorView h = tensorStore_.temp_ffn_h(idx_, batch_size, seq_len);
  TensorView a = tensorStore_.temp_ffn_a(idx_, batch_size, seq_len);

  ops_.matmul(x, W1, h);
  ops_.add_bias_relu_rowwise(h, b1, a);
  ops_.matmul(a, W2, out);
  ops_.add_bias_rowwise(out, b2, out);

  cache_x_ = x;
  cache_h_ = a;
  cache_a_ = a;
  has_cache_ = true;
  observer_->on_ffn_end(idx_);
}

void FusedBiasReluFFN::backward(const TensorView &dout, TensorView &dx) {
  observer_->on_ffn_start(idx_);
  require(gradientStore_ != nullptr, "backward requires gradient store");
  require(diagnostics_ != nullptr, "backward requires diagnostics controller");
  require(has_cache_, "backward called before forward");
  require(dout.device() == cache_x_.device(), "dout device mismatch");
  require(dout.dtype() == cache_x_.dtype(), "dout dtype mismatch");
  require(dx.rank() == cache_x_.rank() && dx.dim(0) == cache_x_.dim(0) &&
              dx.dim(1) == cache_x_.dim(1) && dx.dim(2) == cache_x_.dim(2),
          "dx shape mismatch");
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);

  const TensorView &W1 = tensorStore_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorStore_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorStore_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorStore_.param_ffn_b2(idx_);
  TensorView dW2 = gradientStore_->grad_for_param(W2);
  ops_.matmul_left_transposed(cache_a_, dout, dW2);
  diagnostics_->bk_ffn_dW2(idx_, dW2);
  TensorView db2 = gradientStore_->grad_for_param(b2);
  ops_.row_sum(dout, db2);
  diagnostics_->bk_ffn_db2(idx_, db2);

  TensorView da = tensorStore_.temp_ffn_da(idx_, batch_size, seq_len);
  ops_.matmul_right_transposed(dout, W2, da);
  diagnostics_->bk_ffn_da(idx_, da);

  TensorView dh = tensorStore_.temp_ffn_dh(idx_, batch_size, seq_len);
  ops_.relu_backward(cache_a_, da, dh);
  diagnostics_->bk_ffn_dh(idx_, dh);

  TensorView dW1 = gradientStore_->grad_for_param(W1);
  ops_.matmul_left_transposed(cache_x_, dh, dW1);
  diagnostics_->bk_ffn_dW1(idx_, dW1);
  TensorView db1 = gradientStore_->grad_for_param(b1);
  ops_.row_sum(dh, db1);
  diagnostics_->bk_ffn_db1(idx_, db1);

  ops_.matmul_right_transposed(dh, W1, dx);
  diagnostics_->bk_ffn_dx(idx_, dx);
  observer_->on_ffn_end(idx_);
}

#undef require
