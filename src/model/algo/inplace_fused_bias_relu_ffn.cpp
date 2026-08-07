#include "inplace_fused_bias_relu_ffn.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"

void InplaceFusedBiasReluFFN::forward(const TensorView &x, TensorView &out) {
  observer_->on_ffn_start(idx_);
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::validate_bsd_io(x, out, model_dim,
                                       "InplaceFusedBiasReluFFN");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;

  const TensorView &W1 = tensorStore_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorStore_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorStore_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorStore_.param_ffn_b2(idx_);
  TensorContracts::validate_same_device_dtype(W1, x,
                                              "InplaceFusedBiasReluFFN",
                                              "W1/x");
  TensorContracts::validate_same_device_dtype(W2, x,
                                              "InplaceFusedBiasReluFFN",
                                              "W2/x");

  TensorView h = tensorStore_.temp_ffn_h(idx_, batch_size, seq_len);

  ops_.gemm_ranked_matrix_rhs(x, W1, h);
  ops_.add_bias_relu_rowwise_inplace(h, b1);
  ops_.gemm_ranked_matrix_rhs(h, W2, out);
  ops_.add_bias_rowwise(out, b2, out);

  cache_x_ = x;
  cache_h_ = h;
  observer_->on_ffn_end(idx_);
}

void InplaceFusedBiasReluFFN::backward(const TensorView &dout, TensorView &dx) {
  observer_->on_ffn_start(idx_);
  TensorContracts::validate_same_device_dtype(dout, cache_x_,
                                              "InplaceFusedBiasReluFFN",
                                              "dout/cache_x");
  TensorContracts::validate_bsd_shape_like(dx, cache_x_,
                                           "InplaceFusedBiasReluFFN", "dx");
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);

  const TensorView &W1 = tensorStore_.param_ffn_w1(idx_);
  const TensorView &b1 = tensorStore_.param_ffn_b1(idx_);
  const TensorView &W2 = tensorStore_.param_ffn_w2(idx_);
  const TensorView &b2 = tensorStore_.param_ffn_b2(idx_);
  TensorView dW2 = gradientStore_->grad_for_param(W2);
  ops_.gemm_ranked_reduce_lhs_t(cache_h_, dout, dW2);
  diagnostics_->bk_ffn_dW2(idx_, dW2);
  TensorView db2 = gradientStore_->grad_for_param(b2);
  ops_.row_sum(dout, db2);
  diagnostics_->bk_ffn_db2(idx_, db2);

  TensorView da = tensorStore_.temp_ffn_da(idx_, batch_size, seq_len);
  ops_.gemm_ranked_matrix_rhs_t(dout, W2, da);
  diagnostics_->bk_ffn_da(idx_, da);

  ops_.relu_backward_inplace(cache_h_, da);
  TensorView dh = da;
  diagnostics_->bk_ffn_dh(idx_, dh);

  TensorView dW1 = gradientStore_->grad_for_param(W1);
  ops_.gemm_ranked_reduce_lhs_t(cache_x_, dh, dW1);
  diagnostics_->bk_ffn_dW1(idx_, dW1);
  TensorView db1 = gradientStore_->grad_for_param(b1);
  ops_.row_sum(dh, db1);
  diagnostics_->bk_ffn_db1(idx_, db1);

  ops_.gemm_ranked_matrix_rhs_t(dh, W1, dx);
  diagnostics_->bk_ffn_dx(idx_, dx);
  observer_->on_ffn_end(idx_);
}
