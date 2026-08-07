#include "self_attention.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"

#include <cmath>

SelfAttention::SelfAttention(int layer_index, const Config &cfg,
                             TensorStore &tensor_store,
                             GradientStore *gradient_store, Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorStore_(tensor_store),
      gradientStore_(gradient_store),
      ops_(ops) {
  validate_contract();
}

void SelfAttention::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void SelfAttention::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void SelfAttention::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t num_heads = static_cast<int64_t>(cfg_.model.n_heads);
  TensorContracts::validate_attention_config(model_dim, num_heads,
                                             "SelfAttention");

  const TensorView &Wqkv = tensorStore_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorStore_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorStore_.param_attn_out_w(idx_);
  const TensorView &bo = tensorStore_.param_attn_out_b(idx_);

  TensorContracts::validate_attention_params(Wqkv, bqkv, Wo, bo, model_dim,
                                             "SelfAttention");
}

void SelfAttention::forward(const TensorView &x, TensorView &out) {
  observer_->on_attention_start(idx_);
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::validate_bsd_io(x, out, model_dim, "SelfAttention");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;

  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &Wqkv = tensorStore_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorStore_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorStore_.param_attn_out_w(idx_);
  const TensorView &bo = tensorStore_.param_attn_out_b(idx_);

  TensorContracts::validate_same_device_dtype(Wqkv, x, "SelfAttention",
                                              "Wqkv/x");
  TensorContracts::validate_same_device_dtype(Wo, x, "SelfAttention", "Wo/x");

  TensorView qkv = tensorStore_.temp_attn_qkv(idx_, batch_size, seq_len);
  ops_.matmul(x, Wqkv, qkv);
  ops_.add_bias_rowwise(qkv, bqkv, qkv);

  TensorView Q = qkv.subcols(0, model_dim);
  TensorView K = qkv.subcols(model_dim, model_dim);
  TensorView V = qkv.subcols(2 * model_dim, model_dim);

  TensorView context = tensorStore_.temp_attn_context(idx_, batch_size, seq_len);
  ops_.fill(context, 0.0f);

  TensorView scores = tensorStore_.temp_attn_scores(idx_, batch_size, seq_len);
  TensorView weights = tensorStore_.temp_attn_weights(idx_, batch_size, seq_len);
  TensorView cached_weights =
      tensorStore_.temp_attn_cached_weights(idx_, batch_size, seq_len);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;

    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView context_h = context.subcols(col0, dh);
    TensorView cached_weights_h = cached_weights.select(0, h);

    ops_.matmul_right_transposed(Qh, Kh, scores);
    ops_.mul_scalar(scores, scale, scores);
    ops_.apply_causal_mask_inplace(scores);
    ops_.softmax_rows(scores, weights);
    ops_.copy(weights, cached_weights_h);
    ops_.matmul(weights, Vh, context_h);
  }

  ops_.matmul(context, Wo, out);
  ops_.add_bias_rowwise(out, bo, out);

  cache_x_ = x;
  cache_qkv_ = qkv;
  cache_context_ = context;
  observer_->on_attention_end(idx_);
}

void SelfAttention::backward(const TensorView &dout, TensorView &dx) {
  observer_->on_attention_start(idx_);
  TensorContracts::validate_bsd_shape_like(dout, cache_x_, "SelfAttention",
                                           "dout");
  TensorContracts::validate_bsd_shape_like(dx, cache_x_, "SelfAttention",
                                           "dx");

  const int64_t model_dim = cache_x_.dim(2);
  TensorContracts::validate_bsd_tensor(cache_x_, model_dim, "SelfAttention",
                                       "cached x");
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

    ops_.softmax_backward_rows(weights_h, dweights, dscores);
    diagnostics_->bk_attn_dscores_softmax_backward(idx_, h, dscores);
    ops_.apply_causal_mask_inplace(dscores, 0.0f);
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
