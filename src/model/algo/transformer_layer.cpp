#include "transformer_layer.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"

#include <cmath>

TransformerLayer::TransformerLayer(int layer_index, const Config &cfg,
                                   TensorStore &tensor_store,
                                   GradientStore *gradient_store, Ops &ops,
                                   const ModelAlgoFactory &algo_factory)
    : idx_(layer_index),
      cfg_(cfg),
      tensorStore_(tensor_store),
      gradientStore_(gradient_store),
      ops_(ops) {
  attn_ =
      algo_factory.create_attention(layer_index, cfg, tensor_store, gradient_store, ops);
  ffn_ = algo_factory.create_ffn(layer_index, cfg, tensor_store, gradient_store, ops);
  validate_contract();
}

void TransformerLayer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  attn_->set_observer(observer);
  ffn_->set_observer(observer);
}

void TransformerLayer::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
  attn_->set_diagnostics(diagnostics);
  ffn_->set_diagnostics(diagnostics);
}

void TransformerLayer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorView &ln1_gamma = tensorStore_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorStore_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorStore_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorStore_.param_ln2_beta(idx_);

  TensorContracts::validate_layernorm_params(ln1_gamma, ln1_beta, model_dim,
                                             "TransformerLayer", "ln1");
  TensorContracts::validate_layernorm_params(ln2_gamma, ln2_beta, model_dim,
                                             "TransformerLayer", "ln2");
  TensorContracts::validate_same_device_dtype(ln1_gamma, ln2_gamma,
                                              "TransformerLayer", "ln1/ln2");
}

void TransformerLayer::forward(const TensorView &x, TensorView &out) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::validate_bsd_io(x, out, model_dim,
                                       "TransformerLayer");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;

  const TensorView &ln1_gamma = tensorStore_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorStore_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorStore_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorStore_.param_ln2_beta(idx_);

  TensorContracts::validate_same_device_dtype(ln1_gamma, x,
                                              "TransformerLayer",
                                              "ln1_gamma/x");
  TensorContracts::validate_same_device_dtype(ln2_gamma, x,
                                              "TransformerLayer",
                                              "ln2_gamma/x");

  TensorView ln1 = tensorStore_.temp_layer_ln1(idx_, batch_size, seq_len);
  TensorView attn_out =
      tensorStore_.temp_layer_attn_out(idx_, batch_size, seq_len);
  TensorView y = tensorStore_.temp_layer_resid1(idx_, batch_size, seq_len);

  TensorView ln2 = tensorStore_.temp_layer_ln2(idx_, batch_size, seq_len);
  TensorView ffn_out =
      tensorStore_.temp_layer_ffn_out(idx_, batch_size, seq_len);

  ops_.layernorm(x, ln1_gamma, ln1_beta, ln1);
  attn_->forward(ln1, attn_out);
  ops_.add(x, attn_out, y);

  ops_.layernorm(y, ln2_gamma, ln2_beta, ln2);
  ffn_->forward(ln2, ffn_out);

  ops_.add(y, ffn_out, out);

  cache_x_ = x;
  cache_y_ = y;
  cache_ln1_ = ln1;
  cache_ln2_ = ln2;
}

void TransformerLayer::backward(const TensorView &dout, TensorView &dx) {
  TensorContracts::validate_bsd_shape_like(dout, cache_x_,
                                           "TransformerLayer", "dout");
  TensorContracts::validate_bsd_shape_like(dx, cache_x_, "TransformerLayer",
                                           "dx");

  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t model_dim = cache_x_.dim(2);
  (void)model_dim;

  const TensorView &ln1_gamma = tensorStore_.param_ln1_gamma(idx_);
  const TensorView &ln1_beta = tensorStore_.param_ln1_beta(idx_);
  const TensorView &ln2_gamma = tensorStore_.param_ln2_gamma(idx_);
  const TensorView &ln2_beta = tensorStore_.param_ln2_beta(idx_);
  TensorView dln2 = tensorStore_.temp_layer_dln2(idx_, batch_size, seq_len);
  ffn_->backward(dout, dln2);
  diagnostics_->bk_layer_dln2_after_ffn(idx_, dln2);

  TensorView dy_ln2 =
      tensorStore_.temp_layer_dy_ln2(idx_, batch_size, seq_len);
  TensorView dln2_gamma = gradientStore_->grad_for_param(ln2_gamma);
  TensorView dln2_beta = gradientStore_->grad_for_param(ln2_beta);
  ops_.layernorm_backward(cache_y_, ln2_gamma, dln2, dy_ln2, dln2_gamma,
                          dln2_beta);
  diagnostics_->bk_layer_dy_ln2(idx_, dy_ln2);
  diagnostics_->bk_layer_dln2_gamma(idx_, dln2_gamma);
  diagnostics_->bk_layer_dln2_beta(idx_, dln2_beta);

  TensorView dy_total =
      tensorStore_.temp_layer_dy_total(idx_, batch_size, seq_len);
  ops_.add(dout, dy_ln2, dy_total);
  diagnostics_->bk_layer_dy_total(idx_, dy_total);

  TensorView dln1 = tensorStore_.temp_layer_dln1(idx_, batch_size, seq_len);
  attn_->backward(dy_total, dln1);
  diagnostics_->bk_layer_dln1_after_attn(idx_, dln1);

  TensorView dx_ln1 =
      tensorStore_.temp_layer_dx_ln1(idx_, batch_size, seq_len);
  TensorView dln1_gamma = gradientStore_->grad_for_param(ln1_gamma);
  TensorView dln1_beta = gradientStore_->grad_for_param(ln1_beta);
  ops_.layernorm_backward(cache_x_, ln1_gamma, dln1, dx_ln1, dln1_gamma,
                          dln1_beta);
  diagnostics_->bk_layer_dx_ln1(idx_, dx_ln1);
  diagnostics_->bk_layer_dln1_gamma(idx_, dln1_gamma);
  diagnostics_->bk_layer_dln1_beta(idx_, dln1_beta);

  ops_.add(dy_total, dx_ln1, dx);
  diagnostics_->bk_layer_dx(idx_, dx);
}
