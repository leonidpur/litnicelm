#include "output_head.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"


OutputHead::OutputHead(const Config &cfg, TensorStore &tensor_store,
                       GradientStore *gradient_store, Ops &ops)
    : cfg_(cfg),
      tensorStore_(tensor_store),
      gradientStore_(gradient_store),
      ops_(ops) {
  validate_contract();
}

void OutputHead::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void OutputHead::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void OutputHead::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);

  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();
  const TensorView &lm_w = tensorStore_.param_lm_head_w();

  TensorContracts::validate_output_head_params(lnf_g, lnf_b, lm_w, model_dim,
                                               vocab_size, "OutputHead");
}

void OutputHead::forward(const TensorView &x, TensorView &logits,
                         TensorView *last_hidden) {
  observer_->on_output_head_start();
  TensorContracts::validate_same_device_dtype(x, logits, "OutputHead",
                                              "x/logits");
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  TensorContracts::validate_bsd_tensor(x, model_dim, "OutputHead", "x");
  const int64_t batch_size = x.dim(0);
  const int64_t seq_len = x.dim(1);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  TensorContracts::validate_logits_bsv(logits, batch_size, seq_len, vocab_size,
                                       "OutputHead", "logits");

  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();
  const TensorView &lm_w = tensorStore_.param_lm_head_w();
  TensorContracts::validate_same_device_dtype(lnf_g, logits, "OutputHead",
                                              "lnf_gamma/logits");
  TensorContracts::validate_same_device_dtype(lm_w, logits, "OutputHead",
                                              "lm_head_w/logits");

  TensorView Xn = tensorStore_.temp_tr_Xn(batch_size, seq_len);
  ops_.layernorm(x, lnf_g, lnf_b, Xn);

  if (last_hidden != nullptr) {
    TensorContracts::validate_bsd_shape_like(*last_hidden, x, "OutputHead",
                                             "last_hidden");
    TensorContracts::validate_same_device_dtype(*last_hidden, logits,
                                                "OutputHead",
                                                "last_hidden/logits");
    ops_.copy(Xn, *last_hidden);
  }

  ops_.gemm_ranked_matrix_rhs(Xn, lm_w, logits);
  cache_x_ = x;
  cache_xn_ = Xn;
  observer_->on_output_head_end();
}

void OutputHead::backward(const TensorView &dlogits, TensorView &dx) {
  observer_->on_output_head_start();
  TensorContracts::validate_bsd_shape_like(dx, cache_x_, "OutputHead", "dx");

  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  TensorContracts::validate_logits_bsv(dlogits, batch_size, seq_len,
                                       vocab_size, "OutputHead", "dlogits");

  const TensorView &lm_w = tensorStore_.param_lm_head_w();
  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();

  TensorView d_lm_w = gradientStore_->grad_for_param(lm_w);
  diagnostics_->bk_transformer_dlogits(dlogits);
  ops_.gemm_ranked_reduce_lhs_t(cache_xn_, dlogits, d_lm_w);
  diagnostics_->bk_transformer_d_lm_w(d_lm_w);
  observer_->probe_output_head_ready(lm_w, d_lm_w);

  TensorView d_xn = tensorStore_.temp_bw_d_xn(batch_size, seq_len);
  ops_.gemm_ranked_matrix_rhs_t(dlogits, lm_w, d_xn);
  diagnostics_->bk_transformer_d_xn(d_xn);

  TensorView d_lnf_g = gradientStore_->grad_for_param(lnf_g);
  TensorView d_lnf_b = gradientStore_->grad_for_param(lnf_b);
  ops_.layernorm_backward(cache_x_, lnf_g, d_xn, dx, d_lnf_g, d_lnf_b);
  diagnostics_->bk_transformer_d_xlast(dx);
  diagnostics_->bk_transformer_d_lnf_g(d_lnf_g);
  diagnostics_->bk_transformer_d_lnf_b(d_lnf_b);
  observer_->on_output_head_end();
}
