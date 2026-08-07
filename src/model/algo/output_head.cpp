#include "output_head.hpp"
#include "training_diagnostics_controller.hpp"
#include <utils/assert.hpp>

#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("OutputHead: ") + std::string(msg);                    \
  })

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
  require(diagnostics != nullptr, "diagnostics must be non-null");
  diagnostics_ = diagnostics;
}

void OutputHead::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);

  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();
  const TensorView &lm_w = tensorStore_.param_lm_head_w();

  require(lnf_g.dim(0) == 1 && lnf_g.dim(1) == model_dim,
          "lnf_gamma must be [1, D]");
  require(lnf_b.dim(0) == 1 && lnf_b.dim(1) == model_dim,
          "lnf_beta must be [1, D]");
  require(lm_w.dim(0) == model_dim && lm_w.dim(1) == vocab_size,
          "lm_head_w must be [D, V]");

  require(lnf_g.device() == lnf_b.device() && lnf_g.device() == lm_w.device(),
          "output-head parameter devices must match");
  require(lnf_g.dtype() == lnf_b.dtype() && lnf_g.dtype() == lm_w.dtype(),
          "output-head parameter dtypes must match");
}

void OutputHead::forward(const TensorView &x, TensorView &logits,
                         TensorView *last_hidden) {
  observer_->on_output_head_start();
  require(x.device() == logits.device(), "x/logits device mismatch");
  require(x.dtype() == logits.dtype(), "x/logits dtype mismatch");
  require(x.rank() == 3, "x must be semantic [B, S, D]");

  const int64_t batch_size = x.dim(0);
  const int64_t seq_len = x.dim(1);
  const int64_t model_dim = x.dim(2);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model),
          "x.dim(2) != d_model");
  require(logits.rank() == 3 && logits.dim(0) == batch_size &&
              logits.dim(1) == seq_len && logits.dim(2) == vocab_size,
          "logits must be semantic [B, S, V]");

  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();
  const TensorView &lm_w = tensorStore_.param_lm_head_w();
  require(lnf_g.device() == logits.device(), "lnf_gamma device mismatch");
  require(lm_w.device() == logits.device(), "lm_head_w device mismatch");
  require(lnf_g.dtype() == logits.dtype(), "lnf_gamma dtype mismatch");
  require(lm_w.dtype() == logits.dtype(), "lm_head_w dtype mismatch");

  TensorView Xn = tensorStore_.temp_tr_Xn(batch_size, seq_len);
  ops_.layernorm(x, lnf_g, lnf_b, Xn);

  if (last_hidden != nullptr) {
    require(last_hidden->rank() == 3 && last_hidden->dim(0) == batch_size &&
                last_hidden->dim(1) == seq_len &&
                last_hidden->dim(2) == model_dim,
            "last_hidden must be semantic [B, S, D]");
    require(last_hidden->device() == logits.device(),
            "last_hidden device mismatch");
    require(last_hidden->dtype() == logits.dtype(), "last_hidden dtype mismatch");
    ops_.copy(Xn, *last_hidden);
  }

  ops_.matmul(Xn, lm_w, logits);
  cache_x_ = x;
  cache_xn_ = Xn;
  has_cache_ = true;
  observer_->on_output_head_end();
}

void OutputHead::backward(const TensorView &dlogits, TensorView &dx) {
  observer_->on_output_head_start();
  require(gradientStore_ != nullptr, "backward requires gradient store");
  require(diagnostics_ != nullptr, "backward requires diagnostics controller");
  require(has_cache_, "backward called before forward");
  require(dlogits.rank() == 3, "dlogits must be semantic [B, S, V]");
  require(dx.rank() == cache_x_.rank() && dx.dim(0) == cache_x_.dim(0) &&
              dx.dim(1) == cache_x_.dim(1) && dx.dim(2) == cache_x_.dim(2),
          "dx shape mismatch");

  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  require(dlogits.dim(0) == batch_size && dlogits.dim(1) == seq_len &&
              dlogits.dim(2) == vocab_size,
          "dlogits shape mismatch");

  const TensorView &lm_w = tensorStore_.param_lm_head_w();
  const TensorView &lnf_g = tensorStore_.param_lnf_gamma();
  const TensorView &lnf_b = tensorStore_.param_lnf_beta();

  TensorView d_lm_w = gradientStore_->grad_for_param(lm_w);
  diagnostics_->bk_transformer_dlogits(dlogits);
  ops_.matmul_left_transposed(cache_xn_, dlogits, d_lm_w);
  diagnostics_->bk_transformer_d_lm_w(d_lm_w);
  observer_->probe_output_head_ready(lm_w, d_lm_w);

  TensorView d_xn = tensorStore_.temp_bw_d_xn(batch_size, seq_len);
  ops_.matmul_right_transposed(dlogits, lm_w, d_xn);
  diagnostics_->bk_transformer_d_xn(d_xn);

  TensorView d_lnf_g = gradientStore_->grad_for_param(lnf_g);
  TensorView d_lnf_b = gradientStore_->grad_for_param(lnf_b);
  ops_.layernorm_backward(cache_x_, lnf_g, d_xn, dx, d_lnf_g, d_lnf_b);
  diagnostics_->bk_transformer_d_xlast(dx);
  diagnostics_->bk_transformer_d_lnf_g(d_lnf_g);
  diagnostics_->bk_transformer_d_lnf_b(d_lnf_b);
  observer_->on_output_head_end();
}

#undef require
