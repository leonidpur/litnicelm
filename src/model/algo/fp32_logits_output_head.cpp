// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "fp32_logits_output_head.hpp"

#include "tensor_contracts.hpp"

#include <litnicelm_utils/checks.hpp>

#include <stdexcept>

Fp32LogitsOutputHead::Fp32LogitsOutputHead(const Config &cfg,
                                           TensorStore &tensor_store,
                                           const Ops &ops)
    : cfg_(cfg), tensorStore_(tensor_store), ops_(ops) {
  TensorContracts::check_output_head_params(
      tensorStore_.param_lnf_gamma(), tensorStore_.param_lnf_beta(),
      tensorStore_.param_lm_head_w(), static_cast<int64_t>(cfg_.model.d_model),
      static_cast<int64_t>(cfg_.model.vocab_size), "Fp32LogitsOutputHead");
  checker().check(CheckKind::Contract,
                  tensorStore_.param_lm_head_w().dtype() == DType::F32,
                  "Fp32LogitsOutputHead", "lm_head_w must be F32");
  checker().check(CheckKind::Contract,
                  tensorStore_.compute_lm_head_w().dtype() == DType::BF16,
                  "Fp32LogitsOutputHead", "needs BF16 activations");
}

void Fp32LogitsOutputHead::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void Fp32LogitsOutputHead::forward(const TensorView &x, TensorView &logits,
                                   TensorView *last_hidden) {
  observer_->on_output_head_start();
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  checker().check_bsd(x, -1, -1, model_dim, "Fp32LogitsOutputHead x");
  const int64_t batch_size = x.dim(0);
  const int64_t seq_len = x.dim(1);
  TensorContracts::check_logits_bsv(
      logits, batch_size, seq_len, static_cast<int64_t>(cfg_.model.vocab_size),
      "Fp32LogitsOutputHead", "logits");
  checker().check(CheckKind::Contract, logits.dtype() == DType::F32,
                  "Fp32LogitsOutputHead", "logits must be F32");

  TensorView Xn = tensorStore_.temp_tr_Xn(batch_size, seq_len);
  ops_.layernorm(x, tensorStore_.param_lnf_gamma(), tensorStore_.param_lnf_beta(),
                 Xn);
  if (last_hidden != nullptr) {
    checker().check_same_shape(*last_hidden, x, "Fp32LogitsOutputHead last_hidden");
    ops_.copy(Xn, *last_hidden);
  }
  TensorView Xn_f32 = tensorStore_.temp_infer_xn_f32(seq_len);
  ops_.cast(Xn, Xn_f32);
  ops_.gemm_ranked_matrix_rhs(Xn_f32, tensorStore_.param_lm_head_w(), logits);
  observer_->on_output_head_end();
}

void Fp32LogitsOutputHead::backward(const TensorView &, TensorView &) {
  throw std::runtime_error("Fp32LogitsOutputHead: inference only (no backward)");
}
