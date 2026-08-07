#pragma once

#include "backend/device_backend.hpp"
#include "dataset.hpp"
#include "gradient_factory.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include <config.hpp>
#include <types.hpp>

#include <functional>
#include <stdexcept>
#include <string>

class Transformer;

class TrainingDiagnosticsController {
public:
  TrainingDiagnosticsController(TensorFactory &tensor_factory, Ops &ops,
                                Transformer &model,
                                GradientFactory &gradient_factory,
                                DeviceBackend &device_backend,
                                const RuntimeFlags &runtime_flags,
                                const Config &cfg);

  static void check_finite(Ops &ops, const TensorView &tensor,
                           const std::string &label);

  void bk_transformer_dlogits(const TensorView &dlogits) const;
  void bk_transformer_d_lm_w(const TensorView &d_lm_w) const;
  void bk_transformer_d_xn(const TensorView &d_xn) const;
  void bk_transformer_d_xlast(const TensorView &d_xlast) const;
  void bk_transformer_d_lnf_g(const TensorView &d_lnf_g) const;
  void bk_transformer_d_lnf_b(const TensorView &d_lnf_b) const;
  void bk_transformer_layer_d_prev(int layer, const TensorView &d_prev) const;
  void bk_transformer_d_cur_before_embeddings(const TensorView &d_cur) const;
  void bk_transformer_d_tok(const TensorView &d_tok) const;
  void bk_transformer_d_pos(const TensorView &d_pos) const;

  void bk_layer_dln2_after_ffn(int layer, const TensorView &dln2) const;
  void bk_layer_dy_ln2(int layer, const TensorView &dy_ln2) const;
  void bk_layer_dln2_gamma(int layer, const TensorView &dln2_gamma) const;
  void bk_layer_dln2_beta(int layer, const TensorView &dln2_beta) const;
  void bk_layer_dy_total(int layer, const TensorView &dy_total) const;
  void bk_layer_dln1_after_attn(int layer, const TensorView &dln1) const;
  void bk_layer_dx_ln1(int layer, const TensorView &dx_ln1) const;
  void bk_layer_dln1_gamma(int layer, const TensorView &dln1_gamma) const;
  void bk_layer_dln1_beta(int layer, const TensorView &dln1_beta) const;
  void bk_layer_dx(int layer, const TensorView &dx) const;

  void bk_ffn_dW2(int layer, const TensorView &dW2) const;
  void bk_ffn_db2(int layer, const TensorView &db2) const;
  void bk_ffn_da(int layer, const TensorView &da) const;
  void bk_ffn_dh(int layer, const TensorView &dh) const;
  void bk_ffn_dW1(int layer, const TensorView &dW1) const;
  void bk_ffn_db1(int layer, const TensorView &db1) const;
  void bk_ffn_dx(int layer, const TensorView &dx) const;

  void bk_attn_dWo(int layer, const TensorView &dWo) const;
  void bk_attn_dbo(int layer, const TensorView &dbo) const;
  void bk_attn_dcontext(int layer, const TensorView &dcontext) const;
  void bk_attn_dweights(int layer, int64_t head,
                        const TensorView &dweights) const;
  void bk_attn_dVh(int layer, int64_t head, const TensorView &dVh) const;
  void bk_attn_dscores_softmax_backward(int layer, int64_t head,
                                        const TensorView &dscores) const;
  void bk_attn_dscores_masked(int layer, int64_t head,
                              const TensorView &dscores) const;
  void bk_attn_dQh(int layer, int64_t head, const TensorView &dQh) const;
  void bk_attn_dKh(int layer, int64_t head, const TensorView &dKh) const;
  void bk_attn_dx(int layer, const TensorView &dx) const;
  void bk_attn_dWqkv(int layer, const TensorView &dWqkv) const;
  void bk_attn_dbqkv(int layer, const TensorView &dbqkv) const;

  void after_forward(const TensorView &logits) const;
  void after_loss_scalar(const TensorView &loss_scalar) const;
  void after_logits_targets_backward(const TensorView &dlogits) const;
  void after_cross_entropy_backward(const TensorView &dlogits) const;

  double compute_global_grad_norm() const;

  [[noreturn]] void handle_nonfinite_grad_norm(
      const TrainBatch &batch, const TensorView &original_dlogits,
      const std::runtime_error &original_error,
      const std::function<void()> &zero_gradients) const;

private:
  TensorFactory &tensorFactory_;
  Ops &ops_;
  Transformer &model_;
  GradientFactory &gradientFactory_;
  DeviceBackend &deviceBackend_;
  const RuntimeFlags &runtimeFlags_;
  const Config &cfg_;

  void replay_forward_loss_backward(const TrainBatch &batch) const;
};
