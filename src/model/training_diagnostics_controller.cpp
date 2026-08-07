#include "training_diagnostics_controller.hpp"

#include "host_tensor_stage.hpp"
#include "algo/transformer.hpp"

#include <cmath>
#include <sstream>

TrainingDiagnosticsController::TrainingDiagnosticsController(
    TensorStore &tensor_store, Ops &ops, Transformer &model,
    GradientStore &gradient_store, DeviceBackend &device_backend,
    const RuntimeFlags &runtime_flags, const Config &cfg)
    : tensorStore_(tensor_store),
      ops_(ops),
      model_(model),
      gradientStore_(gradient_store),
      deviceBackend_(device_backend),
      runtimeFlags_(runtime_flags),
      cfg_(cfg) {}

void TrainingDiagnosticsController::check_finite(
    Ops &ops, const TensorView &tensor, const std::string &label) {
  if (tensor.dtype() != DType::F32 || tensor.numel() == 0) {
    return;
  }

  Tensor staged =
      host_tensor_stage::stage_to_cpu(ops.backend(), tensor, label.c_str());
  const TensorView staged_view = staged.view();
  const float *values = reinterpret_cast<const float *>(staged_view.data());
  const uint64_t count = staged_view.numel();
  for (uint64_t i = 0; i < count; ++i) {
    const float value = values[i];
    if (!std::isfinite(value)) {
      std::ostringstream oss;
      oss << "Non-finite tensor value after " << label << " at flat_index="
          << i << " value=" << value;
      throw std::runtime_error(oss.str());
    }
  }
}

void TrainingDiagnosticsController::bk_transformer_dlogits(
    const TensorView &dlogits) const {
  if (!cfg_.training.diagnostics.bk_transformer_dlogits) {
    return;
  }
  check_finite(ops_, dlogits, "bk.transformer.dlogits");
}

void TrainingDiagnosticsController::bk_transformer_d_lm_w(
    const TensorView &d_lm_w) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_lm_w) {
    return;
  }
  check_finite(ops_, d_lm_w, "bk.transformer.d_lm_w");
}

void TrainingDiagnosticsController::bk_transformer_d_xn(
    const TensorView &d_xn) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_xn) {
    return;
  }
  check_finite(ops_, d_xn, "bk.transformer.d_xn");
}

void TrainingDiagnosticsController::bk_transformer_d_xlast(
    const TensorView &d_xlast) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_xlast) {
    return;
  }
  check_finite(ops_, d_xlast, "bk.transformer.d_xlast");
}

void TrainingDiagnosticsController::bk_transformer_d_lnf_g(
    const TensorView &d_lnf_g) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_lnf_g) {
    return;
  }
  check_finite(ops_, d_lnf_g, "bk.transformer.d_lnf_g");
}

void TrainingDiagnosticsController::bk_transformer_d_lnf_b(
    const TensorView &d_lnf_b) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_lnf_b) {
    return;
  }
  check_finite(ops_, d_lnf_b, "bk.transformer.d_lnf_b");
}

void TrainingDiagnosticsController::bk_transformer_layer_d_prev(
    int layer, const TensorView &d_prev) const {
  if (!cfg_.training.diagnostics.bk_transformer_layer_d_prev) {
    return;
  }
  check_finite(ops_, d_prev, "bk.transformer.layer" + std::to_string(layer) + ".d_prev");
}

void TrainingDiagnosticsController::bk_transformer_d_cur_before_embeddings(
    const TensorView &d_cur) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_cur_before_embeddings) {
    return;
  }
  check_finite(ops_, d_cur,
                                 "bk.transformer.d_cur_before_embeddings");
}

void TrainingDiagnosticsController::bk_transformer_d_tok(
    const TensorView &d_tok) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_tok) {
    return;
  }
  check_finite(ops_, d_tok, "bk.transformer.d_tok");
}

void TrainingDiagnosticsController::bk_transformer_d_pos(
    const TensorView &d_pos) const {
  if (!cfg_.training.diagnostics.bk_transformer_d_pos) {
    return;
  }
  check_finite(ops_, d_pos, "bk.transformer.d_pos");
}

void TrainingDiagnosticsController::bk_layer_dln2_after_ffn(
    int layer, const TensorView &dln2) const {
  if (!cfg_.training.diagnostics.bk_layer_dln2_after_ffn) {
    return;
  }
  check_finite(ops_, dln2, "bk.layer" + std::to_string(layer) + ".dln2_after_ffn");
}

void TrainingDiagnosticsController::bk_layer_dy_ln2(
    int layer, const TensorView &dy_ln2) const {
  if (!cfg_.training.diagnostics.bk_layer_dy_ln2) {
    return;
  }
  check_finite(ops_, dy_ln2, "bk.layer" + std::to_string(layer) + ".dy_ln2");
}

void TrainingDiagnosticsController::bk_layer_dln2_gamma(
    int layer, const TensorView &dln2_gamma) const {
  if (!cfg_.training.diagnostics.bk_layer_dln2_gamma) {
    return;
  }
  check_finite(ops_, dln2_gamma, "bk.layer" + std::to_string(layer) + ".dln2_gamma");
}

void TrainingDiagnosticsController::bk_layer_dln2_beta(
    int layer, const TensorView &dln2_beta) const {
  if (!cfg_.training.diagnostics.bk_layer_dln2_beta) {
    return;
  }
  check_finite(ops_, dln2_beta, "bk.layer" + std::to_string(layer) + ".dln2_beta");
}

void TrainingDiagnosticsController::bk_layer_dy_total(
    int layer, const TensorView &dy_total) const {
  if (!cfg_.training.diagnostics.bk_layer_dy_total) {
    return;
  }
  check_finite(ops_, dy_total, "bk.layer" + std::to_string(layer) + ".dy_total");
}

void TrainingDiagnosticsController::bk_layer_dln1_after_attn(
    int layer, const TensorView &dln1) const {
  if (!cfg_.training.diagnostics.bk_layer_dln1_after_attn) {
    return;
  }
  check_finite(ops_, dln1, "bk.layer" + std::to_string(layer) + ".dln1_after_attn");
}

void TrainingDiagnosticsController::bk_layer_dx_ln1(
    int layer, const TensorView &dx_ln1) const {
  if (!cfg_.training.diagnostics.bk_layer_dx_ln1) {
    return;
  }
  check_finite(ops_, dx_ln1, "bk.layer" + std::to_string(layer) + ".dx_ln1");
}

void TrainingDiagnosticsController::bk_layer_dln1_gamma(
    int layer, const TensorView &dln1_gamma) const {
  if (!cfg_.training.diagnostics.bk_layer_dln1_gamma) {
    return;
  }
  check_finite(ops_, dln1_gamma, "bk.layer" + std::to_string(layer) + ".dln1_gamma");
}

void TrainingDiagnosticsController::bk_layer_dln1_beta(
    int layer, const TensorView &dln1_beta) const {
  if (!cfg_.training.diagnostics.bk_layer_dln1_beta) {
    return;
  }
  check_finite(ops_, dln1_beta, "bk.layer" + std::to_string(layer) + ".dln1_beta");
}

void TrainingDiagnosticsController::bk_layer_dx(
    int layer, const TensorView &dx) const {
  if (!cfg_.training.diagnostics.bk_layer_dx) {
    return;
  }
  check_finite(ops_, dx, "bk.layer" + std::to_string(layer) + ".dx");
}

void TrainingDiagnosticsController::bk_ffn_dW2(
    int layer, const TensorView &dW2) const {
  if (!cfg_.training.diagnostics.bk_ffn_dW2) {
    return;
  }
  check_finite(ops_, dW2, "bk.ffn" + std::to_string(layer) + ".dW2");
}

void TrainingDiagnosticsController::bk_ffn_db2(
    int layer, const TensorView &db2) const {
  if (!cfg_.training.diagnostics.bk_ffn_db2) {
    return;
  }
  check_finite(ops_, db2, "bk.ffn" + std::to_string(layer) + ".db2");
}

void TrainingDiagnosticsController::bk_ffn_da(
    int layer, const TensorView &da) const {
  if (!cfg_.training.diagnostics.bk_ffn_da) {
    return;
  }
  check_finite(ops_, da, "bk.ffn" + std::to_string(layer) + ".da");
}

void TrainingDiagnosticsController::bk_ffn_dh(
    int layer, const TensorView &dh) const {
  if (!cfg_.training.diagnostics.bk_ffn_dh) {
    return;
  }
  check_finite(ops_, dh, "bk.ffn" + std::to_string(layer) + ".dh");
}

void TrainingDiagnosticsController::bk_ffn_dW1(
    int layer, const TensorView &dW1) const {
  if (!cfg_.training.diagnostics.bk_ffn_dW1) {
    return;
  }
  check_finite(ops_, dW1, "bk.ffn" + std::to_string(layer) + ".dW1");
}

void TrainingDiagnosticsController::bk_ffn_db1(
    int layer, const TensorView &db1) const {
  if (!cfg_.training.diagnostics.bk_ffn_db1) {
    return;
  }
  check_finite(ops_, db1, "bk.ffn" + std::to_string(layer) + ".db1");
}

void TrainingDiagnosticsController::bk_ffn_dx(
    int layer, const TensorView &dx) const {
  if (!cfg_.training.diagnostics.bk_ffn_dx) {
    return;
  }
  check_finite(ops_, dx, "bk.ffn" + std::to_string(layer) + ".dx");
}

void TrainingDiagnosticsController::bk_attn_dWo(
    int layer, const TensorView &dWo) const {
  if (!cfg_.training.diagnostics.bk_attn_dWo) {
    return;
  }
  check_finite(ops_, dWo, "bk.attn" + std::to_string(layer) + ".dWo");
}

void TrainingDiagnosticsController::bk_attn_dbo(
    int layer, const TensorView &dbo) const {
  if (!cfg_.training.diagnostics.bk_attn_dbo) {
    return;
  }
  check_finite(ops_, dbo, "bk.attn" + std::to_string(layer) + ".dbo");
}

void TrainingDiagnosticsController::bk_attn_dcontext(
    int layer, const TensorView &dcontext) const {
  if (!cfg_.training.diagnostics.bk_attn_dcontext) {
    return;
  }
  check_finite(ops_, dcontext, "bk.attn" + std::to_string(layer) + ".dcontext");
}

void TrainingDiagnosticsController::bk_attn_dweights(
    int layer, int64_t head, const TensorView &dweights) const {
  if (!cfg_.training.diagnostics.bk_attn_dweights) {
    return;
  }
  check_finite(ops_, dweights, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dweights");
}

void TrainingDiagnosticsController::bk_attn_dVh(
    int layer, int64_t head, const TensorView &dVh) const {
  if (!cfg_.training.diagnostics.bk_attn_dVh) {
    return;
  }
  check_finite(ops_, dVh, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dVh");
}

void TrainingDiagnosticsController::bk_attn_dscores_softmax_backward(
    int layer, int64_t head, const TensorView &dscores) const {
  if (!cfg_.training.diagnostics.bk_attn_dscores_softmax_backward) {
    return;
  }
  check_finite(ops_, dscores, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dscores_softmax_backward");
}

void TrainingDiagnosticsController::bk_attn_dscores_masked(
    int layer, int64_t head, const TensorView &dscores) const {
  if (!cfg_.training.diagnostics.bk_attn_dscores_masked) {
    return;
  }
  check_finite(ops_, dscores, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dscores_masked");
}

void TrainingDiagnosticsController::bk_attn_dQh(
    int layer, int64_t head, const TensorView &dQh) const {
  if (!cfg_.training.diagnostics.bk_attn_dQh) {
    return;
  }
  check_finite(ops_, dQh, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dQh");
}

void TrainingDiagnosticsController::bk_attn_dKh(
    int layer, int64_t head, const TensorView &dKh) const {
  if (!cfg_.training.diagnostics.bk_attn_dKh) {
    return;
  }
  check_finite(ops_, dKh, "bk.attn" + std::to_string(layer) + ".head" +
                      std::to_string(head) + ".dKh");
}

void TrainingDiagnosticsController::bk_attn_dx(
    int layer, const TensorView &dx) const {
  if (!cfg_.training.diagnostics.bk_attn_dx) {
    return;
  }
  check_finite(ops_, dx, "bk.attn" + std::to_string(layer) + ".dx");
}

void TrainingDiagnosticsController::bk_attn_dWqkv(
    int layer, const TensorView &dWqkv) const {
  if (!cfg_.training.diagnostics.bk_attn_dWqkv) {
    return;
  }
  check_finite(ops_, dWqkv, "bk.attn" + std::to_string(layer) + ".dWqkv");
}

void TrainingDiagnosticsController::bk_attn_dbqkv(
    int layer, const TensorView &dbqkv) const {
  if (!cfg_.training.diagnostics.bk_attn_dbqkv) {
    return;
  }
  check_finite(ops_, dbqkv, "bk.attn" + std::to_string(layer) + ".dbqkv");
}

void TrainingDiagnosticsController::after_forward(
    const TensorView &logits) const {
  if (!cfg_.training.diagnostics.fw_after_forward_logits) {
    return;
  }
  check_finite(ops_, logits, "trainer.forward_logits");
}

void TrainingDiagnosticsController::after_loss_scalar(
    const TensorView &loss_scalar) const {
  if (!cfg_.training.diagnostics.fw_after_loss_scalar) {
    return;
  }
  check_finite(ops_, loss_scalar, "trainer.loss_scalar");
}

void TrainingDiagnosticsController::after_logits_targets_backward(
    const TensorView &dlogits) const {
  if (!cfg_.training.diagnostics.fw_after_logits_targets_backward) {
    return;
  }
  check_finite(ops_, dlogits,
                                 "trainer.dlogits_from_logits_targets");
}

void TrainingDiagnosticsController::after_cross_entropy_backward(
    const TensorView &dlogits) const {
  if (!cfg_.training.diagnostics.fw_after_cross_entropy_backward) {
    return;
  }
  check_finite(ops_, dlogits,
                                 "trainer.dlogits_from_cross_entropy");
}

double TrainingDiagnosticsController::compute_global_grad_norm() const {
  const TensorView grad_view = gradientStore_.full_gradient_view();
  const double sum_sq = static_cast<double>(ops_.sum_squares_f32(grad_view));
  const double norm = std::sqrt(sum_sq);
  if (!std::isfinite(norm)) {
    throw std::runtime_error(
        "Trainer::compute_global_grad_norm produced non-finite norm | " +
        gradientStore_.first_nonfinite_diagnostic(deviceBackend_));
  }
  return norm;
}

void TrainingDiagnosticsController::replay_forward_loss_backward(
    const TrainBatch &batch) const {
  const int64_t batch_size = batch.batch_size();
  const int64_t seq_len = batch.seq_len();
  TensorView replay_logits = tensorStore_.temp_tr_logits(batch_size, seq_len);
  TensorView replay_loss_scalar = tensorStore_.temp_tr_loss();

  model_.forward(batch.ids, replay_logits);
  after_forward(replay_logits);
  if (runtimeFlags_.probe.loss) {
    ops_.cross_entropy_mean(replay_logits, batch.targets, replay_loss_scalar);
    after_loss_scalar(replay_loss_scalar);
    ops_.backward_from_logits_targets(replay_logits, batch.targets);
    after_logits_targets_backward(replay_logits);
  } else {
    ops_.cross_entropy_mean_backward_inplace(replay_logits, batch.targets,
                                             replay_loss_scalar);
    after_loss_scalar(replay_loss_scalar);
    after_cross_entropy_backward(replay_logits);
  }
  model_.backward(batch.ids, replay_logits, runtimeFlags_.probe);
  (void)compute_global_grad_norm();
}

void TrainingDiagnosticsController::handle_nonfinite_grad_norm(
    const TrainBatch &batch, const TensorView &original_dlogits,
    const std::runtime_error &original_error,
    const std::function<void()> &zero_gradients) const {
  if (!cfg_.training.diagnostics.on_nonfinite_grad_norm_replay) {
    throw original_error;
  }

  std::string original_buffer_diagnostic;
  if (cfg_.training.diagnostics.on_nonfinite_grad_norm_original_dlogits) {
    try {
      check_finite(
          ops_, original_dlogits,
          "trainer.original_dlogits_at_grad_norm_failure");
    } catch (const std::exception &diagnostic_err) {
      original_buffer_diagnostic =
          std::string(" | original buffer diagnostic: ") +
          diagnostic_err.what();
    }
  }

  zero_gradients();
  try {
    replay_forward_loss_backward(batch);
  } catch (const std::exception &diagnostic_err) {
    throw std::runtime_error(std::string(original_error.what()) +
                             original_buffer_diagnostic +
                             " | replay diagnostic: " +
                             diagnostic_err.what());
  }
  throw std::runtime_error(
      std::string(original_error.what()) + original_buffer_diagnostic +
      " | replay diagnostic: replayed forward/loss/backward did not reproduce "
      "the non-finite gradient");
}
