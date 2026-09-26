#include "position_encoding.hpp"

#include "gradient_store.hpp"
#include "ops.hpp"
#include "tensor_contracts.hpp"
#include "tensor_store.hpp"
#include "training_diagnostics_controller.hpp"
#include <report_interface.hpp>

#include <stdexcept>
#include <string>

LearnedPositionEncoding::LearnedPositionEncoding(const Config &cfg)
    : cfg_(cfg) {}

void LearnedPositionEncoding::bind(TensorStore &tensor_store,
                                   GradientStore *gradient_store, Ops &ops) {
  ops_ = &ops;
  pos_emb_ = tensor_store.param_pos_embedding();
  TensorContracts::validate_position_embedding_param(
      pos_emb_, static_cast<int64_t>(cfg_.model.max_seq_len),
      static_cast<int64_t>(cfg_.model.d_model), "LearnedPositionEncoding");
  if (gradient_store != nullptr) {
    d_pos_ = gradient_store->grad_for_param(pos_emb_);
  }
}

void LearnedPositionEncoding::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void LearnedPositionEncoding::apply_to_input(TensorView &x) {
  TensorContracts::require(x.device() == pos_emb_.device(),
                           "LearnedPositionEncoding",
                           "pos_emb/x device mismatch");
  const int64_t seq_len = x.dim(1);
  TensorView pos_slice = pos_emb_.subrows(0, seq_len);
  ops_->add(x, pos_slice, x);
}

void LearnedPositionEncoding::backward_input(const TensorView &dx) {
  TensorContracts::require(d_pos_.data() != nullptr, "LearnedPositionEncoding",
                           "backward_input requires a gradient store");
  TensorContracts::require(dx.rank() == 3, "LearnedPositionEncoding",
                           "dx must be [B, S, D]");
  const int64_t batch_size = dx.dim(0);
  const int64_t seq_len = dx.dim(1);
  // d_pos[s] += sum_b dx[b, s], accumulated in batch order.
  TensorView d_pos_slice = d_pos_.subrows(0, seq_len);
  for (int64_t b = 0; b < batch_size; ++b) {
    ops_->add_inplace(d_pos_slice, dx.select(0, b));
  }
  if (diagnostics_ != nullptr) {
    diagnostics_->bk_transformer_d_pos(d_pos_);
  }
}

void LearnedPositionEncoding::report_probes(ReportSink &sink) const {
  sink.report_probe_tensor("embeddings", "pos_embedding", pos_emb_);
  if (d_pos_.data() != nullptr) {
    sink.report_probe_tensor("embeddings", "pos_embedding.grad", d_pos_);
  }
}

RopePositionEncoding::RopePositionEncoding(const Config &cfg) : cfg_(cfg) {
  const uint32_t n_heads = cfg_.model.n_heads;
  if (n_heads == 0 || cfg_.model.d_model % n_heads != 0 ||
      (cfg_.model.d_model / n_heads) % 2 != 0) {
    throw std::runtime_error(
        "RopePositionEncoding: head_dim = d_model / n_heads must be even, got "
        "d_model=" + std::to_string(cfg_.model.d_model) +
        " n_heads=" + std::to_string(n_heads));
  }
}

void RopePositionEncoding::bind(TensorStore &, GradientStore *, Ops &ops) {
  ops_ = &ops;
}

void RopePositionEncoding::apply_to_qk(TensorView &qkv) {
  rotate_qk(qkv, /*inverse=*/false);
}

// The rotation is orthogonal, so its gradient is the inverse rotation.
void RopePositionEncoding::backward_qk(TensorView &dqkv) {
  rotate_qk(dqkv, /*inverse=*/true);
}

void RopePositionEncoding::rotate_qk(TensorView &qkv, bool inverse) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t n_heads = static_cast<int64_t>(cfg_.model.n_heads);
  TensorContracts::require(qkv.rank() == 3 && qkv.dim(2) == 3 * model_dim,
                           "RopePositionEncoding", "qkv must be [B, S, 3D]");
  TensorView q = qkv.subcols(0, model_dim);
  TensorView k = qkv.subcols(model_dim, model_dim);
  ops_->rotary_embedding_inplace(q, n_heads, kBase, inverse);
  ops_->rotary_embedding_inplace(k, n_heads, kBase, inverse);
}
