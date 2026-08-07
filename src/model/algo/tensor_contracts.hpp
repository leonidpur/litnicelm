#pragma once

#include "tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace TensorContracts {

struct BatchSeqDims {
  int64_t batch_size;
  int64_t seq_len;
};

inline void require(bool cond, const char *who, const std::string &msg) {
  if (!cond) [[unlikely]] {
    throw std::runtime_error(std::string(who) + ": " + msg);
  }
}

inline void validate_same_device_dtype(const TensorView &a,
                                       const TensorView &b,
                                       const char *who,
                                       const char *label) {
  require(a.device() == b.device(), who,
          std::string(label) + " device mismatch");
  require(a.dtype() == b.dtype(), who, std::string(label) + " dtype mismatch");
}

inline void validate_bsd_tensor(const TensorView &x, int64_t model_dim,
                                const char *who, const char *label) {
  require(model_dim > 0, who, "d_model must be positive");
  require(x.rank() == 3, who,
          std::string(label) + " must be semantic [B, S, D]");
  require(x.dim(0) > 0 && x.dim(1) > 0, who,
          std::string(label) + " must define positive [B, S]");
  require(x.dim(2) == model_dim, who,
          std::string(label) + ".dim(2) != d_model");
  require(x.dim(0) * x.dim(1) ==
              static_cast<int64_t>(x.numel() / static_cast<uint64_t>(model_dim)),
          who, std::string(label) + " [B, S, D] must cover all elements");
}

inline BatchSeqDims validate_bsd_io(const TensorView &x, TensorView &out,
                                    int64_t model_dim, const char *who) {
  validate_same_device_dtype(x, out, who, "x/out");
  validate_bsd_tensor(x, model_dim, who, "x");
  const BatchSeqDims dims{x.dim(0), x.dim(1)};
  require(out.rank() == 3 && out.dim(0) == dims.batch_size &&
              out.dim(1) == dims.seq_len && out.dim(2) == model_dim,
          who, "out must be semantic [B, S, D]");
  return dims;
}

inline void validate_bsd_shape_like(const TensorView &tensor,
                                    const TensorView &reference,
                                    const char *who, const char *label) {
  require(tensor.rank() == reference.rank() &&
              tensor.dim(0) == reference.dim(0) &&
              tensor.dim(1) == reference.dim(1) &&
              tensor.dim(2) == reference.dim(2),
          who, std::string(label) + " shape mismatch");
}

inline BatchSeqDims validate_ids_bs(const TensorView &ids, int64_t max_seq_len,
                                    const char *who) {
  require(ids.rank() == 2, who, "ids must be semantic [B, S]");
  const BatchSeqDims dims{ids.dim(0), ids.dim(1)};
  require(dims.batch_size > 0 && dims.seq_len > 0, who,
          "ids must define [B, S] with positive dims");
  require(dims.batch_size * dims.seq_len == static_cast<int64_t>(ids.numel()),
          who, "ids [B, S] must cover all tokens");
  require(dims.seq_len <= max_seq_len, who,
          "sequence length S=" + std::to_string(dims.seq_len) +
              " exceeds max_seq_len=" + std::to_string(max_seq_len));
  return dims;
}

inline void validate_logits_bsv(const TensorView &logits, int64_t batch_size,
                                int64_t seq_len, int64_t vocab_size,
                                const char *who, const char *label) {
  require(logits.rank() == 3 && logits.dim(0) == batch_size &&
              logits.dim(1) == seq_len && logits.dim(2) == vocab_size,
          who, std::string(label) + " must be semantic [B, S, V]");
}

inline void validate_attention_config(int64_t model_dim, int64_t num_heads,
                                      const char *who) {
  require(num_heads > 0, who, "n_heads must be > 0");
  require((model_dim % num_heads) == 0, who,
          "d_model must be divisible by n_heads");
}

inline void validate_ffn_params(const TensorView &W1, const TensorView &b1,
                                const TensorView &W2, const TensorView &b2,
                                int64_t model_dim, int64_t ffn_dim,
                                const char *who) {
  require(W1.dim(0) == model_dim && W1.dim(1) == ffn_dim, who,
          "W1 must be [D, F]");
  require(b1.dim(0) == 1 && b1.dim(1) == ffn_dim, who, "b1 must be [1, F]");
  require(W2.dim(0) == ffn_dim && W2.dim(1) == model_dim, who,
          "W2 must be [F, D]");
  require(b2.dim(0) == 1 && b2.dim(1) == model_dim, who, "b2 must be [1, D]");
  require(W1.device() == W2.device() && W1.device() == b1.device() &&
              W1.device() == b2.device(),
          who, "FFN parameter devices must match");
  require(W1.dtype() == W2.dtype() && W1.dtype() == b1.dtype() &&
              W1.dtype() == b2.dtype(),
          who, "FFN parameter dtypes must match");
}

inline void validate_attention_params(const TensorView &Wqkv,
                                      const TensorView &bqkv,
                                      const TensorView &Wo,
                                      const TensorView &bo, int64_t model_dim,
                                      const char *who) {
  require(Wqkv.dim(0) == model_dim && Wqkv.dim(1) == 3 * model_dim, who,
          "Wqkv must be [D, 3D]");
  require(bqkv.dim(0) == 1 && bqkv.dim(1) == 3 * model_dim, who,
          "bqkv must be [1, 3D]");
  require(Wo.dim(0) == model_dim && Wo.dim(1) == model_dim, who,
          "Wo must be [D, D]");
  require(bo.dim(0) == 1 && bo.dim(1) == model_dim, who, "bo must be [1, D]");
  require(Wqkv.device() == bqkv.device() && Wqkv.device() == Wo.device() &&
              Wqkv.device() == bo.device(),
          who, "attention parameter devices must match");
  require(Wqkv.dtype() == bqkv.dtype() && Wqkv.dtype() == Wo.dtype() &&
              Wqkv.dtype() == bo.dtype(),
          who, "attention parameter dtypes must match");
}

inline void validate_layernorm_params(const TensorView &gamma,
                                      const TensorView &beta,
                                      int64_t model_dim, const char *who,
                                      const char *prefix) {
  require(gamma.dim(0) == 1 && gamma.dim(1) == model_dim, who,
          std::string(prefix) + "_gamma must be [1, D]");
  require(beta.dim(0) == 1 && beta.dim(1) == model_dim, who,
          std::string(prefix) + "_beta must be [1, D]");
  validate_same_device_dtype(gamma, beta, who, prefix);
}

inline void validate_output_head_params(const TensorView &lnf_gamma,
                                        const TensorView &lnf_beta,
                                        const TensorView &lm_head_w,
                                        int64_t model_dim, int64_t vocab_size,
                                        const char *who) {
  validate_layernorm_params(lnf_gamma, lnf_beta, model_dim, who, "lnf");
  require(lm_head_w.dim(0) == model_dim && lm_head_w.dim(1) == vocab_size, who,
          "lm_head_w must be [D, V]");
  validate_same_device_dtype(lnf_gamma, lm_head_w, who, "lnf/lm_head_w");
}

inline void validate_transformer_embedding_params(const TensorView &tok_emb,
                                                  const TensorView &pos_emb,
                                                  int64_t model_dim,
                                                  int64_t vocab_size,
                                                  int64_t max_seq_len,
                                                  const char *who) {
  require(tok_emb.dim(0) == vocab_size && tok_emb.dim(1) == model_dim, who,
          "tok_embedding must be [V, D]");
  require(pos_emb.dim(0) == max_seq_len && pos_emb.dim(1) == model_dim, who,
          "pos_embedding must be [S, D]");
  validate_same_device_dtype(tok_emb, pos_emb, who,
                             "transformer parameters");
}

} // namespace TensorContracts
