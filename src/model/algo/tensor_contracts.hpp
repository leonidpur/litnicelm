#pragma once

#include "tensor.hpp"

#include <litnicelm_utils/validation.hpp>

#include <cstdint>
#include <string>

// The model's own contracts: what the layers' parameters and activations
// must look like. Each is a named group of generic validator checks, with
// the tensor's role in `who`.
namespace TensorContracts {

struct BatchSeqDims {
  int64_t batch_size;
  int64_t seq_len;
};

inline std::string role(const char *who, const char *what) {
  return std::string(who) + " " + what;
}

inline BatchSeqDims validate_bsd_io(const TensorView &x, TensorView &out,
                                    int64_t model_dim, const char *who) {
  const Validator &v = validator();
  v.validate_positive(static_cast<double>(model_dim), who, "d_model");
  v.validate_same_device_dtype(x, out, role(who, "x/out"));
  v.validate_bsd(x, -1, -1, model_dim, role(who, "x"));
  const BatchSeqDims dims{x.dim(0), x.dim(1)};
  v.validate_bsd(out, dims.batch_size, dims.seq_len, model_dim,
                 role(who, "out"));
  return dims;
}

inline BatchSeqDims validate_ids_bs(const TensorView &ids, int64_t max_seq_len,
                                    const char *who) {
  const Validator &v = validator();
  v.validate_rank(ids, 2, role(who, "ids"));
  const BatchSeqDims dims{ids.dim(0), ids.dim(1)};
  v.validate_positive(static_cast<double>(dims.batch_size), who, "ids batch");
  v.validate_positive(static_cast<double>(dims.seq_len), who, "ids seq");
  v.validate(Check::Contract, dims.seq_len <= max_seq_len, who, [&] {
    return "sequence length S=" + std::to_string(dims.seq_len) +
           " exceeds max_seq_len=" + std::to_string(max_seq_len);
  });
  return dims;
}

inline void validate_logits_bsv(const TensorView &logits, int64_t batch_size,
                                int64_t seq_len, int64_t vocab_size,
                                const char *who, const char *label) {
  validator().validate_bsd(logits, batch_size, seq_len, vocab_size,
                           role(who, label));
}

inline void validate_attention_config(int64_t model_dim, int64_t num_heads,
                                      const char *who) {
  const Validator &v = validator();
  v.validate_positive(static_cast<double>(num_heads), who, "n_heads");
  v.validate(Check::Contract, (model_dim % num_heads) == 0, who,
             "d_model must be divisible by n_heads");
}

inline void validate_ffn_params(const TensorView &W1, const TensorView &b1,
                                const TensorView &W2, const TensorView &b2,
                                int64_t model_dim, int64_t ffn_dim,
                                const char *who) {
  const Validator &v = validator();
  v.validate_dims(W1, {model_dim, ffn_dim}, role(who, "W1 [D, F]"));
  v.validate_dims(b1, {1, ffn_dim}, role(who, "b1 [1, F]"));
  v.validate_dims(W2, {ffn_dim, model_dim}, role(who, "W2 [F, D]"));
  v.validate_dims(b2, {1, model_dim}, role(who, "b2 [1, D]"));
  v.validate_same_device_dtype(W1, b1, role(who, "W1/b1"));
  v.validate_same_device_dtype(W1, W2, role(who, "W1/W2"));
  v.validate_same_device_dtype(W1, b2, role(who, "W1/b2"));
}

inline void validate_attention_params(const TensorView &Wqkv,
                                      const TensorView &bqkv,
                                      const TensorView &Wo,
                                      const TensorView &bo, int64_t model_dim,
                                      const char *who) {
  const Validator &v = validator();
  v.validate_dims(Wqkv, {model_dim, 3 * model_dim}, role(who, "Wqkv [D, 3D]"));
  v.validate_dims(bqkv, {1, 3 * model_dim}, role(who, "bqkv [1, 3D]"));
  v.validate_dims(Wo, {model_dim, model_dim}, role(who, "Wo [D, D]"));
  v.validate_dims(bo, {1, model_dim}, role(who, "bo [1, D]"));
  v.validate_same_device_dtype(Wqkv, bqkv, role(who, "Wqkv/bqkv"));
  v.validate_same_device_dtype(Wqkv, Wo, role(who, "Wqkv/Wo"));
  v.validate_same_device_dtype(Wqkv, bo, role(who, "Wqkv/bo"));
}

inline void validate_layernorm_params(const TensorView &gamma,
                                      const TensorView &beta,
                                      int64_t model_dim, const char *who,
                                      const char *prefix) {
  const Validator &v = validator();
  const std::string name = role(who, prefix);
  v.validate_dims(gamma, {1, model_dim}, name + "_gamma [1, D]");
  v.validate_dims(beta, {1, model_dim}, name + "_beta [1, D]");
  v.validate_same_device_dtype(gamma, beta, name + "_gamma/beta");
}

inline void validate_output_head_params(const TensorView &lnf_gamma,
                                        const TensorView &lnf_beta,
                                        const TensorView &lm_head_w,
                                        int64_t model_dim, int64_t vocab_size,
                                        const char *who) {
  validate_layernorm_params(lnf_gamma, lnf_beta, model_dim, who, "lnf");
  const Validator &v = validator();
  v.validate_dims(lm_head_w, {model_dim, vocab_size},
                  role(who, "lm_head_w [D, V]"));
  v.validate_same_device_dtype(lnf_gamma, lm_head_w,
                               role(who, "lnf/lm_head_w"));
}

inline void validate_transformer_embedding_params(const TensorView &tok_emb,
                                                  int64_t model_dim,
                                                  int64_t vocab_size,
                                                  const char *who) {
  validator().validate_dims(tok_emb, {vocab_size, model_dim},
                            role(who, "tok_embedding [V, D]"));
}

inline void validate_position_embedding_param(const TensorView &pos_emb,
                                              int64_t max_seq_len,
                                              int64_t model_dim,
                                              const char *who) {
  validator().validate_dims(pos_emb, {max_seq_len, model_dim},
                            role(who, "pos_embedding [S, D]"));
}

} // namespace TensorContracts
