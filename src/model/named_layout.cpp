#include "named_layout.hpp"

#include <config.hpp>

#include <algorithm>
#include <stdexcept>

namespace {
void push_slice(std::vector<LayoutSlice> &out, uint64_t alignment,
                const std::string &name, uint64_t bytes, uint64_t &cursor,
                DType dtype = DType::F32) {
  cursor = NamedLayout::align_up(cursor, alignment);
  out.push_back(LayoutSlice{name, cursor, bytes, dtype});
  cursor += bytes;
}
}

NamedLayout build_param_layout(const Config &cfg) {
  NamedLayout layout;

  if (cfg.memory.alignment_bytes == 0) {
    throw std::runtime_error("build_param_layout: alignment_bytes must be > 0");
  }
  if (cfg.model.n_layers == 0 || cfg.model.d_model == 0 ||
      cfg.model.d_ff == 0 || cfg.model.target_vocab_size == 0 ||
      cfg.model.window_capacity == 0) {
    throw std::runtime_error("build_param_layout: model dimensions must be > 0");
  }

  uint64_t cursor = 0;

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tok_embedding",
             NamedLayout::tensor_bytes(cfg.model.target_vocab_size,
                                       cfg.model.d_model, DType::F32,
                                       "tok_embedding"),
             cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "pos_embedding",
             NamedLayout::tensor_bytes(cfg.model.window_capacity,
                                       cfg.model.d_model, DType::F32,
                                       "pos_embedding"),
             cursor);

  for (uint64_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln1_gamma",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "ln1_gamma"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln1_beta",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "ln1_beta"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_qkv_w",
               NamedLayout::tensor_bytes(cfg.model.d_model,
                                         3 * cfg.model.d_model, DType::F32,
                                         "attn_qkv_w"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_qkv_b",
               NamedLayout::tensor_bytes(1, 3 * cfg.model.d_model, DType::F32,
                                         "attn_qkv_b"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_out_w",
               NamedLayout::tensor_bytes(cfg.model.d_model, cfg.model.d_model,
                                         DType::F32, "attn_out_w"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_out_b",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "attn_out_b"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln2_gamma",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "ln2_gamma"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln2_beta",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "ln2_beta"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_w1",
               NamedLayout::tensor_bytes(cfg.model.d_model, cfg.model.d_ff,
                                         DType::F32, "ffn_w1"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_b1",
               NamedLayout::tensor_bytes(1, cfg.model.d_ff, DType::F32,
                                         "ffn_b1"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_w2",
               NamedLayout::tensor_bytes(cfg.model.d_ff, cfg.model.d_model,
                                         DType::F32, "ffn_w2"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_b2",
               NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                         "ffn_b2"),
               cursor);
  }

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "lnf_gamma",
             NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                       "lnf_gamma"),
             cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "lnf_beta",
             NamedLayout::tensor_bytes(1, cfg.model.d_model, DType::F32,
                                       "lnf_beta"),
             cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "lm_head_w",
             NamedLayout::tensor_bytes(cfg.model.d_model,
                                       cfg.model.target_vocab_size, DType::F32,
                                       "lm_head_w"),
             cursor);

  layout.total_bytes_ = NamedLayout::align_up(cursor, cfg.memory.alignment_bytes);
  return layout;
}

NamedLayout build_temp_layout(const Config &cfg) {
  NamedLayout layout;

  if (cfg.memory.alignment_bytes == 0) {
    throw std::runtime_error("build_temp_layout: alignment_bytes must be > 0");
  }
  if (cfg.training.batch_size == 0 || cfg.training.window_training == 0 ||
      cfg.model.n_layers == 0 || cfg.model.n_heads == 0 ||
      cfg.model.d_model == 0 || cfg.model.d_ff == 0 ||
      cfg.model.target_vocab_size == 0 || cfg.model.window_capacity == 0) {
    throw std::runtime_error("build_temp_layout: training/model dimensions must be > 0");
  }
  if ((cfg.model.d_model % cfg.model.n_heads) != 0) {
    throw std::runtime_error("build_temp_layout: d_model must be divisible by n_heads");
  }

  const uint64_t training_tokens =
      NamedLayout::checked_mul(cfg.training.batch_size,
                               cfg.training.window_training, "training.tokens");
  const uint64_t T = std::max<uint64_t>(training_tokens, cfg.model.window_capacity);
  const uint64_t D = cfg.model.d_model;
  const uint64_t F = cfg.model.d_ff;
  const uint64_t V = cfg.model.target_vocab_size;
  const uint64_t S = cfg.model.window_capacity;
  const uint64_t dh = D / cfg.model.n_heads;

  uint64_t cursor = 0;

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "ds.ids",
             NamedLayout::tensor_bytes(T, 1, DType::I32, "ds.ids"), cursor,
             DType::I32);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "ds.targets",
             NamedLayout::tensor_bytes(T, 1, DType::I32, "ds.targets"), cursor,
             DType::I32);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.ids",
             NamedLayout::tensor_bytes(S, 1, DType::I32, "infer.ids"), cursor,
             DType::I32);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.logits",
             NamedLayout::tensor_bytes(S, V, DType::F32, "infer.logits"),
             cursor);

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.logits",
             NamedLayout::tensor_bytes(T, V, DType::F32, "tr.logits"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.loss",
             NamedLayout::tensor_bytes(1, 1, DType::F32, "tr.loss"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.X",
             NamedLayout::tensor_bytes(T, D, DType::F32, "tr.X"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.Y",
             NamedLayout::tensor_bytes(T, D, DType::F32, "tr.Y"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.Xn",
             NamedLayout::tensor_bytes(T, D, DType::F32, "tr.Xn"), cursor);

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.XnT",
             NamedLayout::tensor_bytes(D, T, DType::F32, "bw.XnT"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_lm_w",
             NamedLayout::tensor_bytes(D, V, DType::F32, "bw.d_lm_w"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.lm_wT",
             NamedLayout::tensor_bytes(V, D, DType::F32, "bw.lm_wT"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_xn",
             NamedLayout::tensor_bytes(T, D, DType::F32, "bw.d_xn"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_xlast",
             NamedLayout::tensor_bytes(T, D, DType::F32, "bw.d_xlast"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_lnf_g",
             NamedLayout::tensor_bytes(1, D, DType::F32, "bw.d_lnf_g"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_lnf_b",
             NamedLayout::tensor_bytes(1, D, DType::F32, "bw.d_lnf_b"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_tok",
             NamedLayout::tensor_bytes(V, D, DType::F32, "bw.d_tok"), cursor);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_pos",
             NamedLayout::tensor_bytes(S, D, DType::F32, "bw.d_pos"), cursor);

  for (uint64_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln1",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.ln1"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "bw.d_prev",
               NamedLayout::tensor_bytes(T, D, DType::F32, "bw.d_prev"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_out",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.attn_out"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "resid1",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.resid1"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln2",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.ln2"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_out",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.ffn_out"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.qkv",
               NamedLayout::tensor_bytes(T, 3 * D, DType::F32, "attn.qkv"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.context",
               NamedLayout::tensor_bytes(T, D, DType::F32, "attn.context"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.scores",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.scores"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weights",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.weights"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weights_cache",
               NamedLayout::tensor_bytes(cfg.model.n_heads * T, T, DType::F32,
                                         "attn.weights_cache"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.head",
               NamedLayout::tensor_bytes(T, dh, DType::F32, "attn.head"), cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.h",
               NamedLayout::tensor_bytes(T, F, DType::F32, "ffn.h"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.a",
               NamedLayout::tensor_bytes(T, F, DType::F32, "ffn.a"), cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln2",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.dln2"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dy_ln2",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.dy_ln2"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln2_gamma",
               NamedLayout::tensor_bytes(1, D, DType::F32, "layer.dln2_gamma"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln2_beta",
               NamedLayout::tensor_bytes(1, D, DType::F32, "layer.dln2_beta"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dy_total",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.dy_total"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln1",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.dln1"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dx_ln1",
               NamedLayout::tensor_bytes(T, D, DType::F32, "layer.dx_ln1"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln1_gamma",
               NamedLayout::tensor_bytes(1, D, DType::F32, "layer.dln1_gamma"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln1_beta",
               NamedLayout::tensor_bytes(1, D, DType::F32, "layer.dln1_beta"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.contextT",
               NamedLayout::tensor_bytes(D, T, DType::F32, "attn.contextT"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dWo",
               NamedLayout::tensor_bytes(D, D, DType::F32, "attn.dWo"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dbo",
               NamedLayout::tensor_bytes(1, D, DType::F32, "attn.dbo"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.WoT",
               NamedLayout::tensor_bytes(D, D, DType::F32, "attn.WoT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dcontext",
               NamedLayout::tensor_bytes(T, D, DType::F32, "attn.dcontext"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dqkv",
               NamedLayout::tensor_bytes(T, 3 * D, DType::F32, "attn.dqkv"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.KhT",
               NamedLayout::tensor_bytes(dh, T, DType::F32, "attn.KhT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.VhT",
               NamedLayout::tensor_bytes(dh, T, DType::F32, "attn.VhT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dweights",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.dweights"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weightsT",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.weightsT"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dscores",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.dscores"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dscoresT",
               NamedLayout::tensor_bytes(T, T, DType::F32, "attn.dscoresT"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.WqkvT",
               NamedLayout::tensor_bytes(3 * D, D, DType::F32, "attn.WqkvT"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.xT",
               NamedLayout::tensor_bytes(D, T, DType::F32, "attn.xT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dWqkv",
               NamedLayout::tensor_bytes(D, 3 * D, DType::F32, "attn.dWqkv"),
               cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dbqkv",
               NamedLayout::tensor_bytes(1, 3 * D, DType::F32, "attn.dbqkv"),
               cursor);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.aT",
               NamedLayout::tensor_bytes(F, T, DType::F32, "ffn.aT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.dW2",
               NamedLayout::tensor_bytes(F, D, DType::F32, "ffn.dW2"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.db2",
               NamedLayout::tensor_bytes(1, D, DType::F32, "ffn.db2"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.W2T",
               NamedLayout::tensor_bytes(D, F, DType::F32, "ffn.W2T"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.da",
               NamedLayout::tensor_bytes(T, F, DType::F32, "ffn.da"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.dh",
               NamedLayout::tensor_bytes(T, F, DType::F32, "ffn.dh"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.xT",
               NamedLayout::tensor_bytes(D, T, DType::F32, "ffn.xT"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.dW1",
               NamedLayout::tensor_bytes(D, F, DType::F32, "ffn.dW1"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.db1",
               NamedLayout::tensor_bytes(1, F, DType::F32, "ffn.db1"), cursor);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.W1T",
               NamedLayout::tensor_bytes(F, D, DType::F32, "ffn.W1T"), cursor);
  }

  layout.total_bytes_ = NamedLayout::align_up(cursor, cfg.memory.alignment_bytes);
  return layout;
}
