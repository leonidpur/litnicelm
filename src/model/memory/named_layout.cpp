#include "named_layout.hpp"

#include <config.hpp>
#include "model_algo_factory.hpp"

#include <algorithm>
#include <stdexcept>

namespace {
void push_slice(std::vector<LayoutSlice> &out, uint64_t alignment,
                const std::string &name, uint64_t bytes, uint64_t &cursor,
                DType dtype) {
  cursor = NamedLayout::align_up(cursor, alignment);
  out.push_back(LayoutSlice{name, cursor, bytes, dtype});
  cursor += bytes;
}

struct ParamSliceSpec {
  std::string name;
  uint64_t bytes = 0;
  DType dtype;
  bool apply_weight_decay = true;
};

void append_param_spec(std::vector<ParamSliceSpec> &out, const std::string &name,
                       uint64_t bytes, bool apply_weight_decay, DType dtype) {
  out.push_back(ParamSliceSpec{name, bytes, dtype, apply_weight_decay});
}
}

DType NamedLayout::uniform_dtype() const {
  if (slices_.empty()) {
    throw std::runtime_error("NamedLayout::uniform_dtype: empty layout");
  }
  const DType dtype = slices_.front().dtype;
  for (const LayoutSlice &s : slices_) {
    if (s.dtype != dtype) {
      throw std::runtime_error("NamedLayout::uniform_dtype: mixed dtypes (" +
                               slices_.front().name + " vs " + s.name + ")");
    }
  }
  return dtype;
}

std::optional<TensorView> NamedLayout::zone_view(void *base, Device device,
                                                 DecayZone zone) const {
  const uint64_t offset = zone == DecayZone::Decay ? 0 : decay_bytes_;
  const uint64_t bytes =
      zone == DecayZone::Decay ? decay_bytes_ : total_bytes_ - decay_bytes_;
  if (bytes == 0) {
    return std::nullopt;
  }
  const DType dtype = uniform_dtype();
  const uint64_t elem = dtype_size(dtype);
  if ((offset % elem) != 0 || (bytes % elem) != 0) {
    throw std::runtime_error("NamedLayout::zone_view: zone not aligned to the element size");
  }
  return TensorView(device, dtype, reinterpret_cast<uint8_t *>(base) + offset,
                    Shape{1, static_cast<int64_t>(bytes / elem)});
}

NamedLayout NamedLayout::build_param_layout(const Config &cfg,
                                            const ModelAlgoFactory &algo,
                                            DType dtype) {
  NamedLayout layout;

  if (cfg.memory.alignment_bytes == 0) {
    throw std::runtime_error("build_param_layout: alignment_bytes must be > 0");
  }
  if (cfg.model.n_layers == 0 || cfg.model.d_model == 0 ||
      cfg.model.d_ff == 0 || cfg.model.vocab_size == 0 ||
      cfg.model.max_seq_len == 0) {
    throw std::runtime_error("build_param_layout: model dimensions must be > 0");
  }

  std::vector<ParamSliceSpec> specs;
  append_param_spec(specs, "tok_embedding",
                    NamedLayout::tensor_bytes(cfg.model.vocab_size,
                                              cfg.model.d_model, dtype,
                                              "tok_embedding"),
                    true, dtype);
  if (algo.create_position_encoding(cfg)->needs_position_table()) {
    append_param_spec(specs, "pos_embedding",
                      NamedLayout::tensor_bytes(cfg.model.max_seq_len,
                                                cfg.model.d_model, dtype,
                                                "pos_embedding"),
                      true, dtype);
  }

  for (uint64_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";

    append_param_spec(specs, p + "ln1_gamma",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                                "ln1_gamma"),
                      false, dtype);
    append_param_spec(specs, p + "ln1_beta",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                                "ln1_beta"),
                      false, dtype);

    append_param_spec(specs, p + "attn_qkv_w",
                      NamedLayout::tensor_bytes(cfg.model.d_model,
                                                3 * cfg.model.d_model,
                                                dtype, "attn_qkv_w"),
                      true, dtype);
    append_param_spec(specs, p + "attn_qkv_b",
                      NamedLayout::tensor_bytes(1, 3 * cfg.model.d_model,
                                                dtype, "attn_qkv_b"),
                      false, dtype);

    append_param_spec(specs, p + "attn_out_w",
                      NamedLayout::tensor_bytes(cfg.model.d_model,
                                                cfg.model.d_model, dtype,
                                                "attn_out_w"),
                      true, dtype);
    append_param_spec(specs, p + "attn_out_b",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model,
                                                dtype, "attn_out_b"),
                      false, dtype);

    append_param_spec(specs, p + "ln2_gamma",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                                "ln2_gamma"),
                      false, dtype);
    append_param_spec(specs, p + "ln2_beta",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                                "ln2_beta"),
                      false, dtype);

    append_param_spec(specs, p + "ffn_w1",
                      NamedLayout::tensor_bytes(cfg.model.d_model, cfg.model.d_ff,
                                                dtype, "ffn_w1"),
                      true, dtype);
    append_param_spec(specs, p + "ffn_b1",
                      NamedLayout::tensor_bytes(1, cfg.model.d_ff, dtype,
                                                "ffn_b1"),
                      false, dtype);
    append_param_spec(specs, p + "ffn_w2",
                      NamedLayout::tensor_bytes(cfg.model.d_ff, cfg.model.d_model,
                                                dtype, "ffn_w2"),
                      true, dtype);
    append_param_spec(specs, p + "ffn_b2",
                      NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                                "ffn_b2"),
                      false, dtype);
  }

  append_param_spec(specs, "lnf_gamma",
                    NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                              "lnf_gamma"),
                    false, dtype);
  append_param_spec(specs, "lnf_beta",
                    NamedLayout::tensor_bytes(1, cfg.model.d_model, dtype,
                                              "lnf_beta"),
                    false, dtype);
  append_param_spec(specs, "lm_head_w",
                    NamedLayout::tensor_bytes(cfg.model.d_model,
                                              cfg.model.vocab_size,
                                              dtype, "lm_head_w"),
                    true, dtype);

  uint64_t decay_cursor = 0;
  for (const auto &spec : specs) {
    if (!spec.apply_weight_decay) {
      continue;
    }
    decay_cursor = NamedLayout::align_up(decay_cursor, cfg.memory.alignment_bytes);
    decay_cursor += spec.bytes;
  }
  layout.decay_bytes_ =
      NamedLayout::align_up(decay_cursor, cfg.memory.alignment_bytes);

  uint64_t decay_write_cursor = 0;
  uint64_t no_decay_write_cursor = layout.decay_bytes_;
  for (const auto &spec : specs) {
    uint64_t &cursor =
        spec.apply_weight_decay ? decay_write_cursor : no_decay_write_cursor;
    cursor = NamedLayout::align_up(cursor, cfg.memory.alignment_bytes);
    layout.slices_.push_back(
        LayoutSlice{spec.name, cursor, spec.bytes, spec.dtype});
    cursor += spec.bytes;
  }
  layout.total_bytes_ =
      NamedLayout::align_up(no_decay_write_cursor, cfg.memory.alignment_bytes);
  return layout;
}

NamedLayout NamedLayout::build_training_temp_layout(
    const Config &cfg, const ModelAlgoFactory &algo, DType dtype) {
  NamedLayout layout;

  if (cfg.memory.alignment_bytes == 0) {
    throw std::runtime_error("build_temp_layout: alignment_bytes must be > 0");
  }
  if (cfg.training.batch_size == 0 || cfg.training.train_seq_len == 0 ||
      cfg.model.n_layers == 0 || cfg.model.n_heads == 0 ||
      cfg.model.d_model == 0 || cfg.model.d_ff == 0 ||
      cfg.model.vocab_size == 0 || cfg.model.max_seq_len == 0) {
    throw std::runtime_error("build_temp_layout: training/model dimensions must be > 0");
  }
  if ((cfg.model.d_model % cfg.model.n_heads) != 0) {
    throw std::runtime_error("build_temp_layout: d_model must be divisible by n_heads");
  }

  const uint64_t training_tokens =
      NamedLayout::checked_mul(cfg.training.batch_size,
                               cfg.training.train_seq_len, "training.tokens");
  const uint64_t T = training_tokens;
  const uint64_t D = cfg.model.d_model;
  const uint64_t F = cfg.model.d_ff;
  const uint64_t V = cfg.model.vocab_size;
  const uint64_t S = cfg.model.max_seq_len;
  const uint64_t train_seq = cfg.training.train_seq_len;
  const uint64_t dh = D / cfg.model.n_heads;

  uint64_t cursor = 0;

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "ds.ids",
             NamedLayout::tensor_bytes(T, 1, DType::I32, "ds.ids"), cursor,
             DType::I32);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "ds.targets",
             NamedLayout::tensor_bytes(T, 1, DType::I32, "ds.targets"), cursor,
             DType::I32);

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.logits",
             NamedLayout::tensor_bytes(T, V, dtype, "tr.logits"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.loss",
             NamedLayout::tensor_bytes(1, 1, dtype, "tr.loss"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.X",
             NamedLayout::tensor_bytes(T, D, dtype, "tr.X"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.Y",
             NamedLayout::tensor_bytes(T, D, dtype, "tr.Y"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "tr.Xn",
             NamedLayout::tensor_bytes(T, D, dtype, "tr.Xn"), cursor, dtype);

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.XnT",
             NamedLayout::tensor_bytes(D, T, dtype, "bw.XnT"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.lm_wT",
             NamedLayout::tensor_bytes(V, D, dtype, "bw.lm_wT"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_xn",
             NamedLayout::tensor_bytes(T, D, dtype, "bw.d_xn"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "bw.d_xlast",
             NamedLayout::tensor_bytes(T, D, dtype, "bw.d_xlast"), cursor, dtype);

  for (uint64_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln1",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.ln1"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "bw.d_prev",
               NamedLayout::tensor_bytes(T, D, dtype, "bw.d_prev"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_out",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.attn_out"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "resid1",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.resid1"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln2",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.ln2"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_out",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.ffn_out"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "hidden",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.hidden"),
               cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.qkv",
               NamedLayout::tensor_bytes(T, 3 * D, dtype, "attn.qkv"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.context",
               NamedLayout::tensor_bytes(T, D, dtype, "attn.context"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.scores",
               NamedLayout::tensor_bytes(T, train_seq, dtype, "attn.scores"),
               cursor, dtype);
    if (algo.attention_needs_weights_buffer()) {
      push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weights",
                 NamedLayout::tensor_bytes(T, train_seq, dtype,
                                           "attn.weights"),
                 cursor, dtype);
    }
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weights_cache",
               NamedLayout::tensor_bytes(cfg.model.n_heads * T, train_seq, dtype,
                                         "attn.weights_cache"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.head",
               NamedLayout::tensor_bytes(T, dh, dtype, "attn.head"), cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.h",
               NamedLayout::tensor_bytes(T, F, dtype, "ffn.h"), cursor, dtype);
    if (algo.ffn_needs_activation_buffers()) {
      push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.a",
                 NamedLayout::tensor_bytes(T, F, dtype, "ffn.a"), cursor, dtype);
    }

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln2",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.dln2"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dy_ln2",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.dy_ln2"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dy_total",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.dy_total"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dln1",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.dln1"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "dx_ln1",
               NamedLayout::tensor_bytes(T, D, dtype, "layer.dx_ln1"),
               cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.contextT",
               NamedLayout::tensor_bytes(D, T, dtype, "attn.contextT"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.WoT",
               NamedLayout::tensor_bytes(D, D, dtype, "attn.WoT"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dcontext",
               NamedLayout::tensor_bytes(T, D, dtype, "attn.dcontext"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dqkv",
               NamedLayout::tensor_bytes(T, 3 * D, dtype, "attn.dqkv"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.KhT",
               NamedLayout::tensor_bytes(dh, T, dtype, "attn.KhT"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.VhT",
               NamedLayout::tensor_bytes(dh, T, dtype, "attn.VhT"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dweights",
               NamedLayout::tensor_bytes(T, train_seq, dtype, "attn.dweights"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weightsT",
               NamedLayout::tensor_bytes(T, train_seq, dtype, "attn.weightsT"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dscores",
               NamedLayout::tensor_bytes(T, train_seq, dtype, "attn.dscores"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.dscoresT",
               NamedLayout::tensor_bytes(T, train_seq, dtype, "attn.dscoresT"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.WqkvT",
               NamedLayout::tensor_bytes(3 * D, D, dtype, "attn.WqkvT"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.xT",
               NamedLayout::tensor_bytes(D, T, dtype, "attn.xT"), cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.aT",
               NamedLayout::tensor_bytes(F, T, dtype, "ffn.aT"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.W2T",
               NamedLayout::tensor_bytes(D, F, dtype, "ffn.W2T"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.da",
               NamedLayout::tensor_bytes(T, F, dtype, "ffn.da"), cursor, dtype);
    if (algo.ffn_needs_activation_buffers()) {
      push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.dh",
                 NamedLayout::tensor_bytes(T, F, dtype, "ffn.dh"),
                 cursor, dtype);
    }
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.xT",
               NamedLayout::tensor_bytes(D, T, dtype, "ffn.xT"), cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.W1T",
               NamedLayout::tensor_bytes(F, D, dtype, "ffn.W1T"), cursor, dtype);
  }

  layout.total_bytes_ = NamedLayout::align_up(cursor, cfg.memory.alignment_bytes);
  return layout;
}


NamedLayout NamedLayout::build_inference_temp_layout(
    const Config &cfg, const ModelAlgoFactory &algo, DType dtype) {
  NamedLayout layout;

  if (cfg.memory.alignment_bytes == 0) {
    throw std::runtime_error("build_inference_temp_layout: alignment_bytes must be > 0");
  }
  if (cfg.model.n_layers == 0 || cfg.model.n_heads == 0 ||
      cfg.model.d_model == 0 || cfg.model.d_ff == 0 ||
      cfg.model.vocab_size == 0 || cfg.model.max_seq_len == 0) {
    throw std::runtime_error(
        "build_inference_temp_layout: model dimensions must be > 0");
  }
  if ((cfg.model.d_model % cfg.model.n_heads) != 0) {
    throw std::runtime_error(
        "build_inference_temp_layout: d_model must be divisible by n_heads");
  }

  const uint64_t D = cfg.model.d_model;
  const uint64_t F = cfg.model.d_ff;
  const uint64_t V = cfg.model.vocab_size;
  const uint64_t S = cfg.model.max_seq_len;
  const uint64_t H = cfg.model.n_heads;
  const uint64_t dh = D / H;

  uint64_t cursor = 0;

  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.ids",
             NamedLayout::tensor_bytes(S, 1, DType::I32, "infer.ids"), cursor,
             DType::I32);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.logits",
             NamedLayout::tensor_bytes(S, V, dtype, "infer.logits"),
             cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.X",
             NamedLayout::tensor_bytes(S, D, dtype, "infer.X"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.Y",
             NamedLayout::tensor_bytes(S, D, dtype, "infer.Y"), cursor, dtype);
  push_slice(layout.slices_, cfg.memory.alignment_bytes, "infer.Xn",
             NamedLayout::tensor_bytes(S, D, dtype, "infer.Xn"), cursor, dtype);

  for (uint64_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "infer.layer" + std::to_string(l) + ".";

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln1",
               NamedLayout::tensor_bytes(S, D, dtype, "infer.layer.ln1"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn_out",
               NamedLayout::tensor_bytes(S, D, dtype,
                                         "infer.layer.attn_out"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "resid1",
               NamedLayout::tensor_bytes(S, D, dtype,
                                         "infer.layer.resid1"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ln2",
               NamedLayout::tensor_bytes(S, D, dtype, "infer.layer.ln2"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn_out",
               NamedLayout::tensor_bytes(S, D, dtype,
                                         "infer.layer.ffn_out"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "hidden",
               NamedLayout::tensor_bytes(S, D, dtype,
                                         "infer.layer.hidden"),
               cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.qkv",
               NamedLayout::tensor_bytes(S, 3 * D, dtype,
                                         "infer.attn.qkv"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.context",
               NamedLayout::tensor_bytes(S, D, dtype,
                                         "infer.attn.context"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.scores",
               NamedLayout::tensor_bytes(S, S, dtype,
                                         "infer.attn.scores"),
               cursor, dtype);
    if (algo.attention_needs_weights_buffer()) {
      push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.weights",
                 NamedLayout::tensor_bytes(S, S, dtype,
                                           "infer.attn.weights"),
                 cursor, dtype);
    }
    push_slice(layout.slices_, cfg.memory.alignment_bytes,
               p + "attn.weights_cache",
               NamedLayout::tensor_bytes(H * S, S, dtype,
                                         "infer.attn.weights_cache"),
               cursor, dtype);
    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "attn.head",
               NamedLayout::tensor_bytes(S, dh, dtype,
                                         "infer.attn.head"),
               cursor, dtype);

    push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.h",
               NamedLayout::tensor_bytes(S, F, dtype, "infer.ffn.h"),
               cursor, dtype);
    if (algo.ffn_needs_activation_buffers()) {
      push_slice(layout.slices_, cfg.memory.alignment_bytes, p + "ffn.a",
                 NamedLayout::tensor_bytes(S, F, dtype, "infer.ffn.a"),
                 cursor, dtype);
    }
  }

  layout.total_bytes_ = NamedLayout::align_up(cursor, cfg.memory.alignment_bytes);
  return layout;
}

NamedLayout NamedLayout::build_temp_layout(const Config &cfg,
                                           const ModelAlgoFactory &algo,
                                           DType dtype) {
  return build_training_temp_layout(cfg, algo, dtype);
}
