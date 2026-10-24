#include <inference.hpp>

#include "host_tensor_stage.hpp"
#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include <config.hpp>
#include "dataset.hpp"
#include "memory/inference_memory_manager.hpp"
#include "ops.hpp"
#include <tokenizer_factory.hpp>
#include "algo/transformer.hpp"
#include <litnicelm_utils/log.hpp>
#include "model_events.hpp"
#include "program_terminal.hpp"
#include "runtime_config.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void validate_vocab_contract_or_throw(const Config &cfg,
                                      const Tokenizer &tokenizer) {
  const uint32_t tok_vocab = static_cast<uint32_t>(tokenizer.vocab_size());
  const DatasetHeader header =
      TextDataset::read_header_or_throw(cfg.tokenization.output_binary);
  const uint32_t ds_vocab = header.vocab_size;
  if (tok_vocab != ds_vocab) {
    throw std::runtime_error("Mismatched Vocab: tokenizer(" +
                             std::to_string(tok_vocab) + ") != dataset header(" +
                             std::to_string(ds_vocab) + ")");
  }
}

struct InferRuntime {
  Config cfg;
  std::unique_ptr<DeviceBackend> backend;
  // Dtypes of every tensor the memory manager lays out.
  const PrecisionPolicy precision;
  std::unique_ptr<Tokenizer> tokenizer;
  // Model algorithm choices; memory and model are built with it.
  ModelAlgoFactory algo;
  std::unique_ptr<InferenceMemoryManager> memory_manager;
  const Ops ops;
  std::unique_ptr<Transformer> model;

  explicit InferRuntime(const Config &base_cfg)
      : cfg(base_cfg),
        backend(DeviceBackend::create_instance(cfg)),
        precision(PrecisionPolicy::create(cfg, backend->precision_variants())),
        tokenizer(TokenizerFactory::create(cfg)),
        algo(ModelAlgoConfig::from_config(cfg)),
        ops(*backend) {
    cfg = with_tokenizer_vocab_size(cfg, *tokenizer);
    validate_vocab_contract_or_throw(cfg, *tokenizer);
    memory_manager = std::make_unique<InferenceMemoryManager>(cfg, algo,
                                                              *backend, precision);
    model = std::make_unique<Transformer>(cfg, algo,
                                          memory_manager->tensor_store(),
                                          nullptr, ops);
  }
};

void load_weights_or_throw(InferRuntime &rt) {
  CheckpointIO(rt.cfg, rt.algo, *rt.backend, rt.memory_manager->data_arena(),
               rt.memory_manager->adam_state())
      .load(rt.cfg.paths.model_file_best);
  rt.memory_manager->sync_compute_weights();
}

std::vector<float> read_last_row_logits(DeviceBackend &backend,
                                        const TensorView &logits) {
  Tensor logits_host =
      host_tensor_stage::stage_to_cpu_f32(backend, logits, "Inference::argmax_last_row");
  TensorView logits_view = logits_host.view();
  if (logits_view.rank() == 3) {
    if (logits_view.dim(0) != 1) {
      throw std::runtime_error(
          "read_last_row_logits expects inference logits with batch_size == 1");
    }
    logits_view = logits_view.select(0, 0);
  }
  if (logits_view.rank() != 2) {
    throw std::runtime_error("read_last_row_logits expects rank-2 or rank-3 logits");
  }

  const int64_t r = logits_view.dim(0) - 1;
  const int64_t C = logits_view.dim(1);
  std::vector<float> row_logits(static_cast<size_t>(C), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    row_logits[static_cast<size_t>(c)] = logits_view.at_f32(r, c);
  }
  return row_logits;
}

std::vector<int32_t> prompt_ids(const Tokenizer &tokenizer,
                                const std::string &prompt) {
  std::vector<int32_t> ids = tokenizer.encode(prompt);
  if (ids.empty()) {
    ids.push_back(0);
  }
  return ids;
}

uint32_t trained_context(const InferRuntime &rt) {
  return std::min(rt.cfg.model.max_seq_len, rt.cfg.training.train_seq_len);
}

// The model only sees the last `context` tokens: generation slides this
// window forward instead of stopping once the text reaches the context size.
std::vector<int32_t> context_window(const std::vector<int32_t> &ids,
                                    uint32_t context) {
  if (ids.size() <= static_cast<size_t>(context)) {
    return ids;
  }
  return std::vector<int32_t>(ids.end() - static_cast<std::ptrdiff_t>(context),
                              ids.end());
}

TensorView forward_prompt(InferRuntime &rt, const std::vector<int32_t> &ids) {
  const int64_t prompt_token_rows = static_cast<int64_t>(ids.size());
  TensorView ids_t =
      rt.memory_manager->tensor_store().temp_infer_ids(prompt_token_rows);
  Tensor ids_host = Tensor::make_cpu(DType::I32, ids_t.shape());
  auto *p = reinterpret_cast<int32_t *>(ids_host.view().data());
  for (size_t i = 0; i < ids.size(); ++i) {
    p[i] = ids[i];
  }
  host_tensor_stage::copy_from_cpu(*rt.backend, ids_host.view(), ids_t,
                                   "Inference::forward_prompt(ids)");

  TensorView logits =
      rt.memory_manager->tensor_store().temp_infer_logits(prompt_token_rows);
  rt.model->forward(ids_t, logits);
  return logits;
}

std::string decode_token_safe(const Tokenizer &tokenizer, int32_t id) {
  try {
    return tokenizer.decode({id});
  } catch (...) {
    return "<invalid>";
  }
}

std::string escape_token_for_log(const std::string &token) {
  std::string out;
  out.reserve(token.size());
  for (char ch : token) {
    const unsigned char byte = static_cast<unsigned char>(ch);
    if (byte == 92) {
      out.push_back(static_cast<char>(92));
      out.push_back(static_cast<char>(92));
      continue;
    }
    if (byte == 34) {
      out.push_back(static_cast<char>(92));
      out.push_back(static_cast<char>(34));
      continue;
    }
    if (byte == 10) {
      out.push_back(static_cast<char>(92));
      out.push_back(static_cast<char>(110));
      continue;
    }
    if (byte == 13) {
      out.push_back(static_cast<char>(92));
      out.push_back(static_cast<char>(114));
      continue;
    }
    if (byte == 9) {
      out.push_back(static_cast<char>(92));
      out.push_back(static_cast<char>(116));
      continue;
    }
    out += ch;
  }
  return out;
}

int32_t select_next_token_greedy(const std::vector<float> &row_logits) {
  int32_t best = 0;
  float bestv = row_logits.at(0);
  for (size_t i = 1; i < row_logits.size(); ++i) {
    if (row_logits[i] > bestv) {
      bestv = row_logits[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

int32_t select_next_token_sampled(const std::vector<float> &row_logits,
                                  const InferenceConfig &cfg,
                                  std::mt19937 &rng) {
  if (row_logits.empty()) {
    throw std::runtime_error("select_next_token_sampled: empty logits");
  }
  if (cfg.temp <= 0.0f || cfg.top_k == 1) {
    return select_next_token_greedy(row_logits);
  }
  if (!(cfg.top_p > 0.0f && cfg.top_p <= 1.0f)) {
    throw std::runtime_error("InferenceConfig.top_p must be in (0,1]");
  }

  struct Candidate {
    int32_t id = 0;
    double scaled_logit = 0.0;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(row_logits.size());
  const double inv_temp = 1.0 / static_cast<double>(cfg.temp);
  for (size_t i = 0; i < row_logits.size(); ++i) {
    candidates.push_back(
        Candidate{static_cast<int32_t>(i),
                  static_cast<double>(row_logits[i]) * inv_temp});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.scaled_logit > b.scaled_logit;
            });

  if (cfg.top_k > 0 && cfg.top_k < candidates.size()) {
    candidates.resize(static_cast<size_t>(cfg.top_k));
  }

  if (cfg.top_p < 1.0f) {
    const double max_logit = candidates.front().scaled_logit;
    std::vector<double> probs(candidates.size(), 0.0);
    double prob_sum = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      const double p = std::exp(candidates[i].scaled_logit - max_logit);
      probs[i] = p;
      prob_sum += p;
    }
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += probs[keep] / prob_sum;
      if (cumulative >= static_cast<double>(cfg.top_p)) {
        ++keep;
        break;
      }
    }
    candidates.resize(std::max<size_t>(1, keep));
  }

  const double max_logit = candidates.front().scaled_logit;
  std::vector<double> weights(candidates.size(), 0.0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    weights[i] = std::exp(candidates[i].scaled_logit - max_logit);
  }
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return candidates[dist(rng)].id;
}

// Softmax probabilities of the two most likely next tokens.
struct TopTwo {
  double first = 0.0;
  double second = 0.0;
};

TopTwo top_two_probabilities(const std::vector<float> &row_logits) {
  const float max_logit = *std::max_element(row_logits.begin(), row_logits.end());
  double sum = 0.0;
  double first = 0.0;
  double second = 0.0;
  for (const float logit : row_logits) {
    const double w = std::exp(static_cast<double>(logit - max_logit));
    sum += w;
    if (w > first) {
      second = first;
      first = w;
    } else if (w > second) {
      second = w;
    }
  }
  return {first / sum, second / sum};
}

std::string generate_text(InferRuntime &rt, const std::string &prompt) {
  std::vector<int32_t> ids = prompt_ids(*rt.tokenizer, prompt);
  std::mt19937 rng(rt.cfg.inference.seed);

  const uint32_t max_seq = trained_context(rt);
  const uint32_t max_new = rt.cfg.inference.max_new;
  std::ostringstream start;
  start << R"(Inference started: prompt=")" << escape_token_for_log(prompt)
        << R"(" max_new=)" << max_new << " temp=" << rt.cfg.inference.temp
        << " top_k=" << rt.cfg.inference.top_k
        << " top_p=" << rt.cfg.inference.top_p
        << " seed=" << rt.cfg.inference.seed
        << " trained_context=" << max_seq;
  if (ids.size() > static_cast<size_t>(max_seq)) {
    start << " (prompt_tokens=" << ids.size() << ", model sees the last "
          << max_seq << ")";
  }
  logger().log(model_events::kInferenceStart, start.str());
  TopTwo least_margin{1.0, 0.0};
  uint32_t least_margin_step = 0;
  for (uint32_t step = 0; step < max_new; ++step) {
    TensorView logits = forward_prompt(rt, context_window(ids, max_seq));
    const std::vector<float> row_logits =
        read_last_row_logits(*rt.backend, logits);
    if (rt.cfg.inference.report_margin) {
      const TopTwo top = top_two_probabilities(row_logits);
      if (top.first - top.second < least_margin.first - least_margin.second) {
        least_margin = top;
        least_margin_step = step;
      }
    }
    const int32_t next =
        select_next_token_sampled(row_logits, rt.cfg.inference, rng);
    ids.push_back(next);
  }
  if (rt.cfg.inference.report_margin && max_new > 0) {
    std::ostringstream margin;
    margin << std::fixed << std::setprecision(6)
           << "min top-1 minus top-2 probability: "
           << least_margin.first - least_margin.second << " at step "
           << least_margin_step << " (top-1 " << least_margin.first << ", top-2 "
           << least_margin.second << ")";
    logger().log(model_events::kInferenceMargin, margin.str());
  }

  logger().log(model_events::kInferenceEnd, "Inference complete: tokens=" +
                                          std::to_string(ids.size()));
  return rt.tokenizer->decode(ids);
}

std::string inspect_distribution(InferRuntime &rt, const std::string &prompt) {
  constexpr size_t kTopK = 5;

  const std::vector<int32_t> ids = context_window(
      prompt_ids(*rt.tokenizer, prompt), trained_context(rt));
  const TensorView logits = forward_prompt(rt, ids);
  const std::vector<float> row_logits = read_last_row_logits(*rt.backend, logits);
  const int64_t cols = static_cast<int64_t>(row_logits.size());
  float max_logit = row_logits.at(0);
  for (int64_t c = 1; c < cols; ++c) {
    if (row_logits[static_cast<size_t>(c)] > max_logit) {
      max_logit = row_logits[static_cast<size_t>(c)];
    }
  }

  std::vector<double> probs(static_cast<size_t>(cols), 0.0);
  double sum = 0.0;
  for (int64_t c = 0; c < cols; ++c) {
    const double w = std::exp(static_cast<double>(row_logits[static_cast<size_t>(c)] -
                                                  max_logit));
    probs[static_cast<size_t>(c)] = w;
    sum += w;
  }
  for (double &p : probs) {
    p /= sum;
  }

  std::vector<int32_t> order(static_cast<size_t>(cols), 0);
  for (int64_t c = 0; c < cols; ++c) {
    order[static_cast<size_t>(c)] = static_cast<int32_t>(c);
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
  });

  double entropy = 0.0;
  for (double p : probs) {
    if (p > 0.0) {
      entropy -= p * std::log(p);
    }
  }

  std::ostringstream oss;
  oss << R"(Prompt=")" << prompt << R"(")"
      << static_cast<char>(10);
  oss << R"(Argmax: id=)" << order.front() << R"( token=")"
      << decode_token_safe(*rt.tokenizer, order.front()) << R"(")"
      << static_cast<char>(10);
  oss << "Entropy: " << std::fixed << std::setprecision(6) << entropy
      << static_cast<char>(10);
  oss << "Top-" << std::min(kTopK, order.size())
      << " next-token distribution:" << static_cast<char>(10);
  for (size_t i = 0; i < std::min(kTopK, order.size()); ++i) {
    const int32_t id = order[i];
    oss << (i + 1) << ". id=" << id << R"( token=")"
        << decode_token_safe(*rt.tokenizer, id) << R"(")"
        << " logit=" << std::fixed << std::setprecision(6)
        << row_logits[static_cast<size_t>(id)]
        << " prob=" << probs[static_cast<size_t>(id)]
        << static_cast<char>(10);
  }
  return oss.str();
}
} // namespace

int run_infer_mode(const Config &cfg, ProgramTerminal &terminal) {
  InferRuntime rt(cfg);
  load_weights_or_throw(rt);
  terminal.generated_text(generate_text(rt, rt.cfg.inference.prompt));
  return 0;
}

int run_inspect_mode(const Config &cfg, ProgramTerminal &terminal) {
  InferRuntime rt(cfg);
  load_weights_or_throw(rt);
  terminal.token_distribution(inspect_distribution(rt, rt.cfg.inference.prompt));
  return 0;
}

int run_inferloop_mode(const Config &cfg, ProgramTerminal &terminal) {
  InferRuntime rt(cfg);
  load_weights_or_throw(rt);

  terminal.loop_ready();
  std::string prompt;
  while (true) {
    terminal.prompt();
    if (!std::getline(std::cin, prompt)) {
      break;
    }
    if (prompt == ":q" || prompt == "exit" || prompt == "quit") {
      break;
    }
    terminal.generated_text(generate_text(rt, prompt));
  }
  terminal.loop_ended();
  return 0;
}
