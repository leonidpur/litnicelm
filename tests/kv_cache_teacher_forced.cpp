// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

// Teacher-forced check of the KV cache on a trained model: the same token
// sequence goes through the no-cache path (one forward over the whole
// window, as inference.kv_cache: false runs every step) and the cached path
// (prefill of the first `prompt` tokens, then one token per call at its
// absolute position), and the FP32 logits are compared at every position.
// Prints the max abs logit difference, the positions whose argmax differs,
// and passes when the max difference is within the tolerance.
//
// Usage: litnice_kv_cache_teacher_forced <config.yaml> <tokens.bin>
//          <tokens> <prompt> <tolerance>
// The tokens are the first <tokens> of a LitniceLM .bin dataset.

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "host_tensor_stage.hpp"
#include "memory/inference_memory_manager.hpp"
#include "model_algo_factory.hpp"
#include "ops.hpp"
#include "runtime_config.hpp"
#include "transformer.hpp"
#include <config.hpp>
#include <tokenizer_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> read_tokens(const std::string &path, size_t count) {
  std::ifstream in(path, std::ios::binary);
  uint32_t header[3] = {0, 0, 0};
  in.read(reinterpret_cast<char *>(header), sizeof(header));
  if (!in || header[0] != 0x4C4E4750u) {
    throw std::runtime_error(path + ": not a LitniceLM dataset");
  }
  std::vector<uint32_t> raw(count);
  in.read(reinterpret_cast<char *>(raw.data()),
          static_cast<std::streamsize>(count * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error(path + ": fewer tokens than requested");
  }
  return std::vector<int32_t>(raw.begin(), raw.end());
}

struct Model {
  Config cfg;
  std::unique_ptr<DeviceBackend> backend;
  PrecisionPolicy precision;
  std::unique_ptr<Tokenizer> tokenizer;
  ModelAlgoFactory algo;
  std::unique_ptr<InferenceMemoryManager> memory;
  Ops ops;
  std::unique_ptr<Transformer> transformer;

  explicit Model(const Config &base)
      : cfg(base),
        backend(DeviceBackend::create_instance(cfg)),
        precision(PrecisionPolicy::create(cfg, backend->precision_variants())),
        tokenizer(TokenizerFactory::create(cfg)),
        algo(ModelAlgoConfig::from_config(cfg), ModelRole::Inference),
        ops(*backend) {
    cfg = with_tokenizer_vocab_size(cfg, *tokenizer);
    memory = std::make_unique<InferenceMemoryManager>(cfg, algo, *backend, precision);
    transformer = std::make_unique<Transformer>(cfg, algo, memory->tensor_store(),
                                                nullptr, ops);
    CheckpointIO(cfg, algo, *backend, memory->data_arena(), memory->adam_state())
        .load(cfg.paths.model_file_best);
    memory->sync_compute_weights();
  }

  // Logits [ids.size(), V] (F32, host) of ids at [first_position, ...).
  std::vector<float> forward(const std::vector<int32_t> &ids, int64_t first_position) {
    const int64_t rows = static_cast<int64_t>(ids.size());
    TensorStore &store = memory->tensor_store();
    TensorView ids_t = store.temp_infer_ids(rows);
    Tensor ids_host = Tensor::make_cpu(DType::I32, ids_t.shape());
    std::copy(ids.begin(), ids.end(), reinterpret_cast<int32_t *>(ids_host.view().data()));
    host_tensor_stage::copy_from_cpu(*backend, ids_host.view(), ids_t, "teacher_forced ids");
    TensorView logits = store.temp_infer_logits(rows);
    transformer->forward(ids_t, logits, nullptr, first_position);
    Tensor host = host_tensor_stage::stage_to_cpu_f32(*backend, logits, "teacher_forced logits");
    const float *p = reinterpret_cast<const float *>(host.view().data());
    return std::vector<float>(p, p + rows * static_cast<int64_t>(cfg.model.vocab_size));
  }
};

size_t argmax(const float *row, size_t n) {
  return static_cast<size_t>(std::max_element(row, row + n) - row);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: " << argv[0]
              << " <config.yaml> <tokens.bin> <tokens> <prompt> <tolerance>\n";
    return 2;
  }
  try {
    Config cfg = Config::load_from_file(argv[1]);
    cfg.inference.kv_cache = true;
    const size_t count = std::stoul(argv[3]);
    const size_t prompt = std::stoul(argv[4]);
    const double tolerance = std::stod(argv[5]);
    if (prompt == 0 || prompt > count || count > cfg.model.max_seq_len) {
      throw std::runtime_error("need 0 < prompt <= tokens <= max_seq_len");
    }
    const std::vector<int32_t> ids = read_tokens(argv[2], count);
    Model model(cfg);
    const size_t V = model.cfg.model.vocab_size;

    const std::vector<float> full = model.forward(ids, 0);
    std::vector<float> cached =
        model.forward(std::vector<int32_t>(ids.begin(), ids.begin() + prompt), 0);
    for (size_t t = prompt; t < count; ++t) {
      const std::vector<float> row = model.forward({ids[t]}, static_cast<int64_t>(t));
      cached.insert(cached.end(), row.begin(), row.end());
    }

    double worst = 0.0;
    size_t worst_at = 0;
    size_t argmax_differs = 0;
    for (size_t t = 0; t < count; ++t) {
      const float *a = full.data() + t * V;
      const float *b = cached.data() + t * V;
      for (size_t v = 0; v < V; ++v) {
        const double d = std::fabs(static_cast<double>(a[v]) - b[v]);
        if (d > worst) {
          worst = d;
          worst_at = t;
        }
      }
      if (argmax(a, V) != argmax(b, V)) {
        ++argmax_differs;
        std::cout << "argmax differs at position " << t << "\n";
      }
    }
    const bool ok = worst <= tolerance;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << "activations " << (model.precision.dtype(ItemType::Activation) == DType::BF16 ? "bf16" : "fp32")
              << ", " << count << " tokens, prompt " << prompt
              << ": max |logit(no cache) - logit(cache)| = " << worst
              << " at position " << worst_at << " (tolerance " << tolerance
              << "); argmax differs at " << argmax_differs << " positions\n";
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
