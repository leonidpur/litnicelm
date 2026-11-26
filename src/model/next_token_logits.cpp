// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "next_token_logits.hpp"

#include "algo/transformer.hpp"
#include "backend/device_backend.hpp"
#include "host_tensor_stage.hpp"
#include "tensor_store.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// The last `context` tokens of ids.
std::vector<int32_t> context_window(const std::vector<int32_t> &ids,
                                    uint32_t context) {
  if (ids.size() <= static_cast<size_t>(context)) {
    return ids;
  }
  return std::vector<int32_t>(ids.end() - static_cast<std::ptrdiff_t>(context),
                              ids.end());
}

// Runs ids at absolute positions [first_position, ...) and returns the last
// row's logits as FP32 on the host.
std::vector<float> run_last_row(Transformer &model, TensorStore &tensor_store,
                                DeviceBackend &backend,
                                const std::vector<int32_t> &ids,
                                int64_t first_position) {
  const int64_t rows = static_cast<int64_t>(ids.size());
  TensorView ids_t = tensor_store.temp_infer_ids(rows);
  Tensor ids_host = Tensor::make_cpu(DType::I32, ids_t.shape());
  std::copy(ids.begin(), ids.end(),
            reinterpret_cast<int32_t *>(ids_host.view().data()));
  host_tensor_stage::copy_from_cpu(backend, ids_host.view(), ids_t,
                                   "NextTokenLogits(ids)");
  TensorView logits = tensor_store.temp_infer_logits(rows);
  model.forward(ids_t, logits, nullptr, first_position);

  const TensorView last = logits.select(0, 0).subrows(rows - 1, 1);
  Tensor host = host_tensor_stage::stage_to_cpu_f32(backend, last,
                                                    "NextTokenLogits(logits)");
  const TensorView row = host.view();
  std::vector<float> out(static_cast<size_t>(row.dim(1)));
  for (int64_t c = 0; c < row.dim(1); ++c) {
    out[static_cast<size_t>(c)] = row.at_f32(0, c);
  }
  return out;
}

} // namespace

FullWindowLogits::FullWindowLogits(Transformer &model, TensorStore &tensor_store,
                                   DeviceBackend &backend, uint32_t context)
    : model_(model), tensorStore_(tensor_store), backend_(backend),
      context_(context) {}

std::vector<float> FullWindowLogits::next_logits(const std::vector<int32_t> &ids) {
  return run_last_row(model_, tensorStore_, backend_,
                      context_window(ids, context_), 0);
}

KvCacheLogits::KvCacheLogits(Transformer &model, TensorStore &tensor_store,
                             DeviceBackend &backend, uint32_t context)
    : model_(model), tensorStore_(tensor_store), backend_(backend),
      context_(context) {}

std::vector<float> KvCacheLogits::next_logits(const std::vector<int32_t> &ids) {
  if (cached_ > 0 && cached_ + 1 == ids.size() && ids.size() <= context_) {
    const int64_t position = static_cast<int64_t>(cached_);
    cached_ += 1;
    return run_last_row(model_, tensorStore_, backend_, {ids.back()}, position);
  }
  const std::vector<int32_t> window = context_window(ids, context_);
  cached_ = window.size();
  return run_last_row(model_, tensorStore_, backend_, window, 0);
}

std::unique_ptr<INextTokenLogits>
make_next_token_logits(const Config &cfg, Transformer &model,
                       TensorStore &tensor_store, DeviceBackend &backend,
                       uint32_t context) {
  if (cfg.inference.kv_cache) {
    return std::make_unique<KvCacheLogits>(model, tensor_store, backend, context);
  }
  return std::make_unique<FullWindowLogits>(model, tensor_store, backend, context);
}
