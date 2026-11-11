// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "cudnn_attention.hpp"
#include "cuda_checks.hpp"

#include <stdexcept>
#include <string>

#ifdef LITNICELM_WITH_CUDNN

#include <cudnn_frontend.h>

#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace cuda_cublas_plugin {

namespace {

enum Uid : fe::graph::Tensor_attributes::uid_t {
  kQ = 1, kK, kV, kO, kStats, kDO, kDQ, kDK, kDV,
};

// Strides in elements of the [B, H, S, head_dim] view of a [B, S, H*hd]
// tensor whose rows (S) and batches (B) may be strided.
std::vector<int64_t> bhsd_strides(const TensorView &bsx, int64_t head_dim) {
  const int64_t elem = static_cast<int64_t>(dtype_size(bsx.dtype()));
  return {bsx.stride_bytes(0) / elem, head_dim, bsx.stride_bytes(1) / elem, 1};
}

std::vector<int64_t> stats_strides(const TensorView &stats) {
  const int64_t elem = static_cast<int64_t>(sizeof(float));
  return {stats.stride_bytes(0) / elem, stats.stride_bytes(1) / elem,
          stats.stride_bytes(2) / elem, 1};
}

int64_t scale_bits(float scale) {
  uint32_t bits = 0;
  std::memcpy(&bits, &scale, sizeof(bits));
  return static_cast<int64_t>(bits);
}

void check_fe(fe::error_t status, const char *what) {
  if (!status.is_good()) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: cuDNN ") + what +
                             " failed: " + status.get_message());
  }
}

void check_cudnn(cudnnStatus_t status, const char *what) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " failed: " + cudnnGetErrorString(status));
  }
}

fe::graph::Tensor_attributes bhsd(const char *name, Uid uid, int64_t B,
                                  int64_t H, int64_t S, int64_t hd,
                                  const std::vector<int64_t> &strides) {
  return fe::graph::Tensor_attributes()
      .set_name(name)
      .set_uid(uid)
      .set_dim({B, H, S, hd})
      .set_stride(strides);
}

} // namespace

struct CudnnAttention::Impl {
  cudnnHandle_t handle = nullptr;
  void *workspace = nullptr;
  int64_t workspace_bytes = 0;
  // Keyed by direction, shape and every stride.
  std::map<std::vector<int64_t>, std::shared_ptr<fe::graph::Graph>> graphs;

  void *workspace_of(const fe::graph::Graph &graph) {
    int64_t bytes = 0;
    check_fe(graph.get_workspace_size(bytes), "get_workspace_size");
    if (bytes > workspace_bytes) {
      if (workspace != nullptr) {
        check_cuda(cudaFree(workspace), "cudaFree(cuDNN workspace)");
        workspace = nullptr;
      }
      check_cuda(cudaMalloc(&workspace, static_cast<size_t>(bytes)),
                 "cudaMalloc(cuDNN workspace)");
      workspace_bytes = bytes;
    }
    return workspace;
  }

  std::shared_ptr<fe::graph::Graph> build(std::shared_ptr<fe::graph::Graph> graph) {
    check_fe(graph->build(handle, {fe::HeurMode_t::A}), "graph build");
    return graph;
  }

  static std::shared_ptr<fe::graph::Graph> new_graph() {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    return graph;
  }
};

CudnnAttention::CudnnAttention() : impl_(std::make_unique<Impl>()) {
  check_cudnn(cudnnCreate(&impl_->handle), "cudnnCreate");
}

CudnnAttention::~CudnnAttention() {
  if (impl_->workspace != nullptr) {
    cudaFree(impl_->workspace);
  }
  if (impl_->handle != nullptr) {
    cudnnDestroy(impl_->handle);
  }
}

bool CudnnAttention::available() { return true; }

void CudnnAttention::forward(const TensorView &qkv, int64_t n_heads,
                             float scale, TensorView &out, TensorView &stats) {
  const int64_t B = qkv.dim(0);
  const int64_t S = qkv.dim(1);
  const int64_t D = out.dim(2);
  const int64_t hd = D / n_heads;
  const bool training = stats.data() != nullptr;
  const std::vector<int64_t> qkv_strides = bhsd_strides(qkv, hd);
  const std::vector<int64_t> out_strides = bhsd_strides(out, hd);
  std::vector<int64_t> key{0, B, n_heads, S, hd, training ? 1 : 0};
  key.insert(key.end(), qkv_strides.begin(), qkv_strides.end());
  key.insert(key.end(), out_strides.begin(), out_strides.end());
  std::vector<int64_t> st_strides;
  if (training) {
    st_strides = stats_strides(stats);
    key.insert(key.end(), st_strides.begin(), st_strides.end());
  }
  key.push_back(scale_bits(scale));

  std::shared_ptr<fe::graph::Graph> &graph = impl_->graphs[key];
  if (graph == nullptr) {
    auto g = Impl::new_graph();
    auto Q = g->tensor(bhsd("Q", kQ, B, n_heads, S, hd, qkv_strides));
    auto K = g->tensor(bhsd("K", kK, B, n_heads, S, hd, qkv_strides));
    auto V = g->tensor(bhsd("V", kV, B, n_heads, S, hd, qkv_strides));
    auto options = fe::graph::SDPA_attributes()
                       .set_name("causal_attention")
                       .set_generate_stats(training)
                       .set_attn_scale(scale)
                       .set_causal_mask(true);
    auto [O, Stats] = g->sdpa(Q, K, V, options);
    O->set_output(true).set_uid(kO).set_dim({B, n_heads, S, hd}).set_stride(out_strides);
    if (training) {
      Stats->set_output(true)
          .set_uid(kStats)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({B, n_heads, S, 1})
          .set_stride(st_strides);
    }
    graph = impl_->build(g);
  }

  auto *base = static_cast<uint8_t *>(qkv.data());
  const size_t third = static_cast<size_t>(D) * dtype_size(qkv.dtype());
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> pack{
      {kQ, base}, {kK, base + third}, {kV, base + 2 * third}, {kO, out.data()}};
  if (training) {
    pack[kStats] = stats.data();
  }
  check_fe(graph->execute(impl_->handle, pack, impl_->workspace_of(*graph)),
           "sdpa forward");
}

void CudnnAttention::backward(const TensorView &qkv, const TensorView &out,
                              const TensorView &dout, const TensorView &stats,
                              int64_t n_heads, float scale, TensorView &dqkv) {
  const int64_t B = qkv.dim(0);
  const int64_t S = qkv.dim(1);
  const int64_t D = out.dim(2);
  const int64_t hd = D / n_heads;
  const std::vector<int64_t> qkv_strides = bhsd_strides(qkv, hd);
  const std::vector<int64_t> out_strides = bhsd_strides(out, hd);
  const std::vector<int64_t> dout_strides = bhsd_strides(dout, hd);
  const std::vector<int64_t> dqkv_strides = bhsd_strides(dqkv, hd);
  const std::vector<int64_t> st_strides = stats_strides(stats);
  std::vector<int64_t> key{1, B, n_heads, S, hd};
  for (const auto *strides : {&qkv_strides, &out_strides, &dout_strides,
                              &dqkv_strides, &st_strides}) {
    key.insert(key.end(), strides->begin(), strides->end());
  }
  key.push_back(scale_bits(scale));

  std::shared_ptr<fe::graph::Graph> &graph = impl_->graphs[key];
  if (graph == nullptr) {
    auto g = Impl::new_graph();
    auto Q = g->tensor(bhsd("Q", kQ, B, n_heads, S, hd, qkv_strides));
    auto K = g->tensor(bhsd("K", kK, B, n_heads, S, hd, qkv_strides));
    auto V = g->tensor(bhsd("V", kV, B, n_heads, S, hd, qkv_strides));
    auto O = g->tensor(bhsd("O", kO, B, n_heads, S, hd, out_strides));
    auto dO = g->tensor(bhsd("dO", kDO, B, n_heads, S, hd, dout_strides));
    auto Stats = g->tensor(fe::graph::Tensor_attributes()
                               .set_name("Stats")
                               .set_uid(kStats)
                               .set_dim({B, n_heads, S, 1})
                               .set_stride(st_strides)
                               .set_data_type(fe::DataType_t::FLOAT));
    auto options = fe::graph::SDPA_backward_attributes()
                       .set_name("causal_attention_backward")
                       .set_attn_scale(scale)
                       .set_causal_mask(true);
    auto [dQ, dK, dV] = g->sdpa_backward(Q, K, V, O, dO, Stats, options);
    dQ->set_output(true).set_uid(kDQ).set_dim({B, n_heads, S, hd}).set_stride(dqkv_strides);
    dK->set_output(true).set_uid(kDK).set_dim({B, n_heads, S, hd}).set_stride(dqkv_strides);
    dV->set_output(true).set_uid(kDV).set_dim({B, n_heads, S, hd}).set_stride(dqkv_strides);
    graph = impl_->build(g);
  }

  const size_t third = static_cast<size_t>(D) * dtype_size(qkv.dtype());
  auto *base = static_cast<uint8_t *>(qkv.data());
  auto *dbase = static_cast<uint8_t *>(dqkv.data());
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> pack{
      {kQ, base},          {kK, base + third},       {kV, base + 2 * third},
      {kO, out.data()},    {kDO, dout.data()},       {kStats, stats.data()},
      {kDQ, dbase},        {kDK, dbase + third},     {kDV, dbase + 2 * third}};
  check_fe(graph->execute(impl_->handle, pack, impl_->workspace_of(*graph)),
           "sdpa backward");
}

} // namespace cuda_cublas_plugin

#else // !LITNICELM_WITH_CUDNN

namespace cuda_cublas_plugin {

struct CudnnAttention::Impl {};

CudnnAttention::CudnnAttention() {
  throw std::runtime_error("cuda_cublas_plugin: built without cuDNN");
}
CudnnAttention::~CudnnAttention() = default;
bool CudnnAttention::available() { return false; }
void CudnnAttention::forward(const TensorView &, int64_t, float, TensorView &,
                             TensorView &) {}
void CudnnAttention::backward(const TensorView &, const TensorView &,
                              const TensorView &, const TensorView &, int64_t,
                              float, TensorView &) {}

} // namespace cuda_cublas_plugin

#endif
