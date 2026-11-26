// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "cudnn_attention.hpp"
#include "cuda_checks.hpp"

#include <stdexcept>
#include <string>

#ifdef LITNICELM_WITH_CUDNN

#include <cudnn_frontend.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
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

using VariantPack = std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *>;

// Another plan replaces the heuristic's first choice only when it is at least
// this much faster, so near-ties cannot flip the plan, and with it the
// rounding, from run to run.
constexpr float kMinPlanSpeedup = 0.95f;
constexpr int kPlanWarmupRuns = 2;
constexpr int kPlanTimedRuns = 5;

struct CudnnAttention::Impl {
  // How a graph's plan is chosen, decided when the graph is created.
  // Heuristics rank cuDNN's plans, but for small batch x heads their first
  // choice can be far from the fastest (a causal forward at B=1, H=4, S=1024
  // ran 110 us on it against 59 us on its third plan), so training times
  // every supported plan once. Inference (no softmax statistics) meets a new
  // sequence length at nearly every prompt, and timing every plan of each
  // cost ~0.1 s, so it runs the heuristic's first choice untimed.
  struct PlanSelection {
    virtual ~PlanSelection() = default;
    virtual fe::BuildPlanPolicy_t build_policy() const = 0;
    // The plan to run on the graph built with build_policy().
    virtual int64_t plan_index(Impl &impl, const fe::graph::Graph &graph,
                               const VariantPack &pack, const char *what,
                               double build_ms) = 0;
  };

  // Every plan heuristic A supports, timed on the first call (choose_plan).
  struct TimedPlanSelection final : PlanSelection {
    int64_t index = -1;  // -1 until chosen on the first call
    fe::BuildPlanPolicy_t build_policy() const override {
      return fe::BuildPlanPolicy_t::ALL;
    }
    int64_t plan_index(Impl &impl, const fe::graph::Graph &graph,
                       const VariantPack &pack, const char *what,
                       double build_ms) override {
      if (index < 0) {
        index = impl.choose_plan(graph, pack, what, build_ms);
      }
      return index;
    }
  };

  // Only the heuristic's first choice is built; it runs untimed.
  struct HeuristicPlanSelection final : PlanSelection {
    fe::BuildPlanPolicy_t build_policy() const override {
      return fe::BuildPlanPolicy_t::HEURISTICS_CHOICE;
    }
    int64_t plan_index(Impl &, const fe::graph::Graph &, const VariantPack &,
                       const char *, double) override {
      return 0;
    }
  };

  // Training graphs (with softmax statistics, and every backward) are timed.
  static std::unique_ptr<PlanSelection> selection_for(bool training) {
    if (training) {
      return std::make_unique<TimedPlanSelection>();
    }
    return std::make_unique<HeuristicPlanSelection>();
  }

  // A built SDPA graph and how its plan is chosen.
  struct SdpaPlan {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unique_ptr<PlanSelection> selection;
    double build_ms = 0.0;  // building the plans, part of the selection cost
  };

  cudnnHandle_t handle = nullptr;
  void *workspace = nullptr;
  int64_t workspace_bytes = 0;
  // Keyed by direction, shape and every stride.
  std::map<std::vector<int64_t>, SdpaPlan> plans;

  void *workspace_of(const fe::graph::Graph &graph, int64_t index) {
    int64_t bytes = 0;
    check_fe(graph.get_workspace_size_plan_at_index(index, bytes),
             "get_workspace_size");
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

  // Builds the plans the selection asks for, in heuristic A's ranking order.
  void build(SdpaPlan &plan, std::shared_ptr<fe::graph::Graph> graph,
             std::unique_ptr<PlanSelection> selection) {
    const auto start = std::chrono::steady_clock::now();
    check_fe(graph->build(handle, {fe::HeurMode_t::A}, selection->build_policy()),
             "graph build");
    plan.build_ms = elapsed_ms(start);
    plan.graph = std::move(graph);
    plan.selection = std::move(selection);
  }

  static double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
  }

  // execute_plan_at_index adds pointers to values it holds only for that
  // call (the attention scale) to the pack it is given, and later calls keep
  // them, so every execution gets its own copy of the pack.
  fe::error_t run_plan(const fe::graph::Graph &graph, const VariantPack &pack,
                       void *ws, int64_t index) {
    VariantPack fresh = pack;
    return graph.execute_plan_at_index(handle, fresh, ws, index);
  }

  // Median time of one execution in microseconds, or -1 when the plan does
  // not execute. Its outputs are rewritten by the call that follows.
  float time_plan(const fe::graph::Graph &graph, const VariantPack &pack, int64_t index) {
    void *ws = workspace_of(graph, index);
    for (int i = 0; i < kPlanWarmupRuns; ++i) {
      if (!run_plan(graph, pack, ws, index).is_good()) {
        return -1.0f;
      }
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
    std::vector<float> us(kPlanTimedRuns);
    for (float &t : us) {
      check_cuda(cudaEventRecord(start), "cudaEventRecord");
      check_fe(run_plan(graph, pack, ws, index), "sdpa plan timing");
      check_cuda(cudaEventRecord(stop), "cudaEventRecord");
      check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
      check_cuda(cudaEventElapsedTime(&t, start, stop), "cudaEventElapsedTime");
      t *= 1000.0f;
    }
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy");
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
  }

  std::string plan_name(const fe::graph::Graph &graph, int64_t index) {
    std::string name;
    check_fe(graph.get_plan_name_at_index(index, name), "get_plan_name");
    return name;
  }

  // The heuristic's first plan (index 0) unless another runs at least
  // kMinPlanSpeedup faster; logged so a run's plan can be reproduced, with
  // the one-time selection cost (building plus timing every plan).
  int64_t choose_plan(const fe::graph::Graph &graph, const VariantPack &pack,
                      const char *what, double build_ms) {
    const auto start = std::chrono::steady_clock::now();
    const int64_t count = graph.get_execution_plan_count();
    std::vector<float> us(static_cast<size_t>(count));
    int64_t fastest = -1;
    for (int64_t i = 0; i < count; ++i) {
      us[i] = time_plan(graph, pack, i);
      if (us[i] >= 0.0f && (fastest < 0 || us[i] < us[fastest])) {
        fastest = i;
      }
    }
    if (fastest < 0) {
      throw std::runtime_error(std::string("cuda_cublas_plugin: cuDNN ") + what +
                               ": no plan executes");
    }
    int64_t chosen = fastest;
    if (us[0] >= 0.0f && us[fastest] > kMinPlanSpeedup * us[0]) {
      chosen = 0;
    }
    std::cerr << "cuda_cublas_plugin: cuDNN " << what << " plan " << chosen << "/"
              << count << " " << plan_name(graph, chosen) << " (" << us[chosen]
              << " us; heuristic first choice " << us[0] << " us; selection "
              << build_ms + elapsed_ms(start) << " ms)\n";
    return chosen;
  }

  void execute(SdpaPlan &plan, const VariantPack &pack, const char *what) {
    const int64_t index =
        plan.selection->plan_index(*this, *plan.graph, pack, what, plan.build_ms);
    check_fe(run_plan(*plan.graph, pack, workspace_of(*plan.graph, index), index),
             what);
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

  Impl::SdpaPlan &plan = impl_->plans[key];
  if (plan.graph == nullptr) {
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
    impl_->build(plan, g, Impl::selection_for(training));
  }

  auto *base = static_cast<uint8_t *>(qkv.data());
  const size_t third = static_cast<size_t>(D) * dtype_size(qkv.dtype());
  VariantPack pack{
      {kQ, base}, {kK, base + third}, {kV, base + 2 * third}, {kO, out.data()}};
  if (training) {
    pack[kStats] = stats.data();
  }
  impl_->execute(plan, pack, "sdpa forward");
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

  Impl::SdpaPlan &plan = impl_->plans[key];
  if (plan.graph == nullptr) {
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
    impl_->build(plan, g, Impl::selection_for(/*training=*/true));
  }

  const size_t third = static_cast<size_t>(D) * dtype_size(qkv.dtype());
  auto *base = static_cast<uint8_t *>(qkv.data());
  auto *dbase = static_cast<uint8_t *>(dqkv.data());
  VariantPack pack{
      {kQ, base},          {kK, base + third},       {kV, base + 2 * third},
      {kO, out.data()},    {kDO, dout.data()},       {kStats, stats.data()},
      {kDQ, dbase},        {kDK, dbase + third},     {kDV, dbase + 2 * third}};
  impl_->execute(plan, pack, "sdpa backward");
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
