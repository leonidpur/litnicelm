#include "cuda_backend.hpp"
#include "cublas_gemm.hpp"
#include "cuda_kernel_launchers.hpp"
#include "cuda_tensor_view.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_cublas_plugin {
namespace {

class CuBlasPluginBackend final : public DeviceBackend {
  struct CudaExecutionLane {
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaEvent_t done = nullptr;
  };

public:
  CuBlasPluginBackend() {
    check_cublas(cublasCreate(&handle_), "cublasCreate");
    const cublasStatus_t math_status =
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      check_cublas(math_status, "cublasSetMathMode");
    }
  }

  ~CuBlasPluginBackend() override {
    for (CudaExecutionLane &lane : exec_lanes_) {
      if (lane.cublas != nullptr) {
        cublasDestroy(lane.cublas);
      }
      if (lane.done != nullptr) {
        cudaEventDestroy(lane.done);
      }
      if (lane.stream != nullptr) {
        cudaStreamDestroy(lane.stream);
      }
    }
    if (exec_ready_event_ != nullptr) {
      cudaEventDestroy(exec_ready_event_);
    }
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  Device device() const override { return Device::GPU; }

  void *alloc(uint64_t bytes, uint32_t alignment) override {
    if (bytes == 0) {
      return nullptr;
    }
    if (!is_power_of_two(alignment)) {
      throw std::invalid_argument("cuda_cublas_plugin: invalid allocation alignment");
    }
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, static_cast<size_t>(bytes)), "cudaMalloc");
    return ptr;
  }

  void free(void *ptr) override {
    if (ptr == nullptr) {
      return;
    }
    check_cuda(cudaFree(ptr), "cudaFree");
  }

  DeviceMemoryInfo memory_info() const override {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      return {};
    }
    return DeviceMemoryInfo{true, static_cast<uint64_t>(free_bytes),
                            static_cast<uint64_t>(total_bytes)};
  }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (dst == nullptr || src == nullptr) {
      throw std::invalid_argument("cuda_cublas_plugin: copy_host2device null pointer");
    }
    check_cuda(cudaMemcpy(dst, src, static_cast<size_t>(bytes),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(host_to_device)");
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (dst == nullptr || src == nullptr) {
      throw std::invalid_argument("cuda_cublas_plugin: copy_device2host null pointer");
    }
    check_cuda(cudaMemcpy(dst, src, static_cast<size_t>(bytes),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(device_to_host)");
  }

  void copy(const TensorView &src, TensorView &dst) override {
    require_cuda_row_major(src, "copy(src)");
    require_cuda_row_major(dst, "copy(dst)");
    copy_tensor_2d(src, dst, cudaMemcpyDeviceToDevice, "cudaMemcpy2D(copy)");
  }

  void fill(TensorView &t, float v) override {
    require_cuda_f32_row_major(t, "fill");
    if (tensor_rows(t) == 0 || tensor_cols(t) == 0) {
      return;
    }
    launch_fill(t, v);
  }

  void add(const TensorView &a, const TensorView &b, TensorView &out) override {
    require_cuda_f32_row_major(a, "add(a)");
    require_cuda_f32_row_major(b, "add(b)");
    require_cuda_f32_row_major(out, "add(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    if (a.rank() == 3 && b.rank() == 2 && out.rank() == 3 &&
        a.dim(1) == b.dim(0) && a.dim(2) == b.dim(1) &&
        out.dim(0) == a.dim(0) && out.dim(1) == a.dim(1) &&
        out.dim(2) == a.dim(2)) {
      launch_add_batch_seq_plus_pos(a, b, out, a.dim(1));
      return;
    }
    launch_add(a, b, out);
  }

  void add_inplace(TensorView &a, const TensorView &b) override {
    require_cuda_f32_row_major(a, "add_inplace(a)");
    require_cuda_f32_row_major(b, "add_inplace(b)");
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0) {
      return;
    }
    launch_add_inplace(a, b);
  }

  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override {
    require_cuda_f32_row_major(x, "add_bias_rowwise(x)");
    require_cuda_f32_row_major(bias_1xC, "add_bias_rowwise(bias)");
    require_cuda_f32_row_major(out, "add_bias_rowwise(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_add_bias_rowwise(x, bias_1xC, out);
  }

  void add_bias_relu_rowwise(const TensorView &x, const TensorView &bias_1xC,
                             TensorView &out) override {
    require_cuda_f32_row_major(x, "add_bias_relu_rowwise(x)");
    require_cuda_f32_row_major(bias_1xC, "add_bias_relu_rowwise(bias)");
    require_cuda_f32_row_major(out, "add_bias_relu_rowwise(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_add_bias_relu_rowwise(x, bias_1xC, out);
  }

  void add_bias_relu_rowwise_inplace(TensorView &x,
                                     const TensorView &bias_1xC) override {
    require_cuda_f32_row_major(x, "add_bias_relu_rowwise_inplace(x)");
    require_cuda_f32_row_major(bias_1xC,
                               "add_bias_relu_rowwise_inplace(bias)");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return;
    }
    launch_add_bias_relu_rowwise_inplace(x, bias_1xC);
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    require_cuda_f32_row_major(x, "mul_scalar(x)");
    require_cuda_f32_row_major(out, "mul_scalar(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_mul_scalar(x, s, out);
  }

  float sum_squares_f32(const TensorView &x) override {
    require_cuda_f32_row_major(x, "sum_squares_f32(x)");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return 0.0f;
    }
    float *device_sum_sq = nullptr;
    check_cuda(cudaMalloc(&device_sum_sq, sizeof(float)),
               "cudaMalloc(sum_squares_f32)");
    check_cuda(cudaMemset(device_sum_sq, 0, sizeof(float)),
               "cudaMemset(sum_squares_f32)");
    launch_sum_squares_f32(x, device_sum_sq);
    float host_sum_sq = 0.0f;
    copy_device2host(&host_sum_sq, device_sum_sq, sizeof(float));
    check_cuda(cudaFree(device_sum_sq), "cudaFree(sum_squares_f32)");
    return host_sum_sq;
  }

  void relu(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "relu(x)");
    require_cuda_f32_row_major(out, "relu(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_relu(x, out);
  }

  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override {
    require_cuda_f32_row_major(preact, "relu_backward(preact)");
    require_cuda_f32_row_major(dout, "relu_backward(dout)");
    require_cuda_f32_row_major(dx, "relu_backward(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    launch_relu_backward(preact, dout, dx);
  }

  void relu_backward_inplace(const TensorView &preact,
                             TensorView &dout_dx) override {
    require_cuda_f32_row_major(preact, "relu_backward_inplace(preact)");
    require_cuda_f32_row_major(dout_dx, "relu_backward_inplace(dout_dx)");
    if (tensor_rows(dout_dx) == 0 || tensor_cols(dout_dx) == 0) {
      return;
    }
    launch_relu_backward_inplace(preact, dout_dx);
  }

  void row_sum(const TensorView &x, TensorView &out_1xC) override {
    require_cuda_f32_row_major(x, "row_sum(x)");
    require_cuda_f32_row_major(out_1xC, "row_sum(out)");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return;
    }
    launch_row_sum(x, out_1xC);
  }

  void gemm(const TensorView &a, const TensorView &b,
            TensorView &out) override {
    cublas_gemm(handle_, a, b, out);
  }

  void gemm_ranked_matrix_rhs(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    cublas_gemm_ranked_matrix_rhs(handle_, a, b, out);
  }

  void gemm_ranked_matrix_rhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    cublas_gemm_ranked_matrix_rhs_t(handle_, a, b, out);
  }

  void gemm_ranked_reduce_lhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    cublas_gemm_ranked_reduce_lhs_t(handle_, a, b, out);
  }

  void gemm_batched(const TensorView &a, const TensorView &b,
                    TensorView &out) override {
    cublas_gemm_batched(handle_, a, b, out);
  }

  void gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    cublas_gemm_batched_lhs_t(handle_, a, b, out);
  }

  void gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    cublas_gemm_batched_rhs_t(handle_, a, b, out);
  }

  void transpose(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "transpose(x)");
    require_cuda_f32_row_major(out, "transpose(out)");
    if (x.data() == out.data()) {
      throw std::runtime_error(
          "cuda_cublas_plugin: transpose requires distinct input and output buffers");
    }
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_transpose(x, out);
  }

  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override {
    require_cuda_f32_row_major(x, "layernorm_forward(x)");
    require_cuda_f32_row_major(gamma_1xC, "layernorm_forward(gamma)");
    require_cuda_f32_row_major(beta_1xC, "layernorm_forward(beta)");
    require_cuda_f32_row_major(out, "layernorm_forward(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_layernorm_forward(x, gamma_1xC, beta_1xC, out);
  }

  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override {
    require_cuda_f32_row_major(x, "layernorm_backward(x)");
    require_cuda_f32_row_major(gamma_1xC, "layernorm_backward(gamma)");
    require_cuda_f32_row_major(dout, "layernorm_backward(dout)");
    require_cuda_f32_row_major(dx, "layernorm_backward(dx)");
    require_cuda_f32_row_major(dgamma_1xC, "layernorm_backward(dgamma)");
    require_cuda_f32_row_major(dbeta_1xC, "layernorm_backward(dbeta)");
    check_cuda(cudaMemset(dgamma_1xC.data(), 0,
                          static_cast<size_t>(row_bytes(dgamma_1xC))),
               "cudaMemset(layernorm_backward(dgamma))");
    check_cuda(cudaMemset(dbeta_1xC.data(), 0,
                          static_cast<size_t>(row_bytes(dbeta_1xC))),
               "cudaMemset(layernorm_backward(dbeta))");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return;
    }
    launch_layernorm_backward(x, gamma_1xC, dout, dx, dgamma_1xC, dbeta_1xC);
  }

  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override {
    require_cuda_f32_row_major(table, "embedding_lookup(table)");
    require_cuda_row_major(ids, "embedding_lookup(ids)");
    require_cuda_f32_row_major(out, "embedding_lookup(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_embedding_lookup(table, ids, out);
  }

  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur, TensorView &d_tok,
                                  TensorView &d_pos) override {
    require_cuda_row_major(ids, "accumulate_embedding_grads(ids)");
    require_cuda_f32_row_major(d_cur, "accumulate_embedding_grads(d_cur)");
    require_cuda_f32_row_major(d_tok, "accumulate_embedding_grads(d_tok)");
    require_cuda_f32_row_major(d_pos, "accumulate_embedding_grads(d_pos)");
    if (tensor_rows(d_cur) == 0 || tensor_cols(d_cur) == 0) {
      return;
    }
    launch_accumulate_embedding_grads(ids, d_cur, d_tok, d_pos,
                                      ids.dim(ids.rank() - 1));
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    fallback_cross_entropy_mean(logits, targets, out_loss);
  }

  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override {
    require_cuda_f32_row_major(logits, "cross_entropy_mean_backward_inplace(logits)");
    require_cuda_row_major(targets, "cross_entropy_mean_backward_inplace(targets)");
    require_cuda_f32_row_major(out_loss, "cross_entropy_mean_backward_inplace(out_loss)");
    check_cuda(cudaMemset(out_loss.data(), 0, sizeof(float)),
               "cudaMemset(cross_entropy_mean_backward_inplace(out_loss))");
    launch_cross_entropy_mean_backward_inplace(logits, targets, out_loss);
  }

  float read_scalar_f32(const TensorView &x) override {
    require_cuda_f32_row_major(x, "read_scalar_f32");
    float value = 0.0f;
    copy_device2host(&value, x.data(), sizeof(float));
    return value;
  }

  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override {
    fallback_backward_from_logits_targets(logits, targets);
  }

  void softmax_rows(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "softmax_rows(x)");
    require_cuda_f32_row_major(out, "softmax_rows(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_softmax_rows(x, out);
  }

  void softmax_backward_rows(const TensorView &softmax, const TensorView &dout,
                             TensorView &dx) override {
    require_cuda_f32_row_major(softmax, "softmax_backward_rows(softmax)");
    require_cuda_f32_row_major(dout, "softmax_backward_rows(dout)");
    require_cuda_f32_row_major(dx, "softmax_backward_rows(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    launch_softmax_backward_rows(softmax, dout, dx);
  }

  void scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                  TensorView &out) override {
    require_cuda_f32_row_major(scores, "scaled_causal_softmax_rows(scores)");
    require_cuda_f32_row_major(out, "scaled_causal_softmax_rows(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    launch_scaled_causal_softmax_rows(scores, scale, out);
  }

  bool supports_exec_context_iteration() const override { return true; }

  void start_exec_context_iteration() override {
    ensure_exec_lanes();
    exec_lane_index_ = 0;
    active_exec_lane_index_ = 0;
    used_exec_lane_count_ = 0;
    has_active_exec_group_ = false;
    in_exec_iteration_ = true;

    check_cuda(cudaEventRecord(exec_ready_event_, 0),
               "cudaEventRecord(exec context ready)");
    for (CudaExecutionLane &lane : exec_lanes_) {
      check_cuda(cudaStreamWaitEvent(lane.stream, exec_ready_event_, 0),
                 "cudaStreamWaitEvent(exec context ready)");
    }
  }

  void finish_exec_context_iteration() override {
    if (!in_exec_iteration_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: finish_exec_context_iteration without active iteration");
    }
    if (has_active_exec_group_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: finish_exec_context_iteration with active group");
    }
    for (size_t lane_index = 0; lane_index < used_exec_lane_count_;
         ++lane_index) {
      CudaExecutionLane &lane = exec_lanes_[lane_index];
      check_cuda(cudaEventRecord(lane.done, lane.stream),
                 "cudaEventRecord(exec context done)");
      check_cuda(cudaStreamWaitEvent(0, lane.done, 0),
                 "cudaStreamWaitEvent(exec context done)");
    }
    in_exec_iteration_ = false;
  }

  void start_exec_context_group() override {
    if (!in_exec_iteration_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: start_exec_context_group without active iteration");
    }
    if (has_active_exec_group_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: nested exec context groups are not supported");
    }
    if (exec_lanes_.empty()) {
      throw std::runtime_error("cuda_cublas_plugin: no exec context lanes");
    }
    active_exec_lane_index_ = exec_lane_index_ % exec_lanes_.size();
    used_exec_lane_count_ =
        std::max(used_exec_lane_count_, active_exec_lane_index_ + 1);
    has_active_exec_group_ = true;
  }

  void finish_exec_context_group() override {
    if (!has_active_exec_group_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: finish_exec_context_group without active group");
    }
    has_active_exec_group_ = false;
    ++exec_lane_index_;
  }

  void gemm_batched_exec_context(const TensorView &a, const TensorView &b,
                                 TensorView &out) override {
    cublas_gemm_batched(active_exec_lane().cublas, a, b, out);
  }

  void gemm_batched_lhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) override {
    cublas_gemm_batched_lhs_t(active_exec_lane().cublas, a, b, out);
  }

  void gemm_batched_rhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) override {
    cublas_gemm_batched_rhs_t(active_exec_lane().cublas, a, b, out);
  }

  void scaled_causal_softmax_rows_exec_context(
      const TensorView &scores, float scale, TensorView &out) override {
    require_cuda_f32_row_major(scores,
                               "scaled_causal_softmax_rows_exec_context(scores)");
    require_cuda_f32_row_major(out,
                               "scaled_causal_softmax_rows_exec_context(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    CudaExecutionLane &lane = active_exec_lane();
    launch_scaled_causal_softmax_rows_on_stream(scores, scale, out,
                                                lane.stream);
  }

  void softmax_backward_causal_rows_exec_context(const TensorView &softmax,
                                                 const TensorView &dout,
                                                 TensorView &dx) override {
    require_cuda_f32_row_major(
        softmax, "softmax_backward_causal_rows_exec_context(softmax)");
    require_cuda_f32_row_major(
        dout, "softmax_backward_causal_rows_exec_context(dout)");
    require_cuda_f32_row_major(
        dx, "softmax_backward_causal_rows_exec_context(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    CudaExecutionLane &lane = active_exec_lane();
    launch_softmax_backward_causal_rows_on_stream(softmax, dout, dx,
                                                  lane.stream);
  }

  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) override {
    require_cuda_f32_row_major(softmax,
                               "softmax_backward_causal_rows(softmax)");
    require_cuda_f32_row_major(dout, "softmax_backward_causal_rows(dout)");
    require_cuda_f32_row_major(dx, "softmax_backward_causal_rows(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    launch_softmax_backward_causal_rows(softmax, dout, dx);
  }

  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override {
    require_cuda_f32_row_major(scores, "apply_causal_mask_inplace");
    if (tensor_rows(scores) == 0 || tensor_cols(scores) == 0) {
      return;
    }
    launch_apply_causal_mask_inplace(scores, neg_inf);
  }

  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override {
    require_cuda_f32_row_major(params, "adamw_step(params)");
    require_cuda_f32_row_major(grads, "adamw_step(grads)");
    require_cuda_f32_row_major(m, "adamw_step(m)");
    require_cuda_f32_row_major(v, "adamw_step(v)");
    if (step < 1) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step step must be >= 1");
    }
    if (tensor_rows(params) != tensor_rows(grads) || tensor_cols(params) != tensor_cols(grads)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/grads shape mismatch");
    }
    if (tensor_rows(params) != tensor_rows(m) || tensor_cols(params) != tensor_cols(m)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/m shape mismatch");
    }
    if (tensor_rows(params) != tensor_rows(v) || tensor_cols(params) != tensor_cols(v)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/v shape mismatch");
    }
    if (learning_rate <= 0.0f) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step learning_rate must be > 0");
    }
    if (!(beta1 >= 0.0f && beta1 < 1.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step beta1 must be in [0,1)");
    }
    if (!(beta2 >= 0.0f && beta2 < 1.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step beta2 must be in [0,1)");
    }
    if (tensor_rows(params) == 0 || tensor_cols(params) == 0) {
      return;
    }

    const double t = static_cast<double>(step);
    const float b1_corr =
        1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), t));
    const float b2_corr =
        1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), t));
    if (!(b1_corr > 0.0f && b2_corr > 0.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step invalid bias correction");
    }

    launch_adamw_step(params, grads, m, v, learning_rate, beta1, beta2,
                      weight_decay, 1.0f / b1_corr, 1.0f / b2_corr,
                      apply_weight_decay ? 1u : 0u);
  }

  bool is_file2device_read_supported() const override { return true; }

  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override {
    if (size == 0) {
      return;
    }
    std::vector<uint8_t> host_buffer(static_cast<size_t>(size));
    cpu_backend_.read_file2device(path, host_buffer.data(), size, file_offset);
    copy_host2device(dst, host_buffer.data(), size);
  }

private:
  CudaExecutionLane &active_exec_lane() {
    if (!has_active_exec_group_) {
      throw std::runtime_error(
          "cuda_cublas_plugin: exec context op without active group");
    }
    return exec_lanes_[active_exec_lane_index_];
  }

  void ensure_exec_lanes() {
    if (exec_ready_event_ == nullptr) {
      check_cuda(cudaEventCreateWithFlags(&exec_ready_event_,
                                          cudaEventDisableTiming),
                 "cudaEventCreate(exec context ready)");
    }
    while (exec_lanes_.size() < max_exec_lanes_) {
      CudaExecutionLane lane;
      check_cuda(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking),
                 "cudaStreamCreate(exec context)");
      check_cublas(cublasCreate(&lane.cublas), "cublasCreate(exec context)");
      check_cublas(cublasSetMathMode(lane.cublas, CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode(exec context)");
      check_cublas(cublasSetStream(lane.cublas, lane.stream),
                   "cublasSetStream(exec context)");
      check_cuda(cudaEventCreateWithFlags(&lane.done, cudaEventDisableTiming),
                 "cudaEventCreate(exec context done)");
      exec_lanes_.push_back(lane);
    }
  }

  void copy_stage_from_device(const TensorView &src, HostTensorStage &dst) {
    copy_tensor_2d(src, dst.view, cudaMemcpyDeviceToHost,
                   "cudaMemcpy2D(device_to_host_stage)");
  }

  void copy_stage_to_device(const HostTensorStage &src, const TensorView &dst) {
    copy_tensor_2d(src.view, dst, cudaMemcpyHostToDevice,
                   "cudaMemcpy2D(host_stage_to_device)");
  }

  void fallback_cross_entropy_mean(const TensorView &logits,
                                   const TensorView &targets,
                                   TensorView &out_loss) {
    HostTensorStage logits_host(logits);
    HostTensorStage targets_host(targets);
    HostTensorStage out_loss_host(out_loss);

    copy_stage_from_device(logits, logits_host);
    copy_stage_from_device(targets, targets_host);

    cpu_backend_.cross_entropy_mean(logits_host.view, targets_host.view,
                                    out_loss_host.view);
    copy_stage_to_device(out_loss_host, out_loss);
  }

  void fallback_backward_from_logits_targets(TensorView &logits,
                                             const TensorView &targets) {
    HostTensorStage logits_host(logits);
    HostTensorStage targets_host(targets);

    copy_stage_from_device(logits, logits_host);
    copy_stage_from_device(targets, targets_host);

    cpu_backend_.backward_from_logits_targets(logits_host.view, targets_host.view);
    copy_stage_to_device(logits_host, logits);
  }

  cublasHandle_t handle_ = nullptr;
  DefaultCpuBackend cpu_backend_;
  static constexpr size_t max_exec_lanes_ = 4;
  cudaEvent_t exec_ready_event_ = nullptr;
  std::vector<CudaExecutionLane> exec_lanes_;
  size_t exec_lane_index_ = 0;
  size_t active_exec_lane_index_ = 0;
  size_t used_exec_lane_count_ = 0;
  bool has_active_exec_group_ = false;
  bool in_exec_iteration_ = false;
};



} // namespace

DeviceBackend *create_backend(uint32_t device) {
  if (static_cast<Device>(device) != Device::GPU) {
    throw std::runtime_error("cuda_cublas_plugin: only gpu device is supported");
  }
  return new CuBlasPluginBackend();
}

DeviceBackend &backend_from_opaque(void *backend) {
  if (backend == nullptr) {
    throw std::runtime_error("cuda_cublas_plugin: null backend instance");
  }
  return *reinterpret_cast<DeviceBackend *>(backend);
}

} // namespace cuda_cublas_plugin
