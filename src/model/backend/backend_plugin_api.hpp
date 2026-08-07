#pragma once

#include <cstdint>

inline constexpr uint32_t kBackendTensorMaxRank = 6;

struct BackendTensorView {
  uint32_t device;
  uint32_t dtype;
  void *data;
  uint32_t rank;
  int64_t rows;
  int64_t cols;
  int64_t stride_r_bytes;
  int64_t stride_c_bytes;
  int64_t dims[kBackendTensorMaxRank];
  int64_t strides_bytes[kBackendTensorMaxRank];
};

struct BackendMemoryInfo {
  uint32_t available;
  uint64_t free_bytes;
  uint64_t total_bytes;
};

struct BackendApiV1 {
  uint32_t abi_version;
  void *(*create)(uint32_t device);
  void (*destroy)(void *backend);
  uint32_t (*device)(void *backend);
  void *(*alloc)(void *backend, uint64_t bytes, uint32_t alignment);
  void (*free)(void *backend, void *ptr);
  void (*copy_host2device)(void *backend, void *dst, const void *src,
                           uint64_t bytes);
  void (*copy_device2host)(void *backend, void *dst, const void *src,
                           uint64_t bytes);
  void (*copy)(void *backend, const BackendTensorView *src,
               const BackendTensorView *dst);
  void (*fill)(void *backend, const BackendTensorView *t, float v);
  void (*add)(void *backend, const BackendTensorView *a,
              const BackendTensorView *b, const BackendTensorView *out);
  void (*add_inplace)(void *backend, const BackendTensorView *a,
                      const BackendTensorView *b);
  void (*add_bias_rowwise)(void *backend, const BackendTensorView *x,
                           const BackendTensorView *bias_1xC,
                           const BackendTensorView *out);
  void (*add_bias_relu_rowwise)(void *backend, const BackendTensorView *x,
                                const BackendTensorView *bias_1xC,
                                const BackendTensorView *out);
  void (*add_bias_relu_rowwise_inplace)(void *backend,
                                        const BackendTensorView *x,
                                        const BackendTensorView *bias_1xC);
  void (*mul_scalar)(void *backend, const BackendTensorView *x, float s,
                     const BackendTensorView *out);
  float (*sum_squares_f32)(void *backend, const BackendTensorView *x);
  void (*relu)(void *backend, const BackendTensorView *x,
               const BackendTensorView *out);
  void (*relu_backward)(void *backend, const BackendTensorView *preact,
                        const BackendTensorView *dout,
                        const BackendTensorView *dx);
  void (*relu_backward_inplace)(void *backend, const BackendTensorView *preact,
                                const BackendTensorView *dout_dx);
  void (*row_sum)(void *backend, const BackendTensorView *x,
                  const BackendTensorView *out_1xC);
  void (*matmul)(void *backend, const BackendTensorView *a,
                 const BackendTensorView *b, const BackendTensorView *out);
  void (*matmul_left_transposed)(void *backend, const BackendTensorView *a,
                                 const BackendTensorView *b,
                                 const BackendTensorView *out);
  void (*matmul_right_transposed)(void *backend, const BackendTensorView *a,
                                  const BackendTensorView *b,
                                  const BackendTensorView *out);
  void (*transpose)(void *backend, const BackendTensorView *x,
                    const BackendTensorView *out);
  void (*layernorm_forward)(void *backend, const BackendTensorView *x,
                            const BackendTensorView *gamma_1xC,
                            const BackendTensorView *beta_1xC,
                            const BackendTensorView *out);
  void (*layernorm_backward)(void *backend, const BackendTensorView *x,
                             const BackendTensorView *gamma_1xC,
                             const BackendTensorView *dout,
                             const BackendTensorView *dx,
                             const BackendTensorView *dgamma_1xC,
                             const BackendTensorView *dbeta_1xC);
  void (*embedding_lookup)(void *backend, const BackendTensorView *table,
                           const BackendTensorView *ids,
                           const BackendTensorView *out);
  void (*accumulate_embedding_grads)(void *backend,
                                     const BackendTensorView *ids,
                                     const BackendTensorView *d_cur,
                                     const BackendTensorView *d_tok,
                                     const BackendTensorView *d_pos);
  void (*cross_entropy_mean)(void *backend, const BackendTensorView *logits,
                             const BackendTensorView *targets,
                             const BackendTensorView *out_loss);
  void (*cross_entropy_mean_backward_inplace)(
      void *backend, const BackendTensorView *logits,
      const BackendTensorView *targets, const BackendTensorView *out_loss);
  float (*read_scalar_f32)(void *backend, const BackendTensorView *x);
  void (*backward_from_logits_targets)(void *backend,
                                       const BackendTensorView *logits,
                                       const BackendTensorView *targets);
  void (*softmax_rows)(void *backend, const BackendTensorView *x,
                       const BackendTensorView *out);
  void (*softmax_backward_rows)(void *backend,
                                const BackendTensorView *softmax,
                                const BackendTensorView *dout,
                                const BackendTensorView *dx);
  void (*scaled_causal_softmax_rows)(void *backend,
                                     const BackendTensorView *scores,
                                     float scale,
                                     const BackendTensorView *out);
  uint32_t (*supports_exec_context_iteration)(void *backend);
  void (*start_exec_context_iteration)(void *backend);
  void (*finish_exec_context_iteration)(void *backend);
  void (*start_exec_context_group)(void *backend);
  void (*finish_exec_context_group)(void *backend);
  void (*matmul_exec_context)(void *backend, const BackendTensorView *a,
                              const BackendTensorView *b,
                              const BackendTensorView *out);
  void (*matmul_left_transposed_exec_context)(
      void *backend, const BackendTensorView *a, const BackendTensorView *b,
      const BackendTensorView *out);
  void (*matmul_right_transposed_exec_context)(
      void *backend, const BackendTensorView *a, const BackendTensorView *b,
      const BackendTensorView *out);
  void (*scaled_causal_softmax_rows_exec_context)(
      void *backend, const BackendTensorView *scores, float scale,
      const BackendTensorView *out);
  void (*softmax_backward_causal_rows_exec_context)(
      void *backend, const BackendTensorView *softmax,
      const BackendTensorView *dout, const BackendTensorView *dx);
  void (*softmax_backward_causal_rows)(void *backend,
                                       const BackendTensorView *softmax,
                                       const BackendTensorView *dout,
                                       const BackendTensorView *dx);
  void (*apply_causal_mask_inplace)(void *backend,
                                    const BackendTensorView *scores,
                                    float neg_inf);
  void (*adamw_step)(void *backend, const BackendTensorView *params,
                     const BackendTensorView *grads,
                     const BackendTensorView *m,
                     const BackendTensorView *v, uint64_t step,
                     float learning_rate, float beta1, float beta2,
                     float weight_decay, uint32_t apply_weight_decay);
  uint32_t (*is_file2device_read_supported)(void *backend);
  void (*read_file2device)(void *backend, const char *path, void *dst,
                           uint64_t size, uint64_t file_offset);
  BackendMemoryInfo (*memory_info)(void *backend);
};

inline constexpr uint32_t kBackendApiVersion = 16;

extern "C" {
typedef const BackendApiV1 *(*BackendGetApiFn)();
}
