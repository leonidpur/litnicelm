#pragma once

#include <cstdint>

struct BackendTensorView {
  uint32_t device;
  uint32_t dtype;
  void *data;
  int64_t rows;
  int64_t cols;
  int64_t stride_r_bytes;
  int64_t stride_c_bytes;
};

struct BackendApiV1 {
  uint32_t abi_version;
  void *(*create)(uint32_t device);
  void (*destroy)(void *backend);
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
  void (*mul_scalar)(void *backend, const BackendTensorView *x, float s,
                     const BackendTensorView *out);
  void (*relu)(void *backend, const BackendTensorView *x,
               const BackendTensorView *out);
  void (*matmul)(void *backend, const BackendTensorView *a,
                 const BackendTensorView *b, const BackendTensorView *out);
  void (*matmul_transposed)(void *backend, const BackendTensorView *a,
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
  void (*cross_entropy_mean)(void *backend, const BackendTensorView *logits,
                             const BackendTensorView *targets,
                             const BackendTensorView *out_loss);
  float (*read_scalar_f32)(void *backend, const BackendTensorView *x);
  void (*backward_from_logits_targets)(void *backend,
                                       const BackendTensorView *logits,
                                       const BackendTensorView *targets);
  void (*softmax_rows)(void *backend, const BackendTensorView *x,
                       const BackendTensorView *out);
  void (*apply_causal_mask_inplace)(void *backend,
                                    const BackendTensorView *scores,
                                    float neg_inf);
  uint32_t (*is_file2device_read_supported)(void *backend);
  void (*read_file2device)(void *backend, const char *path, void *dst,
                           uint64_t size, uint64_t file_offset);
};

inline constexpr uint32_t kBackendApiVersion = 1;

extern "C" {
typedef const BackendApiV1 *(*BackendGetApiFn)();
}
