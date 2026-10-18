#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cuda_bf16.h>

namespace cuda_cublas_plugin {
namespace {

__global__ void embedding_lookup_f32(KernelTensorView table,
                                    KernelTensorView ids,
                                    KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const int64_t token = reinterpret_cast<const int32_t *>(row_ptr(ids, r))[0];
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  if (token < 0 || token >= table.rows) {
    out_row[c] = 0.0f;
    return;
  }
  out_row[c] = reinterpret_cast<const float *>(row_ptr(table, token))[c];
}

__global__ void accumulate_embedding_grads_f32(KernelTensorView ids,
                                               KernelTensorView d_cur,
                                               KernelTensorView d_tok) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = d_cur.rows * d_cur.cols;
  if (idx >= total) {
    return;
  }

  const int64_t t = idx / d_cur.cols;
  const int64_t d = idx % d_cur.cols;
  const int64_t token = reinterpret_cast<const int32_t *>(row_ptr(ids, t))[0];
  if (token < 0 || token >= d_tok.rows) {
    return;
  }

  const float g = reinterpret_cast<const float *>(row_ptr(d_cur, t))[d];
  atomicAdd(&reinterpret_cast<float *>(row_ptr_mut(d_tok, token))[d], g);
}

// _bf16act: the F32 table gathered into BF16 activations; BF16 activation
// gradients scattered into the F32 table gradient.
__global__ void embedding_lookup_bf16act(KernelTensorView table,
                                         KernelTensorView ids,
                                         KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const int64_t token = reinterpret_cast<const int32_t *>(row_ptr(ids, r))[0];
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  if (token < 0 || token >= table.rows) {
    out_row[c] = __float2bfloat16(0.0f);
    return;
  }
  out_row[c] = __float2bfloat16(
      reinterpret_cast<const float *>(row_ptr(table, token))[c]);
}

__global__ void accumulate_embedding_grads_bf16act(KernelTensorView ids,
                                                   KernelTensorView d_cur,
                                                   KernelTensorView d_tok) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = d_cur.rows * d_cur.cols;
  if (idx >= total) {
    return;
  }

  const int64_t t = idx / d_cur.cols;
  const int64_t d = idx % d_cur.cols;
  const int64_t token = reinterpret_cast<const int32_t *>(row_ptr(ids, t))[0];
  if (token < 0 || token >= d_tok.rows) {
    return;
  }

  const float g = __bfloat162float(
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(d_cur, t))[d]);
  atomicAdd(&reinterpret_cast<float *>(row_ptr_mut(d_tok, token))[d], g);
}

} // namespace

void launch_embedding_lookup_f32(const TensorView &table,
                                    const TensorView &ids, TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  embedding_lookup_f32<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(table),
                          to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(out));
  check_kernel_launch("embedding_lookup_f32");
}

void launch_accumulate_embedding_grads_f32(const TensorView &ids,
                                              const TensorView &d_cur,
                                              TensorView &d_tok) {
  const int64_t total = tensor_rows(d_cur) * tensor_cols(d_cur);
  accumulate_embedding_grads_f32<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(d_cur),
                          to_kernel_tensor_view(d_tok));
  check_kernel_launch("accumulate_embedding_grads_f32");
}

void launch_embedding_lookup_bf16act(const TensorView &table,
                                     const TensorView &ids, TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  embedding_lookup_bf16act<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(table),
                          to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(out));
  check_kernel_launch("embedding_lookup_bf16act");
}

void launch_accumulate_embedding_grads_bf16act(const TensorView &ids,
                                                  const TensorView &d_cur,
                                                  TensorView &d_tok) {
  const int64_t total = tensor_rows(d_cur) * tensor_cols(d_cur);
  accumulate_embedding_grads_bf16act<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(d_cur),
                          to_kernel_tensor_view(d_tok));
  check_kernel_launch("accumulate_embedding_grads_bf16act");
}

} // namespace cuda_cublas_plugin
