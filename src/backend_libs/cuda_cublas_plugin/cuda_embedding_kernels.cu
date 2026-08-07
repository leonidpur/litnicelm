#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

namespace cuda_cublas_plugin {
namespace {

__global__ void embedding_lookup_kernel(KernelTensorView table,
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
  const int64_t token = load_index(ids, r, 0);
  if (token < 0 || token >= table.rows) {
    store_f32(out, r, c, 0.0f);
    return;
  }
  store_f32(out, r, c, load_f32(table, token, c));
}

__global__ void accumulate_embedding_grads_kernel(KernelTensorView ids,
                                                  KernelTensorView d_cur,
                                                  KernelTensorView d_tok,
                                                  KernelTensorView d_pos,
                                                  int64_t seq_len) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = d_cur.rows * d_cur.cols;
  if (idx >= total) {
    return;
  }

  const int64_t t = idx / d_cur.cols;
  const int64_t d = idx % d_cur.cols;
  const int64_t token = load_index(ids, t, 0);
  if (token < 0 || token >= d_tok.rows) {
    return;
  }

  const float g = load_f32(d_cur, t, d);
  const int64_t seq_pos = seq_len > 0 ? t % seq_len : t;
  atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(d_tok, token, d)), g);
  atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(d_pos, seq_pos, d)), g);
}

} // namespace

void launch_embedding_lookup(const TensorView &table, const TensorView &ids,
                             TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  embedding_lookup_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(table),
                          to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(out));
  check_kernel_launch("embedding_lookup_kernel");
}

void launch_accumulate_embedding_grads(const TensorView &ids,
                                       const TensorView &d_cur,
                                       TensorView &d_tok, TensorView &d_pos,
                                       int64_t seq_len) {
  const int64_t total = tensor_rows(d_cur) * tensor_cols(d_cur);
  accumulate_embedding_grads_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_index_vector_view(ids),
                          to_kernel_tensor_view(d_cur),
                          to_kernel_tensor_view(d_tok),
                          to_kernel_tensor_view(d_pos), seq_len);
  check_kernel_launch("accumulate_embedding_grads_kernel");
}

} // namespace cuda_cublas_plugin
