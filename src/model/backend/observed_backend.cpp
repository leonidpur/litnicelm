// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "observed_backend.hpp"

#include <stdexcept>
#include <utility>

namespace {

class ObserverScope {
public:
  ObserverScope(const std::vector<IBackendObserver *> &observers, OpId op)
      : observers_(observers), op_(op) {
    for (IBackendObserver *observer : observers_) {
      observer->on_op_begin(op_);
    }
  }

  ~ObserverScope() {
    for (auto it = observers_.rbegin(); it != observers_.rend(); ++it) {
      (*it)->on_op_end(op_);
    }
  }

private:
  const std::vector<IBackendObserver *> &observers_;
  OpId op_;
};

} // namespace

ObservedBackend::ObservedBackend(std::unique_ptr<DeviceBackend> wrapped,
                                 std::vector<IBackendObserver *> observers)
    : wrapped_(std::move(wrapped)), observers_(std::move(observers)) {
  if (!wrapped_) throw std::invalid_argument("ObservedBackend: null backend");
  if (observers_.empty()) throw std::invalid_argument("ObservedBackend: no observers");
  for (IBackendObserver *observer : observers_) {
    if (observer == nullptr) throw std::invalid_argument("ObservedBackend: null observer");
  }
}

DeviceBackend &ObservedBackend::wrapped_backend() { return *wrapped_; }
Device ObservedBackend::device() const { return wrapped_->device(); }
DeviceMemoryInfo ObservedBackend::memory_info() const { return wrapped_->memory_info(); }
uint32_t ObservedBackend::precision_variants() const { return wrapped_->precision_variants(); }
bool ObservedBackend::supports_exec_context_iteration() const {
  return wrapped_->supports_exec_context_iteration();
}
bool ObservedBackend::is_file2device_read_supported() const {
  return wrapped_->is_file2device_read_supported();
}

void *ObservedBackend::alloc(uint64_t bytes, uint32_t alignment) {
  ObserverScope observed(observers_, OpId::Alloc);
  return wrapped_->alloc(bytes, alignment);
}
void ObservedBackend::free(void *ptr) {
  ObserverScope observed(observers_, OpId::Free);
  wrapped_->free(ptr);
}
void ObservedBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  ObserverScope observed(observers_, OpId::CopyHostToDevice);
  wrapped_->copy_host2device(dst, src, bytes);
}
void ObservedBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  ObserverScope observed(observers_, OpId::CopyDeviceToHost);
  wrapped_->copy_device2host(dst, src, bytes);
}
void ObservedBackend::copy(const TensorView &src, TensorView &dst) {
  ObserverScope observed(observers_, OpId::Copy); wrapped_->copy(src, dst);
}
void ObservedBackend::cast(const TensorView &src, TensorView &dst) {
  ObserverScope observed(observers_, OpId::Cast); wrapped_->cast(src, dst);
}
void ObservedBackend::fill(TensorView &t, float v) {
  ObserverScope observed(observers_, OpId::Fill); wrapped_->fill(t, v);
}
void ObservedBackend::add(const TensorView &a, const TensorView &b, TensorView &out) {
  ObserverScope observed(observers_, OpId::Add); wrapped_->add(a, b, out);
}
void ObservedBackend::add_inplace(TensorView &a, const TensorView &b) {
  ObserverScope observed(observers_, OpId::AddInplace); wrapped_->add_inplace(a, b);
}
void ObservedBackend::add_bias_rowwise(const TensorView &x, const TensorView &bias,
                                       TensorView &out) {
  ObserverScope observed(observers_, OpId::AddBiasRowwise);
  wrapped_->add_bias_rowwise(x, bias, out);
}
void ObservedBackend::add_bias_relu_rowwise(const TensorView &x,
                                            const TensorView &bias,
                                            TensorView &out) {
  ObserverScope observed(observers_, OpId::AddBiasReluRowwise);
  wrapped_->add_bias_relu_rowwise(x, bias, out);
}
void ObservedBackend::add_bias_relu_rowwise_inplace(TensorView &x,
                                                    const TensorView &bias) {
  ObserverScope observed(observers_, OpId::AddBiasReluRowwiseInplace);
  wrapped_->add_bias_relu_rowwise_inplace(x, bias);
}
void ObservedBackend::mul_scalar(const TensorView &x, float s, TensorView &out) {
  ObserverScope observed(observers_, OpId::MulScalar); wrapped_->mul_scalar(x, s, out);
}
float ObservedBackend::sum_squares_f32(const TensorView &x) {
  ObserverScope observed(observers_, OpId::SumSquaresF32);
  return wrapped_->sum_squares_f32(x);
}
void ObservedBackend::relu(const TensorView &x, TensorView &out) {
  ObserverScope observed(observers_, OpId::Relu); wrapped_->relu(x, out);
}
void ObservedBackend::relu_backward(const TensorView &preact, const TensorView &dout,
                                    TensorView &dx) {
  ObserverScope observed(observers_, OpId::ReluBackward);
  wrapped_->relu_backward(preact, dout, dx);
}
void ObservedBackend::relu_backward_inplace(const TensorView &preact,
                                            TensorView &dout_dx) {
  ObserverScope observed(observers_, OpId::ReluBackwardInplace);
  wrapped_->relu_backward_inplace(preact, dout_dx);
}
void ObservedBackend::row_sum(const TensorView &x, TensorView &out) {
  ObserverScope observed(observers_, OpId::RowSum); wrapped_->row_sum(x, out);
}
void ObservedBackend::gemm(const TensorView &a, const TensorView &b, TensorView &out) {
  ObserverScope observed(observers_, OpId::Gemm); wrapped_->gemm(a, b, out);
}
void ObservedBackend::gemm_ranked_matrix_rhs(const TensorView &a,
                                             const TensorView &b, TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmRankedMatrixRhs);
  wrapped_->gemm_ranked_matrix_rhs(a, b, out);
}
void ObservedBackend::gemm_ranked_matrix_rhs_t(const TensorView &a,
                                               const TensorView &b, TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmRankedMatrixRhsT);
  wrapped_->gemm_ranked_matrix_rhs_t(a, b, out);
}
void ObservedBackend::gemm_ranked_reduce_lhs_t(const TensorView &a,
                                               const TensorView &b, TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmRankedReduceLhsT);
  wrapped_->gemm_ranked_reduce_lhs_t(a, b, out);
}
void ObservedBackend::gemm_batched(const TensorView &a, const TensorView &b,
                                   TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatched); wrapped_->gemm_batched(a, b, out);
}
void ObservedBackend::gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                                         TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatchedLhsT);
  wrapped_->gemm_batched_lhs_t(a, b, out);
}
void ObservedBackend::gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                                         TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatchedRhsT);
  wrapped_->gemm_batched_rhs_t(a, b, out);
}
void ObservedBackend::transpose(const TensorView &x, TensorView &out) {
  ObserverScope observed(observers_, OpId::Transpose); wrapped_->transpose(x, out);
}
void ObservedBackend::layernorm_forward(const TensorView &x, const TensorView &gamma,
                                        const TensorView &beta, TensorView &out) {
  ObserverScope observed(observers_, OpId::LayernormForward);
  wrapped_->layernorm_forward(x, gamma, beta, out);
}
void ObservedBackend::layernorm_backward(const TensorView &x, const TensorView &gamma,
                                         const TensorView &dout, TensorView &dx,
                                         TensorView &dgamma, TensorView &dbeta) {
  ObserverScope observed(observers_, OpId::LayernormBackward);
  wrapped_->layernorm_backward(x, gamma, dout, dx, dgamma, dbeta);
}
void ObservedBackend::embedding_lookup(const TensorView &table, const TensorView &ids,
                                       TensorView &out) {
  ObserverScope observed(observers_, OpId::EmbeddingLookup);
  wrapped_->embedding_lookup(table, ids, out);
}
void ObservedBackend::accumulate_embedding_grads(const TensorView &ids,
                                                 const TensorView &d_cur,
                                                 TensorView &d_tok) {
  ObserverScope observed(observers_, OpId::AccumulateEmbeddingGrads);
  wrapped_->accumulate_embedding_grads(ids, d_cur, d_tok);
}
void ObservedBackend::cross_entropy_mean(const TensorView &logits,
                                         const TensorView &targets,
                                         TensorView &out_loss) {
  ObserverScope observed(observers_, OpId::CrossEntropyMean);
  wrapped_->cross_entropy_mean(logits, targets, out_loss);
}
void ObservedBackend::cross_entropy_mean_backward_inplace(TensorView &logits,
                                                          const TensorView &targets,
                                                          TensorView &out_loss) {
  ObserverScope observed(observers_, OpId::CrossEntropyMeanBackwardInplace);
  wrapped_->cross_entropy_mean_backward_inplace(logits, targets, out_loss);
}
float ObservedBackend::read_scalar_f32(const TensorView &x) {
  ObserverScope observed(observers_, OpId::ReadScalarF32);
  return wrapped_->read_scalar_f32(x);
}
void ObservedBackend::backward_from_logits_targets(TensorView &logits,
                                                   const TensorView &targets) {
  ObserverScope observed(observers_, OpId::BackwardFromLogitsTargets);
  wrapped_->backward_from_logits_targets(logits, targets);
}
void ObservedBackend::softmax_rows(const TensorView &x, TensorView &out) {
  ObserverScope observed(observers_, OpId::SoftmaxRows); wrapped_->softmax_rows(x, out);
}
void ObservedBackend::softmax_backward_rows(const TensorView &softmax,
                                            const TensorView &dout, TensorView &dx) {
  ObserverScope observed(observers_, OpId::SoftmaxBackwardRows);
  wrapped_->softmax_backward_rows(softmax, dout, dx);
}
void ObservedBackend::scaled_causal_softmax_rows(const TensorView &scores,
                                                 float scale, TensorView &out) {
  ObserverScope observed(observers_, OpId::ScaledCausalSoftmaxRows);
  wrapped_->scaled_causal_softmax_rows(scores, scale, out);
}
void ObservedBackend::start_exec_context_iteration() {
  wrapped_->start_exec_context_iteration();
}
void ObservedBackend::finish_exec_context_iteration() {
  wrapped_->finish_exec_context_iteration();
}
void ObservedBackend::start_exec_context_group() { wrapped_->start_exec_context_group(); }
void ObservedBackend::finish_exec_context_group() { wrapped_->finish_exec_context_group(); }
void ObservedBackend::gemm_batched_exec_context(const TensorView &a,
                                                const TensorView &b,
                                                TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatchedExecContext);
  wrapped_->gemm_batched_exec_context(a, b, out);
}
void ObservedBackend::gemm_batched_lhs_t_exec_context(const TensorView &a,
                                                      const TensorView &b,
                                                      TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatchedLhsTExecContext);
  wrapped_->gemm_batched_lhs_t_exec_context(a, b, out);
}
void ObservedBackend::gemm_batched_rhs_t_exec_context(const TensorView &a,
                                                      const TensorView &b,
                                                      TensorView &out) {
  ObserverScope observed(observers_, OpId::GemmBatchedRhsTExecContext);
  wrapped_->gemm_batched_rhs_t_exec_context(a, b, out);
}
void ObservedBackend::scaled_causal_softmax_rows_exec_context(
    const TensorView &scores, float scale, TensorView &out) {
  ObserverScope observed(observers_, OpId::ScaledCausalSoftmaxRowsExecContext);
  wrapped_->scaled_causal_softmax_rows_exec_context(scores, scale, out);
}
void ObservedBackend::softmax_backward_causal_rows_exec_context(
    const TensorView &softmax, const TensorView &dout, TensorView &dx) {
  ObserverScope observed(observers_, OpId::SoftmaxBackwardCausalRowsExecContext);
  wrapped_->softmax_backward_causal_rows_exec_context(softmax, dout, dx);
}
void ObservedBackend::softmax_backward_causal_rows(const TensorView &softmax,
                                                   const TensorView &dout,
                                                   TensorView &dx) {
  ObserverScope observed(observers_, OpId::SoftmaxBackwardCausalRows);
  wrapped_->softmax_backward_causal_rows(softmax, dout, dx);
}
void ObservedBackend::apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  ObserverScope observed(observers_, OpId::ApplyCausalMaskInplace);
  wrapped_->apply_causal_mask_inplace(scores, neg_inf);
}
void ObservedBackend::rotary_embedding_inplace(TensorView &x, int64_t n_heads,
                                               float base, bool inverse) {
  ObserverScope observed(observers_, OpId::RotaryEmbeddingInplace);
  wrapped_->rotary_embedding_inplace(x, n_heads, base, inverse);
}
void ObservedBackend::adamw_step(TensorView &params, const TensorView &grads,
                                 TensorView &m, TensorView &v, uint64_t step,
                                 float learning_rate, float beta1, float beta2,
                                 float weight_decay, bool apply_weight_decay) {
  ObserverScope observed(observers_, OpId::AdamwStep);
  wrapped_->adamw_step(params, grads, m, v, step, learning_rate, beta1, beta2,
                       weight_decay, apply_weight_decay);
}
void ObservedBackend::read_file2device(const std::string &path, void *dst,
                                       uint64_t size, uint64_t offset) {
  ObserverScope observed(observers_, OpId::ReadFileToDevice);
  wrapped_->read_file2device(path, dst, size, offset);
}
void ObservedBackend::timing_mark_begin(uint32_t op_id) {
  wrapped_->timing_mark_begin(op_id);
}
void ObservedBackend::timing_mark_end() { wrapped_->timing_mark_end(); }
BackendTimingReport ObservedBackend::timing_report() {
  return wrapped_->timing_report();
}
void ObservedBackend::timing_prepare(uint64_t max_marks) {
  wrapped_->timing_prepare(max_marks);
}
