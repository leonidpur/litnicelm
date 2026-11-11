// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "backend_observer.hpp"

const char *op_name(OpId op) {
  switch (op) {
  case OpId::Alloc: return "alloc";
  case OpId::Free: return "free";
  case OpId::CopyHostToDevice: return "copy_host2device";
  case OpId::CopyDeviceToHost: return "copy_device2host";
  case OpId::Copy: return "copy";
  case OpId::Cast: return "cast";
  case OpId::Fill: return "fill";
  case OpId::Add: return "add";
  case OpId::AddInplace: return "add_inplace";
  case OpId::AddBiasRowwise: return "add_bias_rowwise";
  case OpId::AddBiasReluRowwise: return "add_bias_relu_rowwise";
  case OpId::AddBiasReluRowwiseInplace: return "add_bias_relu_rowwise_inplace";
  case OpId::MulScalar: return "mul_scalar";
  case OpId::SumSquaresF32: return "sum_squares_f32";
  case OpId::Relu: return "relu";
  case OpId::ReluBackward: return "relu_backward";
  case OpId::ReluBackwardInplace: return "relu_backward_inplace";
  case OpId::RowSum: return "row_sum";
  case OpId::Gemm: return "gemm";
  case OpId::GemmRankedMatrixRhs: return "gemm_ranked_matrix_rhs";
  case OpId::GemmRankedMatrixRhsT: return "gemm_ranked_matrix_rhs_t";
  case OpId::GemmRankedReduceLhsT: return "gemm_ranked_reduce_lhs_t";
  case OpId::GemmBatched: return "gemm_batched";
  case OpId::GemmBatchedLhsT: return "gemm_batched_lhs_t";
  case OpId::GemmBatchedRhsT: return "gemm_batched_rhs_t";
  case OpId::Transpose: return "transpose";
  case OpId::LayernormForward: return "layernorm_forward";
  case OpId::LayernormBackward: return "layernorm_backward";
  case OpId::EmbeddingLookup: return "embedding_lookup";
  case OpId::AccumulateEmbeddingGrads: return "accumulate_embedding_grads";
  case OpId::CrossEntropyMean: return "cross_entropy_mean";
  case OpId::CrossEntropyMeanBackwardInplace: return "cross_entropy_mean_backward_inplace";
  case OpId::ReadScalarF32: return "read_scalar_f32";
  case OpId::BackwardFromLogitsTargets: return "backward_from_logits_targets";
  case OpId::SoftmaxRows: return "softmax_rows";
  case OpId::SoftmaxBackwardRows: return "softmax_backward_rows";
  case OpId::ScaledCausalSoftmaxRows: return "scaled_causal_softmax_rows";
  case OpId::GemmBatchedExecContext: return "gemm_batched_exec_context";
  case OpId::GemmBatchedLhsTExecContext: return "gemm_batched_lhs_t_exec_context";
  case OpId::GemmBatchedRhsTExecContext: return "gemm_batched_rhs_t_exec_context";
  case OpId::ScaledCausalSoftmaxRowsExecContext: return "scaled_causal_softmax_rows_exec_context";
  case OpId::SoftmaxBackwardCausalRowsExecContext: return "softmax_backward_causal_rows_exec_context";
  case OpId::SoftmaxBackwardCausalRows: return "softmax_backward_causal_rows";
  case OpId::ApplyCausalMaskInplace: return "apply_causal_mask_inplace";
  case OpId::RotaryEmbeddingInplace: return "rotary_embedding_inplace";
  case OpId::CausalAttentionForward: return "causal_attention_forward";
  case OpId::CausalAttentionBackward: return "causal_attention_backward";
  case OpId::LinearForward: return "linear_forward";
  case OpId::LinearBackwardParams: return "linear_backward_params";
  case OpId::AdamwStep: return "adamw_step";
  case OpId::ReadFileToDevice: return "read_file2device";
  case OpId::Count: return "count";
  }
  return "unknown";
}
