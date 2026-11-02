#pragma once

#include <cstdint>

class DeviceBackend;

enum class OpId : uint32_t {
  Alloc,
  Free,
  CopyHostToDevice,
  CopyDeviceToHost,
  Copy,
  Cast,
  Fill,
  Add,
  AddInplace,
  AddBiasRowwise,
  AddBiasReluRowwise,
  AddBiasReluRowwiseInplace,
  MulScalar,
  SumSquaresF32,
  Relu,
  ReluBackward,
  ReluBackwardInplace,
  RowSum,
  Gemm,
  GemmRankedMatrixRhs,
  GemmRankedMatrixRhsT,
  GemmRankedReduceLhsT,
  GemmBatched,
  GemmBatchedLhsT,
  GemmBatchedRhsT,
  Transpose,
  LayernormForward,
  LayernormBackward,
  EmbeddingLookup,
  AccumulateEmbeddingGrads,
  CrossEntropyMean,
  CrossEntropyMeanBackwardInplace,
  ReadScalarF32,
  BackwardFromLogitsTargets,
  SoftmaxRows,
  SoftmaxBackwardRows,
  ScaledCausalSoftmaxRows,
  GemmBatchedExecContext,
  GemmBatchedLhsTExecContext,
  GemmBatchedRhsTExecContext,
  ScaledCausalSoftmaxRowsExecContext,
  SoftmaxBackwardCausalRowsExecContext,
  SoftmaxBackwardCausalRows,
  ApplyCausalMaskInplace,
  RotaryEmbeddingInplace,
  AdamwStep,
  ReadFileToDevice,
  Count,
};

const char *op_name(OpId op);

class IBackendObserver {
public:
  virtual ~IBackendObserver() = default;
  virtual void on_backend_ready(DeviceBackend &backend) { (void)backend; }
  virtual void on_op_begin(OpId op) = 0;
  virtual void on_op_end(OpId op) = 0;
};
