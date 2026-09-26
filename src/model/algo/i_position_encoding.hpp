#pragma once

#include "tensor.hpp"

class GradientStore;
class Ops;
class ReportSink;
class TensorStore;
class TrainingDiagnosticsController;

// Single source of truth for how the model encodes token positions.
//
// Modules that own memory, checkpoints and reporting ask the query methods and
// decide themselves what to do. The Transformer calls the compute hooks
// unconditionally; an encoding that has nothing to do there is a no-op.
class IPositionEncoding {
public:
  virtual ~IPositionEncoding() = default;

  virtual const char *name() const = 0;

  // Whether the parameter arena must hold pos_embedding [max_seq_len, D].
  virtual bool needs_position_table() const = 0;

  // Called once by the Transformer after memory is allocated.
  // gradient_store is null for inference.
  virtual void bind(TensorStore &tensor_store, GradientStore *gradient_store,
                    Ops &ops) = 0;
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;

  // x: [B, S, D] token embeddings, updated in place.
  virtual void apply_to_input(TensorView &x) = 0;
  // dx: [B, S, D] gradient reaching the input embeddings.
  virtual void backward_input(const TensorView &dx) = 0;

  virtual void report_probes(ReportSink &sink) const = 0;
};
