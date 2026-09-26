#pragma once

#include "i_position_encoding.hpp"

#include <config.hpp>

// X += pos_embedding[0..S), broadcast over batch.
// Parameter: pos_embedding [max_seq_len, D].
class LearnedPositionEncoding : public IPositionEncoding {
public:
  explicit LearnedPositionEncoding(const Config &cfg);

  const char *name() const override { return "learned"; }
  bool needs_position_table() const override { return true; }

  void bind(TensorStore &tensor_store, GradientStore *gradient_store,
            Ops &ops) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void apply_to_input(TensorView &x) override;
  void backward_input(const TensorView &dx) override;

  void report_probes(ReportSink &sink) const override;

private:
  const Config &cfg_;
  Ops *ops_ = nullptr;
  TensorView pos_emb_;
  TensorView d_pos_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
};

// No explicit position signal; causal attention alone orders tokens.
class NoPositionEncoding : public IPositionEncoding {
public:
  const char *name() const override { return "none"; }
  bool needs_position_table() const override { return false; }

  void bind(TensorStore &, GradientStore *, Ops &) override {}
  void set_diagnostics(TrainingDiagnosticsController *) override {}

  void apply_to_input(TensorView &) override {}
  void backward_input(const TensorView &) override {}

  void report_probes(ReportSink &) const override {}
};
