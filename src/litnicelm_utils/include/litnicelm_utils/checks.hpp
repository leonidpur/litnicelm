#pragma once

// Checks on the program's inputs and internal agreements. Tensor checks take
// the ABI form, TensorViewStruct, so any view can be checked: a TensorView
// is one. One choke point: every failing check goes through
// Checker::report(), which logs a check_failed event and throws. Each check belongs to a category, which a
// future policy will use to decide how a failure is handled.

#include "litnicelm_utils/device.hpp"
#include "litnicelm_utils/dtype.hpp"
#include "litnicelm_utils/layout_slice.hpp"
#include "litnicelm_utils/shape.hpp"
#include "litnicelm_utils/tensor_view_struct.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

enum class CheckKind : uint8_t {
  Layout,      // memory safety: bounds, alignment, slices, null pointers
  Capability,  // what an implementation supports: device, dtype, contiguity
  Contract,    // agreements between tensors and arguments: shapes, dtypes, ranges
  Numeric,     // finiteness and sign of computed values
  Count
};

const char *check_name(CheckKind category);

// How the leading (batch) dims of a matrix product relate; the product
// itself is over the last two dims.
enum class MatmulKind : uint8_t {
  Plain,         // a, b, out are matrices
  BroadcastRhs,  // a and out [..., m, ·], b one matrix used for every batch
  Batched,       // a, b, out share their leading dims
  ReducePrefix,  // a and b share their leading dims, summed into matrix out
};

// Which ends of an interval belong to it.
enum class Interval : uint8_t { Closed, ClosedOpen, Open };

class Checker {
public:
  // --- any condition that fits no named check; the message is built only
  // on failure.
  template <typename MessageFn>
  void check(CheckKind category, bool ok, std::string_view who, MessageFn &&message) const {
    if (!ok) [[unlikely]] {
      report(category, who, std::string(message()));
    }
  }
  void check(CheckKind category, bool ok, std::string_view who, const char *message) const;

  // --- Layout
  void check_not_null(const void *ptr, std::string_view who, std::string_view what) const;
  // offset + bytes <= limit, without overflow.
  void check_in_bounds(uint64_t offset, uint64_t bytes, uint64_t limit,
                       std::string_view who, std::string_view what) const;
  // value is a multiple of alignment.
  void check_aligned(uint64_t value, uint64_t alignment, std::string_view who,
                     std::string_view what) const;
  // 0 <= index < size.
  void check_index(int64_t index, uint64_t size, std::string_view who,
                   std::string_view what) const;
  // The slice holds exactly `shape` of its dtype, aligned, inside the arena.
  void check_slice(const LayoutSlice &slice, const Shape &shape, uint64_t arena_bytes,
                   std::string_view who) const;
  // Each dim of `prefix` fits in the corresponding leading dim of `view`.
  void check_prefix(const TensorViewStruct &view, const Shape &prefix, std::string_view who) const;

  // --- Capability
  void check_supported(const TensorViewStruct &view, Device device, DType dtype, bool contiguous,
                       std::string_view who) const;

  // --- Contract
  void check_same_shape(const TensorViewStruct &a, const TensorViewStruct &b, std::string_view who) const;
  void check_same_device_dtype(const TensorViewStruct &a, const TensorViewStruct &b,
                               std::string_view who) const;
  void check_rank(const TensorViewStruct &view, size_t rank, std::string_view who) const;
  // The view's dims; -1 accepts any value at that position.
  void check_dims(const TensorViewStruct &view, std::initializer_list<int64_t> dims,
                  std::string_view who) const;
  // A [batch, seq, last_dim] tensor; -1 accepts any value at that position.
  void check_bsd(const TensorViewStruct &view, int64_t batch, int64_t seq, int64_t last_dim,
                 std::string_view who) const;
  // out = op(a) x op(b) over the last two dims; op transposes when asked.
  void check_matmul(const TensorViewStruct &a, const TensorViewStruct &b, const TensorViewStruct &out,
                    MatmulKind kind, bool transpose_a, bool transpose_b,
                    std::string_view who) const;
  void check_dtype_one_of(const TensorViewStruct &view, std::initializer_list<DType> dtypes,
                          std::string_view who) const;
  void check_range(double value, double lo, double hi, Interval interval,
                   std::string_view who, std::string_view what) const;
  void check_positive(double value, std::string_view who, std::string_view what) const;

  // --- Numeric
  void check_finite(double value, std::string_view who, std::string_view what) const;

private:
  // The one place a failure goes: logs it, then throws std::runtime_error
  // with "<who>: <message>".
  void report(CheckKind category, std::string_view who, const std::string &message) const;
};

// The process-wide checker. Until set_checker() is called it is a default
// Checker.
Checker &checker();
// Installs checker, which must stay alive while installed.
void set_checker(Checker &checker);

// Installs a checker for the lifetime of the scope, then restores the
// previous one. For tests.
class ScopedChecker {
public:
  explicit ScopedChecker(Checker &checker);
  ~ScopedChecker();
  ScopedChecker(const ScopedChecker &) = delete;
  ScopedChecker &operator=(const ScopedChecker &) = delete;

private:
  Checker &previous_;
};

// "[8, 16, 29]".
std::string shape_text(const Shape &shape);
