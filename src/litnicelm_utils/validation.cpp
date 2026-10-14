#include "litnicelm_utils/validation.hpp"

#include "litnicelm_utils/log.hpp"
#include "litnicelm_utils_events.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

Validator *&installed_validator() {
  static Validator fallback;
  static Validator *installed = &fallback;
  return installed;
}

std::string text(std::string_view s) { return std::string(s); }

Device device_of(const TensorViewStruct &v) { return static_cast<Device>(v.device_id); }
DType dtype_of(const TensorViewStruct &v) { return static_cast<DType>(v.dtype_id); }
Shape shape_of(const TensorViewStruct &v) { return Shape(v.ndim, v.dims); }

// Row-major contiguous: each stride is the element size times the dims after it.
bool is_row_major_contiguous(const TensorViewStruct &v) {
  int64_t expected = static_cast<int64_t>(dtype_size(dtype_of(v)));
  for (size_t i = v.ndim; i > 0; --i) {
    if (v.strides_bytes[i - 1] != expected) {
      return false;
    }
    expected *= v.dims[i - 1];
  }
  return true;
}

std::string view_text(const TensorViewStruct &v) {
  return shape_text(shape_of(v)) + " " + dtype_name(dtype_of(v)) + " " +
         device_name(device_of(v));
}

std::string dims_text(std::initializer_list<int64_t> dims) {
  std::ostringstream oss;
  oss << "[";
  size_t i = 0;
  for (int64_t d : dims) {
    oss << (i++ ? ", " : "") << (d < 0 ? std::string("*") : std::to_string(d));
  }
  oss << "]";
  return oss.str();
}

bool dims_match(const TensorViewStruct &view, std::initializer_list<int64_t> dims) {
  if (view.ndim != dims.size()) {
    return false;
  }
  size_t i = 0;
  for (int64_t d : dims) {
    if (d >= 0 && view.dims[i] != d) {
      return false;
    }
    ++i;
  }
  return true;
}

// Leading dims (all but the last two) of a and b agree.
bool same_prefix(const TensorViewStruct &a, const TensorViewStruct &b) {
  if (a.ndim < 2 || b.ndim < 2 || a.ndim != b.ndim) {
    return false;
  }
  for (size_t i = 0; i + 2 < a.ndim; ++i) {
    if (a.dims[i] != b.dims[i]) {
      return false;
    }
  }
  return true;
}

const char *interval_open(Interval interval) {
  return interval == Interval::Open ? "(" : "[";
}
const char *interval_close(Interval interval) {
  return interval == Interval::Closed ? "]" : ")";
}

} // namespace

const char *check_name(Check category) {
  switch (category) {
  case Check::Layout:
    return "layout";
  case Check::Capability:
    return "capability";
  case Check::Contract:
    return "contract";
  case Check::Numeric:
    return "numeric";
  default:
    return "unknown";
  }
}

std::string shape_text(const Shape &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.rank(); ++i) {
    oss << (i ? ", " : "") << shape.dim(i);
  }
  oss << "]";
  return oss.str();
}

void Validator::report(Check category, std::string_view who,
                       const std::string &message) const {
  const std::string full = text(who) + ": " + message;
  logger().log(litnicelm_utils_events::kValidationFailed, full,
               LogFields().str("category", check_name(category)).str("who", text(who)));
  throw std::runtime_error(full);
}

void Validator::validate(Check category, bool ok, std::string_view who,
                         const char *message) const {
  if (!ok) [[unlikely]] {
    report(category, who, message);
  }
}

// --- Layout

void Validator::validate_not_null(const void *ptr, std::string_view who,
                                  std::string_view what) const {
  if (ptr == nullptr) [[unlikely]] {
    report(Check::Layout, who, text(what) + " is null");
  }
}

void Validator::validate_in_bounds(uint64_t offset, uint64_t bytes, uint64_t limit,
                                   std::string_view who, std::string_view what) const {
  if (offset > limit || bytes > limit - offset) [[unlikely]] {
    report(Check::Layout, who,
           text(what) + " out of bounds: offset " + std::to_string(offset) + " + " +
               std::to_string(bytes) + " bytes exceeds " + std::to_string(limit));
  }
}

void Validator::validate_aligned(uint64_t value, uint64_t alignment, std::string_view who,
                                 std::string_view what) const {
  if (alignment == 0 || value % alignment != 0) [[unlikely]] {
    report(Check::Layout, who,
           text(what) + " " + std::to_string(value) + " is not a multiple of " +
               std::to_string(alignment));
  }
}

void Validator::validate_index(int64_t index, uint64_t size, std::string_view who,
                               std::string_view what) const {
  if (index < 0 || static_cast<uint64_t>(index) >= size) [[unlikely]] {
    report(Check::Layout, who,
           text(what) + " " + std::to_string(index) + " out of range [0, " +
               std::to_string(size) + ")");
  }
}

void Validator::validate_slice(const LayoutSlice &slice, const Shape &shape,
                               uint64_t arena_bytes, std::string_view who) const {
  const uint64_t expected = nbytes(shape, slice.dtype);
  if (expected != slice.bytes) [[unlikely]] {
    report(Check::Layout, who,
           "slice " + slice.name + " holds " + std::to_string(slice.bytes) +
               " bytes, shape " + shape_text(shape) + " " + dtype_name(slice.dtype) +
               " needs " + std::to_string(expected));
  }
  validate_aligned(slice.offset, dtype_size(slice.dtype), who, "slice " + slice.name + " offset");
  validate_in_bounds(slice.offset, slice.bytes, arena_bytes, who, "slice " + slice.name);
}

void Validator::validate_prefix(const TensorViewStruct &view, const Shape &prefix,
                                std::string_view who) const {
  bool ok = prefix.rank() <= view.ndim;
  for (size_t i = 0; ok && i < prefix.rank(); ++i) {
    ok = prefix.dim(i) >= 0 && prefix.dim(i) <= view.dims[i];
  }
  if (!ok) [[unlikely]] {
    report(Check::Layout, who,
           "prefix " + shape_text(prefix) + " does not fit view " + shape_text(shape_of(view)));
  }
}

// --- Capability

void Validator::validate_supported(const TensorViewStruct &view, Device device, DType dtype,
                                   bool contiguous, std::string_view who) const {
  if (device_of(view) != device) [[unlikely]] {
    report(Check::Capability, who,
           std::string("requires a ") + device_name(device) + " tensor, got " +
               device_name(device_of(view)));
  }
  if (dtype_of(view) != dtype) [[unlikely]] {
    report(Check::Capability, who,
           std::string("requires ") + dtype_name(dtype) + ", got " + dtype_name(dtype_of(view)));
  }
  if (contiguous && !is_row_major_contiguous(view)) [[unlikely]] {
    report(Check::Capability, who, "requires a contiguous tensor, got " + view_text(view));
  }
}

// --- Contract

void Validator::validate_same_shape(const TensorViewStruct &a, const TensorViewStruct &b,
                                    std::string_view who) const {
  bool same = a.ndim == b.ndim;
  for (size_t i = 0; same && i < a.ndim; ++i) {
    same = a.dims[i] == b.dims[i];
  }
  if (!same) [[unlikely]] {
    report(Check::Contract, who,
           "shapes differ: " + shape_text(shape_of(a)) + " vs " + shape_text(shape_of(b)));
  }
}

void Validator::validate_same_device_dtype(const TensorViewStruct &a, const TensorViewStruct &b,
                                           std::string_view who) const {
  if (device_of(a) != device_of(b) || dtype_of(a) != dtype_of(b)) [[unlikely]] {
    report(Check::Contract, who,
           std::string("device/dtype differ: ") + device_name(device_of(a)) + " " +
               dtype_name(dtype_of(a)) + " vs " + device_name(device_of(b)) + " " +
               dtype_name(dtype_of(b)));
  }
}

void Validator::validate_rank(const TensorViewStruct &view, size_t rank,
                              std::string_view who) const {
  if (view.ndim != rank) [[unlikely]] {
    report(Check::Contract, who,
           "expected rank " + std::to_string(rank) + ", got " + shape_text(shape_of(view)));
  }
}

void Validator::validate_dims(const TensorViewStruct &view, std::initializer_list<int64_t> dims,
                              std::string_view who) const {
  if (!dims_match(view, dims)) [[unlikely]] {
    report(Check::Contract, who,
           "expected " + dims_text(dims) + ", got " + shape_text(shape_of(view)));
  }
}

void Validator::validate_bsd(const TensorViewStruct &view, int64_t batch, int64_t seq,
                             int64_t last_dim, std::string_view who) const {
  if (!dims_match(view, {batch, seq, last_dim})) [[unlikely]] {
    report(Check::Contract, who,
           "expected [batch, seq, dim] " + dims_text({batch, seq, last_dim}) + ", got " +
               shape_text(shape_of(view)));
  }
}

void Validator::validate_matmul(const TensorViewStruct &a, const TensorViewStruct &b, const TensorViewStruct &out,
                                MatmulKind kind, bool transpose_a, bool transpose_b,
                                std::string_view who) const {
  const auto shapes = [&]() {
    return "a " + shape_text(shape_of(a)) + (transpose_a ? "ᵀ" : "") + ", b " +
           shape_text(shape_of(b)) + (transpose_b ? "ᵀ" : "") + ", out " +
           shape_text(shape_of(out));
  };
  bool ranks = false;
  bool prefixes = false;
  switch (kind) {
  case MatmulKind::Plain:
    ranks = a.ndim == 2 && b.ndim == 2 && out.ndim == 2;
    prefixes = true;
    break;
  case MatmulKind::BroadcastRhs:
    ranks = a.ndim >= 3 && b.ndim == 2 && out.ndim == a.ndim;
    prefixes = ranks && same_prefix(a, out);
    break;
  case MatmulKind::Batched:
    ranks = a.ndim >= 3 && b.ndim == a.ndim && out.ndim == a.ndim;
    prefixes = ranks && same_prefix(a, b) && same_prefix(a, out);
    break;
  case MatmulKind::ReducePrefix:
    ranks = a.ndim >= 3 && b.ndim == a.ndim && out.ndim == 2;
    prefixes = ranks && same_prefix(a, b);
    break;
  }
  if (!ranks) [[unlikely]] {
    report(Check::Contract, who, "matmul ranks do not fit: " + shapes());
    return;  // the dims below need the ranks checked above
  }
  if (!prefixes) [[unlikely]] {
    report(Check::Contract, who, "matmul leading dims differ: " + shapes());
  }
  const int64_t m = a.dims[a.ndim - (transpose_a ? 1 : 2)];
  const int64_t k_a = a.dims[a.ndim - (transpose_a ? 2 : 1)];
  const int64_t k_b = b.dims[b.ndim - (transpose_b ? 1 : 2)];
  const int64_t n = b.dims[b.ndim - (transpose_b ? 2 : 1)];
  if (k_a != k_b) [[unlikely]] {
    report(Check::Contract, who, "matmul inner dims differ: " + shapes());
  }
  if (out.dims[out.ndim - 2] != m || out.dims[out.ndim - 1] != n) [[unlikely]] {
    report(Check::Contract, who,
           "matmul out must end in [" + std::to_string(m) + ", " + std::to_string(n) +
               "]: " + shapes());
  }
}

void Validator::validate_dtype_one_of(const TensorViewStruct &view,
                                      std::initializer_list<DType> dtypes,
                                      std::string_view who) const {
  for (DType d : dtypes) {
    if (dtype_of(view) == d) {
      return;
    }
  }
  std::string allowed;
  for (DType d : dtypes) {
    allowed += (allowed.empty() ? "" : ", ") + std::string(dtype_name(d));
  }
  report(Check::Contract, who,
         std::string("dtype ") + dtype_name(dtype_of(view)) + " not one of " + allowed);
}

void Validator::validate_range(double value, double lo, double hi, Interval interval,
                               std::string_view who, std::string_view what) const {
  const bool low_ok = interval == Interval::Open ? value > lo : value >= lo;
  const bool high_ok = interval == Interval::Closed ? value <= hi : value < hi;
  if (!(low_ok && high_ok)) [[unlikely]] {
    std::ostringstream oss;
    oss << what << " " << value << " outside " << interval_open(interval) << lo << ", " << hi
        << interval_close(interval);
    report(Check::Contract, who, oss.str());
  }
}

void Validator::validate_positive(double value, std::string_view who,
                                  std::string_view what) const {
  if (!(value > 0.0)) [[unlikely]] {
    std::ostringstream oss;
    oss << what << " must be > 0, got " << value;
    report(Check::Contract, who, oss.str());
  }
}

// --- Numeric

void Validator::validate_finite(double value, std::string_view who,
                                std::string_view what) const {
  if (!std::isfinite(value)) [[unlikely]] {
    std::ostringstream oss;
    oss << what << " is not finite: " << value;
    report(Check::Numeric, who, oss.str());
  }
}

// --- access

Validator &validator() { return *installed_validator(); }

void set_validator(Validator &validator) { installed_validator() = &validator; }

ScopedValidator::ScopedValidator(Validator &validator) : previous_(::validator()) {
  set_validator(validator);
}

ScopedValidator::~ScopedValidator() { set_validator(previous_); }
