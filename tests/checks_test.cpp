// Checker tests: each named check passes on valid input and, on invalid
// input, throws "<who>: <message>" after logging one check_failed event with
// its category.

#include <litnicelm_utils/log.hpp>
#include "tensor.hpp"
#include <litnicelm_utils/checks.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

// Keeps each logged record's text and its "category" field.
struct Captured {
  std::string text;
  std::string category;
};
class CaptureDestination final : public ILogDestination {
public:
  explicit CaptureDestination(std::vector<Captured> &out) : out_(out) {}
  void write(const LogRecord &record) override {
    Captured c{std::string(record.def.prefix) + " " + std::string(record.appendix), ""};
    for (const auto &f : record.fields.items()) {
      if (f.key == "category") {
        c.category = std::get<std::string>(f.value);
      }
    }
    out_.push_back(c);
  }

private:
  std::vector<Captured> &out_;
};

struct Fixture {
  std::vector<Captured> logged;
  Logger logger;
  ScopedLogger installed{logger};
  Fixture() {
    ILogDestination &d = logger.add_destination(std::make_unique<CaptureDestination>(logged));
    for (EventType t : {EventType::Info, EventType::Warning, EventType::Error,
                        EventType::Probe, EventType::History}) {
      logger.route(t, d);
    }
  }
};

void expect_pass(const std::string &name, const std::function<void()> &call) {
  Fixture f;
  bool threw = false;
  try {
    call();
  } catch (...) {
    threw = true;
  }
  check(!threw && f.logged.empty(), name + ": passes");
}

void expect_fail(const std::string &name, const char *category, const std::string &text,
                 const std::function<void()> &call) {
  Fixture f;
  std::string what;
  try {
    call();
  } catch (const std::runtime_error &e) {
    what = e.what();
  }
  const bool thrown = what.rfind("test: ", 0) == 0 && what.find(text) != std::string::npos;
  const bool logged = f.logged.size() == 1 && f.logged[0].text == "[Check] " + what &&
                      f.logged[0].category == category;
  check(thrown && logged, name + ": fails as " + category + " with '" + what + "'");
}

} // namespace

int main() {
  const Checker &v = checker();
  float buf[2 * 3 * 4] = {};
  const TensorView m23(Device::CPU, DType::F32, buf, Shape{2, 3});
  const TensorView m34(Device::CPU, DType::F32, buf, Shape{3, 4});
  const TensorView m24(Device::CPU, DType::F32, buf, Shape{2, 4});
  const TensorView b234(Device::CPU, DType::F32, buf, Shape{2, 3, 4});
  const TensorView ids(Device::CPU, DType::I32, buf, Shape{2, 3});
  const TensorView gpu(Device::GPU, DType::F32, buf, Shape{2, 3});

  // generic
  expect_pass("check", [&] { v.check(CheckKind::Contract, true, "test", "never"); });
  expect_fail("check", "numeric", "custom failure",
              [&] { v.check(CheckKind::Numeric, false, "test", [] { return "custom failure"; }); });

  // Layout
  expect_pass("not_null", [&] { v.check_not_null(buf, "test", "base"); });
  expect_fail("not_null", "layout", "base is null", [&] { v.check_not_null(nullptr, "test", "base"); });
  expect_pass("in_bounds", [&] { v.check_in_bounds(8, 8, 16, "test", "slice"); });
  expect_fail("in_bounds", "layout", "exceeds 16", [&] { v.check_in_bounds(9, 8, 16, "test", "slice"); });
  expect_fail("in_bounds overflow", "layout", "out of bounds",
              [&] { v.check_in_bounds(8, std::numeric_limits<uint64_t>::max(), 16, "test", "slice"); });
  expect_pass("aligned", [&] { v.check_aligned(12, 4, "test", "offset"); });
  expect_fail("aligned", "layout", "not a multiple of 4", [&] { v.check_aligned(13, 4, "test", "offset"); });
  expect_pass("index", [&] { v.check_index(2, 3, "test", "layer"); });
  expect_fail("index", "layout", "layer 3 out of range [0, 3)", [&] { v.check_index(3, 3, "test", "layer"); });
  expect_fail("index negative", "layout", "layer -1", [&] { v.check_index(-1, 3, "test", "layer"); });
  const LayoutSlice slice{"w", 16, 24, DType::F32};
  expect_pass("slice", [&] { v.check_slice(slice, Shape{2, 3}, 64, "test"); });
  expect_fail("slice bytes", "layout", "needs 32", [&] { v.check_slice(slice, Shape{2, 4}, 64, "test"); });
  expect_fail("slice bounds", "layout", "exceeds 32", [&] { v.check_slice(slice, Shape{2, 3}, 32, "test"); });
  const LayoutSlice odd{"w", 2, 24, DType::F32};
  expect_fail("slice alignment", "layout", "offset 2 is not a multiple of 4",
              [&] { v.check_slice(odd, Shape{2, 3}, 64, "test"); });
  expect_pass("prefix", [&] { v.check_prefix(b234, Shape{1, 3}, "test"); });
  expect_fail("prefix", "layout", "does not fit", [&] { v.check_prefix(b234, Shape{3}, "test"); });

  // Capability
  expect_pass("supported", [&] { v.check_supported(m23, Device::CPU, DType::F32, true, "test"); });
  expect_fail("supported device", "capability", "requires a cpu tensor, got gpu",
              [&] { v.check_supported(gpu, Device::CPU, DType::F32, true, "test"); });
  expect_fail("supported dtype", "capability", "requires f32, got i32",
              [&] { v.check_supported(ids, Device::CPU, DType::F32, true, "test"); });
  expect_fail("supported contiguity", "capability", "contiguous",
              [&] { v.check_supported(b234.subcols(0, 2), Device::CPU, DType::F32, true, "test"); });

  // Contract
  expect_pass("same_shape", [&] { v.check_same_shape(m23, ids, "test"); });
  expect_fail("same_shape", "contract", "[2, 3] vs [3, 4]", [&] { v.check_same_shape(m23, m34, "test"); });
  expect_pass("same_device_dtype", [&] { v.check_same_device_dtype(m23, m34, "test"); });
  expect_fail("same_device_dtype", "contract", "cpu f32 vs cpu i32",
              [&] { v.check_same_device_dtype(m23, ids, "test"); });
  expect_pass("rank", [&] { v.check_rank(b234, 3, "test"); });
  expect_fail("rank", "contract", "expected rank 2", [&] { v.check_rank(b234, 2, "test"); });
  expect_pass("dims wildcard", [&] { v.check_dims(b234, {-1, 3, 4}, "test"); });
  expect_fail("dims", "contract", "expected [*, 4, 4]", [&] { v.check_dims(b234, {-1, 4, 4}, "test"); });
  expect_pass("bsd", [&] { v.check_bsd(b234, 2, 3, 4, "test"); });
  expect_fail("bsd", "contract", "[batch, seq, dim]", [&] { v.check_bsd(m23, 2, 3, 4, "test"); });
  expect_pass("matmul plain", [&] { v.check_matmul(m23, m34, m24, MatmulKind::Plain, false, false, "test"); });
  expect_fail("matmul inner", "contract", "inner dims differ",
              [&] { v.check_matmul(m23, m24, m24, MatmulKind::Plain, false, false, "test"); });
  expect_fail("matmul out", "contract", "out must end in [2, 4]",
              [&] { v.check_matmul(m23, m34, m23, MatmulKind::Plain, false, false, "test"); });
  expect_pass("matmul transposed b", [&] {
    v.check_matmul(m24, m34, m23, MatmulKind::Plain, false, true, "test");  // [2,4] x [3,4]ᵀ
  });
  const TensorView b223(Device::CPU, DType::F32, buf, Shape{2, 2, 3});
  const TensorView b224(Device::CPU, DType::F32, buf, Shape{2, 2, 4});
  expect_pass("matmul broadcast rhs",
              [&] { v.check_matmul(b223, m34, b224, MatmulKind::BroadcastRhs, false, false, "test"); });
  expect_fail("matmul broadcast ranks", "contract", "ranks do not fit",
              [&] { v.check_matmul(m23, m34, m24, MatmulKind::BroadcastRhs, false, false, "test"); });
  const TensorView b234b(Device::CPU, DType::F32, buf, Shape{2, 3, 4});
  const TensorView c234(Device::CPU, DType::F32, buf, Shape{3, 3, 4});
  const TensorView b244(Device::CPU, DType::F32, buf, Shape{2, 4, 4});
  const TensorView m44(Device::CPU, DType::F32, buf, Shape{4, 4});
  // [2, 3, 4]ᵀ x [2, 3, 4] -> [2, 4, 4]
  expect_pass("matmul batched lhs_t",
              [&] { v.check_matmul(b234, b234b, b244, MatmulKind::Batched, true, false, "test"); });
  expect_fail("matmul batched prefix", "contract", "leading dims differ",
              [&] { v.check_matmul(b234, c234, b244, MatmulKind::Batched, true, false, "test"); });
  // sum over the batch of [3, 4]ᵀ x [3, 4] -> [4, 4]
  expect_pass("matmul reduce prefix",
              [&] { v.check_matmul(b234, b234b, m44, MatmulKind::ReducePrefix, true, false, "test"); });
  expect_fail("matmul reduce out", "contract", "ranks do not fit",
              [&] { v.check_matmul(b234, b234b, b244, MatmulKind::ReducePrefix, true, false, "test"); });
  expect_pass("dtype_one_of", [&] { v.check_dtype_one_of(ids, {DType::I32, DType::F32}, "test"); });
  expect_fail("dtype_one_of", "contract", "i32 not one of f32",
              [&] { v.check_dtype_one_of(ids, {DType::F32}, "test"); });
  expect_pass("range closed-open", [&] { v.check_range(0.0, 0.0, 1.0, Interval::ClosedOpen, "test", "beta1"); });
  expect_fail("range closed-open", "contract", "beta1 1 outside [0, 1)",
              [&] { v.check_range(1.0, 0.0, 1.0, Interval::ClosedOpen, "test", "beta1"); });
  expect_pass("positive", [&] { v.check_positive(0.5, "test", "learning_rate"); });
  expect_fail("positive", "contract", "learning_rate must be > 0, got 0",
              [&] { v.check_positive(0.0, "test", "learning_rate"); });
  expect_fail("positive NaN", "contract", "must be > 0",
              [&] { v.check_positive(std::nan(""), "test", "learning_rate"); });

  // Numeric
  expect_pass("finite", [&] { v.check_finite(1.0, "test", "loss"); });
  expect_fail("finite", "numeric", "loss is not finite",
              [&] { v.check_finite(std::numeric_limits<double>::infinity(), "test", "loss"); });

  // a plain TensorViewStruct, not created as a TensorView
  TensorViewStruct plain{};
  plain.ptr = buf;
  plain.device_id = static_cast<uint32_t>(Device::CPU);
  plain.dtype_id = static_cast<uint32_t>(DType::F32);
  plain.ndim = 2;
  plain.dims[0] = 2;
  plain.dims[1] = 3;
  plain.strides_bytes[0] = 12;
  plain.strides_bytes[1] = 4;
  expect_pass("plain struct supported",
              [&] { v.check_supported(plain, Device::CPU, DType::F32, true, "test"); });
  expect_pass("plain struct same shape as a TensorView", [&] { v.check_same_shape(plain, m23, "test"); });
  plain.strides_bytes[1] = 8;
  expect_fail("plain struct not contiguous", "capability", "contiguous",
              [&] { v.check_supported(plain, Device::CPU, DType::F32, true, "test"); });

  // installation
  {
    Checker other;
    ScopedChecker installed(other);
    check(&checker() == &other, "ScopedChecker installs its checker");
  }
  check(&checker() == &v, "ScopedChecker restores the previous checker");

  std::cout << (g_failures == 0 ? "All checks tests passed.\n" : "Checks tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
