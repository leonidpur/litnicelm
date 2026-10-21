// Precision presets: the config accepts exactly `precision: fp32` and
// `precision: bf16_mixed` (default fp32) and rejects anything else; the fp32
// policy gives each item type its table dtype; on every backend plugin given,
// fp32 passes the capability check and bf16_mixed fails it (until E3).
//
// Usage: litnice_precision_policy_test <tmp dir> <base config.yaml>
//                                      <backend plugin .so>...

#include "backend/backend_plugin_api.hpp"
#include "backend/device_backend.hpp"
#include "memory/precision_policy.hpp"

#include <config.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++failures;
  }
}

// The base config with `precision: <value>` appended (none when value is
// empty), loaded; the load error when it throws.
struct LoadResult {
  std::optional<Config> cfg;
  std::string error;
};

LoadResult load_with_precision(const std::filesystem::path &tmp,
                               const std::string &base, const std::string &value) {
  std::ifstream in(base);
  std::ostringstream text;
  text << in.rdbuf();
  if (!value.empty()) {
    text << "\nprecision: " << value << "\n";
  }
  const std::filesystem::path path = tmp / "precision.yaml";
  std::ofstream(path) << text.str();
  try {
    return {Config::load_from_file(path.string()), ""};
  } catch (const std::runtime_error &e) {
    return {std::nullopt, e.what()};
  }
}

// The message of the error create() throws, or nullopt when it succeeds.
std::optional<std::string> create_error(const Config &cfg, uint32_t variants) {
  try {
    (void)PrecisionPolicy::create(cfg, variants);
    return std::nullopt;
  } catch (const std::runtime_error &e) {
    return std::string(e.what());
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <tmp dir> <base config.yaml> <backend plugin .so>...\n";
    return 2;
  }
  const std::filesystem::path tmp = argv[1];
  std::filesystem::create_directories(tmp);
  const std::string base = argv[2];

  const LoadResult by_default = load_with_precision(tmp, base, "");
  expect(by_default.cfg && by_default.cfg->precision == Precision::Fp32,
         "no precision key loads as fp32");
  const LoadResult fp32 = load_with_precision(tmp, base, "fp32");
  expect(fp32.cfg && fp32.cfg->precision == Precision::Fp32, "precision: fp32 accepted");
  const LoadResult bf16 = load_with_precision(tmp, base, "bf16_mixed");
  expect(bf16.cfg && bf16.cfg->precision == Precision::Bf16Mixed,
         "precision: bf16_mixed accepted");
  for (const std::string bad : {"fp16", "bf16", "BF16_mixed", "FP32", "mixed", "\"\""}) {
    const LoadResult r = load_with_precision(tmp, base, bad);
    expect(!r.cfg && r.error.find("invalid precision") != std::string::npos,
           "precision: " + bad + " rejected (" + r.error + ")");
  }
  if (!fp32.cfg || !bf16.cfg) {
    std::cout << "Precision policy tests FAILED.\n";
    return 1;
  }

  const PrecisionPolicy policy = PrecisionPolicy::create(*fp32.cfg, kPrecisionVariantF32);
  const struct {
    ItemType item;
    const char *name;
    DType want;
  } table[] = {
      {ItemType::Parameter, "Parameter", DType::F32},
      {ItemType::ComputeWeight, "ComputeWeight", DType::F32},
      {ItemType::Gradient, "Gradient", DType::F32},
      {ItemType::OptimizerState, "OptimizerState", DType::F32},
      {ItemType::Activation, "Activation", DType::F32},
      {ItemType::TokenIds, "TokenIds", DType::I32},
      {ItemType::Loss, "Loss", DType::F32},
  };
  for (const auto &row : table) {
    expect(policy.dtype(row.item) == row.want,
           std::string("fp32: ") + row.name + " is " + dtype_name(row.want));
  }
  expect(policy.compute_weight_aliases_parameter(),
         "fp32: ComputeWeight aliases Parameter");

  for (int i = 3; i < argc; ++i) {
    Config cfg = *fp32.cfg;
    cfg.backend.library = argv[i];
    const std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
    const uint32_t variants = backend->precision_variants();
    expect(!create_error(*fp32.cfg, variants),
           std::string("fp32 passes the capability check on ") + argv[i]);
    const std::optional<std::string> error = create_error(*bf16.cfg, variants);
    expect(error && error->find("ComputeWeight storage not implemented (E3)") !=
                        std::string::npos,
           std::string("bf16_mixed fails the capability check on ") + argv[i] +
               (error ? " (" + *error + ")" : ""));
  }

  std::cout << (failures == 0 ? "All precision policy tests passed.\n"
                              : "Precision policy tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
