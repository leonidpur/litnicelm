// Times each op's _f32 and _bf16act variants on a backend plugin, through
// the plugin API, on the shapes of a 512-wide model (128 tokens, FFN 2048,
// 8 heads) at batch 16 (working sets that fit a 32 MB L2) and batch 64
// (4x larger, streaming from DRAM). Prints the speedup next to the one
// memory traffic predicts (bytes moved by _f32 / bytes moved by _bf16act)
// and flags ops well below it. Not a test: run by hand after changing
// kernels.
//
// Usage: litnice_variant_bench <path to backend plugin .so>

#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// A tensor in backend memory, filled with seeded values in [-1, 1].
class BenchTensor {
public:
  BenchTensor(DeviceBackend &backend, DType dtype, Shape shape, uint32_t seed)
      : backend_(backend), dtype_(dtype), shape_(shape),
        bytes_(nbytes(shape, dtype)) {
    data_ = backend_.alloc(bytes_, 256);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> raw(static_cast<size_t>(bytes_));
    for (uint64_t i = 0; i < shape.numel(); ++i) {
      const float v = dist(rng);
      if (dtype == DType::BF16) {
        const uint16_t b = f32_to_bf16_bits(v);
        std::memcpy(raw.data() + i * 2, &b, 2);
      } else {
        std::memcpy(raw.data() + i * 4, &v, 4);
      }
    }
    backend_.copy_host2device(data_, raw.data(), bytes_);
  }
  ~BenchTensor() { backend_.free(data_); }
  BenchTensor(const BenchTensor &) = delete;
  BenchTensor &operator=(const BenchTensor &) = delete;

  TensorView view() const {
    return TensorView(backend_.device(), dtype_, data_, shape_);
  }

private:
  DeviceBackend &backend_;
  DType dtype_;
  Shape shape_;
  uint64_t bytes_;
  void *data_ = nullptr;
};

// Microseconds per call, from 100 calls after 10 warm-up calls; a small
// device-to-host copy waits for the queued kernels to finish.
double time_us(DeviceBackend &backend, const BenchTensor &sync,
               const std::function<void()> &op) {
  float probe = 0.0f;
  auto wait = [&] { backend.copy_device2host(&probe, sync.view().data(), sizeof(probe)); };
  for (int i = 0; i < 10; ++i) {
    op();
  }
  wait();
  constexpr int kCalls = 100;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i) {
    op();
  }
  wait();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(stop - start).count() / kCalls;
}

// One op on both variants: roles give each tensor's dtype in _bf16act
// (A = BF16, P = F32); expected is the traffic ratio _f32 / _bf16act.
struct BenchCase {
  std::string name;
  std::vector<Shape> shapes;
  std::string roles;
  double expected;
  std::function<void(DeviceBackend &, std::vector<TensorView> &)> run;
};

void bench(DeviceBackend &backend, const BenchCase &c) {
  double us[2] = {0.0, 0.0};
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<std::unique_ptr<BenchTensor>> tensors;
    std::vector<TensorView> views;
    for (size_t i = 0; i < c.shapes.size(); ++i) {
      // A = activation, W = weight copy: BF16 in _bf16act; P stays F32.
      const bool bf16 = c.roles[i] == 'A' || c.roles[i] == 'W';
      const DType dtype = variant == 1 && bf16 ? DType::BF16 : DType::F32;
      tensors.push_back(std::make_unique<BenchTensor>(backend, dtype, c.shapes[i],
                                                      static_cast<uint32_t>(i + 1)));
      views.push_back(tensors.back()->view());
    }
    us[variant] = time_us(backend, *tensors[0], [&] { c.run(backend, views); });
  }
  const double speedup = us[0] / us[1];
  const bool low = speedup < 0.8 * c.expected;
  double mb = 0.0;
  for (const Shape &shape : c.shapes) {
    mb += static_cast<double>(nbytes(shape, DType::F32)) / (1 << 20);
  }
  std::cout << std::left << std::setw(40) << c.name << std::right << std::fixed
            << std::setprecision(0) << std::setw(7) << mb
            << std::setprecision(1) << std::setw(9) << us[0] << std::setw(10)
            << us[1] << std::setprecision(2) << std::setw(9) << speedup << "x"
            << std::setw(8) << c.expected << "x" << (low ? "   << below expected" : "")
            << "\n";
}

// The ops at batch size `batch`; names carry the shape.
std::vector<BenchCase> cases(int64_t batch) {
  const Shape act{batch, 128, 512};
  const Shape wide{batch, 128, 2048};
  const Shape qkv{batch, 128, 1536};
  const Shape scores{batch, 8, 128, 128};
  const std::string b = "[" + std::to_string(batch) + ",";
  using V = std::vector<TensorView>;
  return {
      {"fill " + b + "128,512]", {act}, "A", 2.0,
       [](DeviceBackend &b, V &v) { b.fill(v[0], 0.5f); }},
      {"copy " + b + "128,2048]", {wide, wide}, "AA", 2.0,
       [](DeviceBackend &b, V &v) { b.copy(v[0], v[1]); }},
      {"add " + b + "128,512]", {act, act, act}, "AAA", 2.0,
       [](DeviceBackend &b, V &v) { b.add(v[0], v[1], v[2]); }},
      {"add " + b + "128,512] + table [128,512]", {act, Shape{128, 512}, act}, "APA", 2.0,
       [](DeviceBackend &b, V &v) { b.add(v[0], v[1], v[2]); }},
      // F32 accumulator read and written, BF16 addend read: 12 vs 10 bytes.
      {"add_inplace P[128,512] += A", {Shape{128, 512}, Shape{128, 512}}, "PA", 1.2,
       [](DeviceBackend &b, V &v) { b.add_inplace(v[0], v[1]); }},
      {"add_bias_rowwise " + b + "128,1536]", {qkv, Shape{1, 1536}, qkv}, "APA", 2.0,
       [](DeviceBackend &b, V &v) { b.add_bias_rowwise(v[0], v[1], v[2]); }},
      {"add_bias_relu " + b + "128,2048]", {wide, Shape{1, 2048}, wide}, "APA", 2.0,
       [](DeviceBackend &b, V &v) { b.add_bias_relu_rowwise(v[0], v[1], v[2]); }},
      {"add_bias_relu_inplace " + b + "128,2048]", {wide, Shape{1, 2048}}, "AP", 2.0,
       [](DeviceBackend &b, V &v) { b.add_bias_relu_rowwise_inplace(v[0], v[1]); }},
      {"mul_scalar " + b + "8,128,128]", {scores, scores}, "AA", 2.0,
       [](DeviceBackend &b, V &v) { b.mul_scalar(v[0], 0.125f, v[1]); }},
      {"relu " + b + "128,2048]", {wide, wide}, "AA", 2.0,
       [](DeviceBackend &b, V &v) { b.relu(v[0], v[1]); }},
      {"relu_backward " + b + "128,2048]", {wide, wide, wide}, "AAA", 2.0,
       [](DeviceBackend &b, V &v) { b.relu_backward(v[0], v[1], v[2]); }},
      {"relu_backward_inplace " + b + "128,2048]", {wide, wide}, "AA", 2.0,
       [](DeviceBackend &b, V &v) { b.relu_backward_inplace(v[0], v[1]); }},
      {"row_sum (bias grad) " + b + "128,2048]", {wide, Shape{1, 2048}}, "AP", 2.0,
       [](DeviceBackend &b, V &v) { b.row_sum(v[0], v[1]); }},
      {"layernorm " + b + "128,512]", {act, Shape{1, 512}, Shape{1, 512}, act}, "APPA", 2.0,
       [](DeviceBackend &b, V &v) { b.layernorm_forward(v[0], v[1], v[2], v[3]); }},
      {"layernorm_backward " + b + "128,512]",
       {act, Shape{1, 512}, act, act, Shape{1, 512}, Shape{1, 512}}, "APAAPP", 2.0,
       [](DeviceBackend &b, V &v) {
         b.layernorm_backward(v[0], v[1], v[2], v[3], v[4], v[5]);
       }},
      {"causal softmax " + b + "8,128,128]", {scores, scores}, "AA", 2.0,
       [](DeviceBackend &b, V &v) { b.scaled_causal_softmax_rows(v[0], 0.125f, v[1]); }},
      {"causal softmax backward " + b + "8,128,128]", {scores, scores, scores}, "AAA", 2.0,
       [](DeviceBackend &b, V &v) { b.softmax_backward_causal_rows(v[0], v[1], v[2]); }},
      {"causal_mask " + b + "8,128,128]", {scores}, "A", 2.0,
       [](DeviceBackend &b, V &v) { b.apply_causal_mask_inplace(v[0], -1e9f); }},
      {"rotary " + b + "128,512]", {act}, "A", 2.0,
       [](DeviceBackend &b, V &v) { b.rotary_embedding_inplace(v[0], 8, 10000.0f, false); }},
  };
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so>\n";
    return 2;
  }
  Config cfg{};
  cfg.backend.library = argv[1];
  std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  std::cout << std::left << std::setw(40) << "op" << std::right << std::setw(7)
            << "f32 MB" << std::setw(9) << "f32 us" << std::setw(10) << "bf16 us"
            << std::setw(10) << "speedup" << std::setw(9) << "expect" << "\n";
  // A tiny add: the per-call cost of the path, not of the kernel.
  bench(*backend, {"add [1,1,64] (per-call floor)",
                   {Shape{1, 1, 64}, Shape{1, 1, 64}, Shape{1, 1, 64}}, "AAA", 1.0,
                   [](DeviceBackend &b, std::vector<TensorView> &v) { b.add(v[0], v[1], v[2]); }});
  // The F32 gradient arena as the trainer sees it: one N x 1 view, zeroed
  // and scaled every step (a _f32-only path; the "bf16" column runs F32).
  const Shape arena{14 * 1024 * 1024, 1};
  bench(*backend, {"fill gradient arena [14M,1] (F32)", {arena}, "P", 1.0,
                   [](DeviceBackend &b, std::vector<TensorView> &v) { b.fill(v[0], 0.0f); }});
  bench(*backend, {"mul_scalar gradient arena [14M,1] (F32)", {arena, arena}, "PP", 1.0,
                   [](DeviceBackend &b, std::vector<TensorView> &v) { b.mul_scalar(v[0], 0.5f, v[1]); }});
  // GEMMs of the 512-wide model at batch 16 x 128: FP32 SIMT vs BF16 tensor
  // cores (FP32 accumulation).
  std::cout << "-- GEMMs, batch 16 x 128\n";
  const Shape x{16, 128, 512};
  const Shape h{16, 128, 2048};
  const Shape qkv{16, 128, 1536};
  for (const BenchCase &c : std::vector<BenchCase>{
           {"x @ Wqkv [2048,512]x[512,1536]", {x, Shape{512, 1536}, qkv}, "AWA", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_ranked_matrix_rhs(v[0], v[1], v[2]); }},
           {"h @ W2 [2048,2048]x[2048,512]", {h, Shape{2048, 512}, x}, "AWA", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_ranked_matrix_rhs(v[0], v[1], v[2]); }},
           {"dqkv @ Wqkv^T (rhs_t)", {qkv, Shape{512, 1536}, x}, "AWA", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_ranked_matrix_rhs_t(v[0], v[1], v[2]); }},
           {"x^T @ dqkv -> dW F32 (reduce_lhs_t)", {x, qkv, Shape{512, 1536}}, "AAP", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_ranked_reduce_lhs_t(v[0], v[1], v[2]); }},
           {"Q @ K^T, 128 heads x [128,64]", {Shape{128, 128, 64}, Shape{128, 128, 64}, Shape{128, 128, 128}},
            "AAA", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_batched_rhs_t(v[0], v[1], v[2]); }},
           {"P @ V, 128 heads x [128,128]", {Shape{128, 128, 128}, Shape{128, 128, 64}, Shape{128, 128, 64}},
            "AAA", 3.0,
            [](DeviceBackend &b, std::vector<TensorView> &v) { b.gemm_batched(v[0], v[1], v[2]); }},
       }) {
    bench(*backend, c);
  }
  for (const int64_t batch : {16, 64}) {
    std::cout << (batch == 16 ? "-- batch 16: working sets within a 32 MB L2\n"
                              : "-- batch 64: 4x larger, streaming from DRAM\n");
    for (const BenchCase &c : cases(batch)) {
      bench(*backend, c);
    }
  }
  return 0;
}
