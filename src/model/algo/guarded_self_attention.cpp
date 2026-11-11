// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "guarded_self_attention.hpp"

#include <litnicelm_utils/checks.hpp>

#include <stdexcept>
#include <utility>

namespace {

constexpr int64_t kGuardElements = 64;
constexpr uint8_t kCanary = 0xA5;
const char *const kGuardSuffix = ".guard";

// The view's elements in row-major order; rows (the last axis) must be
// contiguous, leading axes may be strided (e.g. a prefix of a longer slice).
std::vector<uint8_t> host_bytes(DeviceBackend &backend, const TensorView &view) {
  if (view.is_contiguous()) {
    std::vector<uint8_t> bytes(static_cast<size_t>(view.bytes()));
    backend.copy_device2host(bytes.data(), view.data(), view.bytes());
    return bytes;
  }
  const size_t rank = view.rank();
  const uint64_t row_bytes =
      static_cast<uint64_t>(view.dim(rank - 1)) * dtype_size(view.dtype());
  checker().check(CheckKind::Layout,
                  static_cast<uint64_t>(view.stride_bytes(rank - 1)) ==
                      dtype_size(view.dtype()),
                  "GuardedSelfAttention",
                  [] { return std::string("checked views need contiguous rows"); });
  std::vector<uint8_t> bytes(static_cast<size_t>(view.bytes()));
  std::vector<int64_t> index(rank - 1, 0);
  for (uint64_t row = 0; row * row_bytes < bytes.size(); ++row) {
    int64_t offset = 0;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
      offset += index[axis] * view.stride_bytes(axis);
    }
    backend.copy_device2host(bytes.data() + row * row_bytes,
                             static_cast<const uint8_t *>(view.data()) + offset,
                             row_bytes);
    for (size_t axis = rank - 1; axis-- > 0;) {
      if (++index[axis] < view.dim(axis)) {
        break;
      }
      index[axis] = 0;
    }
  }
  return bytes;
}

} // namespace

GuardedSelfAttention::GuardedSelfAttention(
    int layer_index, std::unique_ptr<ISelfAttention> inner,
    const std::vector<SliceRequest> &requests, const NodeSlices &slices,
    DeviceBackend &backend)
    : idx_(layer_index), inner_(std::move(inner)), backend_(backend) {
  for (const SliceRequest &request : requests) {
    const TensorView guard = slices.at(request.name + kGuardSuffix);
    const std::vector<uint8_t> canary(static_cast<size_t>(guard.bytes()), kCanary);
    backend_.copy_host2device(guard.data(), canary.data(), guard.bytes());
    guards_.emplace_back(request.name, guard);
    if (request.lifetime == SliceLifetime::Step) {
      saved_.push_back(Watched{request.name, slices.at(request.name), 0});
    }
  }
  saved_.push_back(Watched{"x (attention input)", TensorView{}, 0});
}

std::vector<SliceRequest>
GuardedSelfAttention::with_guards(const std::vector<SliceRequest> &requests) {
  std::vector<SliceRequest> out;
  for (const SliceRequest &request : requests) {
    out.push_back(request);
    out.push_back({request.name + kGuardSuffix, Shape{kGuardElements},
                   request.item, SliceRole::Workspace, SliceLifetime::Step});
  }
  return out;
}

void GuardedSelfAttention::set_observer(ITrainingObserver *observer) {
  inner_->set_observer(observer);
}

void GuardedSelfAttention::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  inner_->set_diagnostics(diagnostics);
}

void GuardedSelfAttention::forward(const TensorView &x, TensorView &out) {
  check_guards("before forward");
  inner_->forward(x, out);
  check_guards("after forward");
  saved_.back().view = x;
  for (Watched &watched : saved_) {
    watched.fingerprint = fingerprint(watched.view);
  }
  has_forward_ = true;
}

void GuardedSelfAttention::backward(const TensorView &dout, TensorView &dx) {
  if (!has_forward_) {
    fail("backward without a forward");
  }
  for (const Watched &watched : saved_) {
    if (fingerprint(watched.view) != watched.fingerprint) {
      fail("saved state " + watched.name + " changed between forward and backward");
    }
  }
  check_guards("before backward");
  inner_->backward(dout, dx);
  check_guards("after backward");
}

void GuardedSelfAttention::check_guards(const char *when) const {
  for (const auto &[name, guard] : guards_) {
    for (uint8_t byte : host_bytes(backend_, guard)) {
      if (byte != kCanary) {
        fail(std::string("write past the end of slice ") + name + " (" + when + ")");
      }
    }
  }
}

// FNV-1a over the view's bytes.
uint64_t GuardedSelfAttention::fingerprint(const TensorView &view) const {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : host_bytes(backend_, view)) {
    hash = (hash ^ byte) * 1099511628211ull;
  }
  return hash;
}

void GuardedSelfAttention::fail(const std::string &what) const {
  throw std::runtime_error("GuardedSelfAttention layer " + std::to_string(idx_) +
                           ": " + what);
}
