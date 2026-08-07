#include "profiling.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
using SteadyClock = std::chrono::steady_clock;

std::string format_duration_ms(double ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);

  if (ms < 1000.0) {
    oss << ms << " ms";
    return oss.str();
  }

  const double total_seconds = ms / 1000.0;
  if (total_seconds < 60.0) {
    oss << total_seconds << " s";
    return oss.str();
  }

  const std::uint64_t whole_seconds =
      static_cast<std::uint64_t>(total_seconds);
  const std::uint64_t hours = whole_seconds / 3600;
  const std::uint64_t minutes = (whole_seconds % 3600) / 60;
  const double seconds = total_seconds -
                         static_cast<double>(hours * 3600 + minutes * 60);

  if (hours > 0) {
    oss << hours << "h " << std::setw(2) << std::setfill('0') << minutes
        << "m " << std::setw(6) << seconds << "s";
    return oss.str();
  }

  oss << minutes << "m " << std::setw(6) << std::setfill('0') << seconds
      << "s";
  return oss.str();
}
}

const char *stage_name(Stage stage) {
  switch (stage) {
  case Stage::ROOT:
    return "root";
  case Stage::TRAIN_STEP:
    return "train_step";
  case Stage::INFER_STEP:
    return "infer_step";
  case Stage::FORWARD:
    return "forward";
  case Stage::BACKWARD:
    return "backward";
  case Stage::LAYER:
    return "layer";
  case Stage::SELF_ATTENTION:
    return "self_attention";
  case Stage::FFN:
    return "ffn";
  case Stage::LAYERNORM:
    return "layernorm";
  case Stage::MATMUL:
    return "matmul";
  case Stage::SOFTMAX:
    return "softmax";
  case Stage::EMBEDDING:
    return "embedding";
  case Stage::LOSS:
    return "loss";
  case Stage::OPTIMIZER_STEP:
    return "optimizer_step";
  case Stage::CHECKPOINT_LOAD:
    return "checkpoint_load";
  case Stage::CHECKPOINT_SAVE:
    return "checkpoint_save";
  default:
    return "unknown";
  }
}

ProfilingController::ProfilingController()
    : root_(Stage::ROOT, -1, nullptr), current_(&root_) {}

void ProfilingController::reset() {
  root_ = ProfileNode(Stage::ROOT, -1, nullptr);
  current_ = &root_;
  active_frames_.clear();
}

void ProfilingController::set_enabled(bool enabled) { enabled_ = enabled; }

bool ProfilingController::enabled() const { return enabled_; }

void ProfilingController::enter(Stage stage, int index) {
  if (!enabled_) {
    return;
  }
  if (current_ == nullptr) {
    throw std::runtime_error("ProfilingController::enter: current node is null");
  }
  ProfileNode *node = current_->get_child(stage, index);
  active_frames_.push_back(ActiveFrame{node, now_ns()});
  current_ = node;
}

void ProfilingController::leave() {
  if (!enabled_) {
    return;
  }
  if (active_frames_.empty()) {
    throw std::runtime_error("ProfilingController::leave: no active frame");
  }

  const std::uint64_t end_tick_ns = now_ns();
  ActiveFrame frame = active_frames_.back();
  active_frames_.pop_back();

  const double elapsed_ms =
      static_cast<double>(end_tick_ns - frame.start_tick_ns) / 1'000'000.0;
  frame.node->total_ms += elapsed_ms;
  frame.node->count += 1;

  current_ = frame.node->parent;
  if (current_ == nullptr) {
    current_ = &root_;
  }
}

ProfileNode &ProfilingController::root() { return root_; }

const ProfileNode &ProfilingController::root() const { return root_; }

std::string ProfilingController::summary() const {
  std::ostringstream oss;
  print_summary(oss);
  return oss.str();
}

void ProfilingController::print_summary(std::ostream &os) const {
  os << "[Profiling Summary]\n";
  for (const auto &child : root_.children) {
    append_summary_line(os, *child, 0);
  }
}

ProfilingController::ScopedProfile::ScopedProfile(ProfilingController &controller,
                                                  Stage stage, int index)
    : controller_(&controller) {
  controller_->enter(stage, index);
}

ProfilingController::ScopedProfile::~ScopedProfile() {
  if (controller_ != nullptr) {
    controller_->leave();
  }
}

std::uint64_t ProfilingController::now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now().time_since_epoch())
          .count());
}

void ProfilingController::append_summary_line(std::ostream &os,
                                              const ProfileNode &node,
                                              int depth) {
  for (int i = 0; i < depth; ++i) {
    os << "  ";
  }

  os << "- " << stage_name(node.stage);
  if (node.index >= 0) {
    os << "[" << node.index << "]";
  }

  os << " total=" << format_duration_ms(node.total_ms)
     << " count=" << node.count;

  if (node.count > 0) {
    os << " avg="
       << format_duration_ms(node.total_ms / static_cast<double>(node.count));
  }
  os << "\n";

  for (const auto &child : node.children) {
    append_summary_line(os, *child, depth + 1);
  }
}
