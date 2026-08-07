#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

enum class Stage : uint8_t {
  ROOT = 0,
  TRAIN_STEP,
  INFER_STEP,
  FORWARD,
  BACKWARD,
  LAYER,
  SELF_ATTENTION,
  FFN,
  LAYERNORM,
  MATMUL,
  SOFTMAX,
  EMBEDDING,
  LOSS,
  OPTIMIZER_STEP,
  CHECKPOINT_LOAD,
  CHECKPOINT_SAVE,
};

const char *stage_name(Stage stage);

struct ProfileNode {
  Stage stage;
  int index;

  double total_ms = 0.0;
  uint64_t count = 0;

  ProfileNode *parent = nullptr;
  std::vector<std::unique_ptr<ProfileNode>> children;

  ProfileNode(Stage s, int idx = -1, ProfileNode *parent_node = nullptr)
      : stage(s), index(idx), parent(parent_node) {}

  ProfileNode *get_child(Stage s, int idx) {
    for (auto &child : children) {
      if (child->stage == s && child->index == idx) {
        return child.get();
      }
    }
    children.push_back(std::make_unique<ProfileNode>(s, idx, this));
    return children.back().get();
  }
};

class ProfilingController {
public:
  ProfilingController();

  void reset();
  void set_enabled(bool enabled);
  bool enabled() const;

  void enter(Stage stage, int index = -1);
  void leave();

  ProfileNode &root();
  const ProfileNode &root() const;

  std::string summary() const;
  void print_summary(std::ostream &os) const;

  class ScopedProfile {
  public:
    ScopedProfile(ProfilingController &controller, Stage stage, int index = -1);
    ~ScopedProfile();

    ScopedProfile(const ScopedProfile &) = delete;
    ScopedProfile &operator=(const ScopedProfile &) = delete;

  private:
    ProfilingController *controller_ = nullptr;
  };

private:
  struct ActiveFrame {
    ProfileNode *node = nullptr;
    std::uint64_t start_tick_ns = 0;
  };

  static std::uint64_t now_ns();
  static void append_summary_line(std::ostream &os, const ProfileNode &node,
                                  int depth);

  bool enabled_ = true;
  ProfileNode root_;
  ProfileNode *current_ = nullptr;
  std::vector<ActiveFrame> active_frames_;
};
