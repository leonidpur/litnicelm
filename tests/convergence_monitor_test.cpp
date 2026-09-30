// Checks ConvergenceMonitor's rules: target loss, patience with min_delta and
// min_epochs, non-finite loss, new-best flags, and state save/restore.

#include "convergence_monitor.hpp"
#include "reporters.hpp"

#include <config.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
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

class MessageCapture final : public ITrainingObserver {
public:
  void on_message(const std::string &line) override { lines.push_back(line); }
  std::vector<std::string> lines;
};

struct Fixture {
  Config cfg{};
  std::vector<std::unique_ptr<ITrainingObserver>> observers;
  MessageCapture *capture = nullptr;
  Reporters reporters{observers};

  Fixture() {
    auto c = std::make_unique<MessageCapture>();
    capture = c.get();
    observers.push_back(std::move(c));
    cfg.training.target_loss = -1.0f;
    cfg.training.min_delta = 0.0f;
    cfg.training.patience_epochs = 0;
    cfg.training.min_epochs = 0;
    cfg.training.stop_on_nonfinite_loss = true;
    cfg.paths.model_file_best = "best.ckpt";
  }
};

EpochMetrics loss(float v) {
  EpochMetrics m;
  m.train_loss = v;
  return m;
}

// Feeds losses for epochs 1..N; returns the epoch that stopped (0 if none).
uint32_t run(ConvergenceMonitor &monitor, const std::vector<float> &losses,
             std::vector<bool> *new_best = nullptr) {
  for (size_t i = 0; i < losses.size(); ++i) {
    const ConvergenceVerdict v =
        monitor.observe(static_cast<uint32_t>(i + 1), loss(losses[i]));
    if (new_best != nullptr) {
      new_best->push_back(v.new_best);
    }
    if (!v.decision.decided) {
      return 9999;  // the monitor always decides
    }
    if (!v.decision.continue_training) {
      return static_cast<uint32_t>(i + 1);
    }
  }
  return 0;
}

bool contains(const std::vector<std::string> &lines, const std::string &text) {
  for (const auto &l : lines) {
    if (l.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  {
    Fixture f;
    f.cfg.training.target_loss = 0.1f;
    ConvergenceMonitor m(f.cfg, f.reporters);
    check(run(m, {0.5f, 0.2f, 0.09f, 0.05f}) == 3, "target loss stops at the first epoch at or below it");
    check(m.stop_reason_text().rfind("target loss reached at epoch 3", 0) == 0,
          "target stop message: " + m.stop_reason_text());
    check(contains(f.capture->lines, "Early stop requested: target loss reached"),
          "stop is reported as a message");
  }
  {
    Fixture f;
    f.cfg.training.min_delta = 0.01f;
    f.cfg.training.patience_epochs = 2;
    ConvergenceMonitor m(f.cfg, f.reporters);
    std::vector<bool> best;
    // 0.995 and 0.99 each gain at most 0.01 over the best, not more.
    check(run(m, {1.0f, 0.995f, 0.99f, 0.5f}, &best) == 3,
          "patience 2 stops after two epochs without improvement > min_delta");
    check(best == std::vector<bool>({true, false, false}), "new_best only on epoch 1");
    check(m.best_epoch() == 1 && m.best_loss() == 1.0f, "best stays at epoch 1");
  }
  {
    Fixture f;
    f.cfg.training.patience_epochs = 1;
    f.cfg.training.min_epochs = 3;
    ConvergenceMonitor m(f.cfg, f.reporters);
    check(run(m, {1.0f, 1.0f, 1.0f, 1.0f}) == 3,
          "patience counts only from min_epochs");
  }
  {
    Fixture f;
    ConvergenceMonitor m(f.cfg, f.reporters);
    std::vector<bool> best;
    check(run(m, {1.0f, 0.8f, 0.9f, 0.7f}, &best) == 0, "no stop rules: never stops");
    check(best == std::vector<bool>({true, true, false, true}), "new_best on each improvement");
  }
  {
    Fixture f;
    ConvergenceMonitor m(f.cfg, f.reporters);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    check(run(m, {1.0f, nan}) == 2, "non-finite loss stops when stop_on_nonfinite_loss");
    check(m.best_loss() == 1.0f && std::isnan(m.last_epoch_loss()),
          "non-finite loss leaves best unchanged, records last loss");
    Fixture g;
    g.cfg.training.stop_on_nonfinite_loss = false;
    ConvergenceMonitor n(g.cfg, g.reporters);
    check(run(n, {1.0f, nan, 0.5f}) == 0, "non-finite loss continues when not configured to stop");
  }
  {
    Fixture f;
    f.cfg.training.min_delta = 0.01f;
    f.cfg.training.patience_epochs = 2;
    ConvergenceMonitor a(f.cfg, f.reporters);
    run(a, {1.0f, 0.999f, 0.998f});  // stops at epoch 3
    const CheckpointConvergenceState saved = a.state();
    ConvergenceMonitor b(f.cfg, f.reporters);
    b.restore(saved);
    const CheckpointConvergenceState again = b.state();
    check(again.best_loss == saved.best_loss && again.best_epoch == saved.best_epoch &&
              again.epochs_without_improvement == saved.epochs_without_improvement &&
              again.flags == saved.flags && again.stop_reason == saved.stop_reason &&
              again.last_epoch_loss == saved.last_epoch_loss,
          "state round-trips through restore()");
    check(b.should_stop(), "restored state keeps the stop");
    b.clear_persisted_stop_for_resume();
    check(!b.should_stop() && b.best_epoch() == 1, "clearing a persisted stop keeps the best");
    check(contains(f.capture->lines, "Resumed checkpoint carries a prior early-stop state"),
          "clearing is reported as a message");
  }

  std::cout << (g_failures == 0 ? "All convergence monitor tests passed.\n"
                                : "Convergence monitor tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
