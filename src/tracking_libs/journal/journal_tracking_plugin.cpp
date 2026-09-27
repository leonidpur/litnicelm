// Experiment-tracking plugin writing the human-readable operation journal
// (paths.journal_file): one appended entry per finished operation or run.
//
// Options: path (defaults to context.journal_file; empty disables writing).

#include <tracking_plugin_api.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct JournalSink {
  std::string path;
};

void set_error(char *err, uint32_t err_len, const std::string &msg) {
  if (err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "%s", msg.c_str());
  }
}

std::string timestamp_local(int64_t unix_ms) {
  const std::time_t t = static_cast<std::time_t>(unix_ms / 1000);
  std::tm tmv{};
  localtime_r(&t, &tmv);
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

std::string format_duration_ms(int64_t ms) {
  if (ms < 0) {
    ms = 0;
  }
  const int64_t total_seconds = ms / 1000;
  const int64_t hours = total_seconds / 3600;
  const int64_t minutes = (total_seconds % 3600) / 60;
  const int64_t seconds = total_seconds % 60;
  const int64_t millis = ms % 1000;

  std::ostringstream oss;
  if (hours > 0) {
    oss << hours << "h ";
  }
  if (hours > 0 || minutes > 0) {
    oss << minutes << "m ";
  }
  oss << seconds << "." << std::setw(3) << std::setfill('0') << millis << "s";
  return oss.str();
}

std::string field_value(const TrackingField &f) {
  std::ostringstream oss;
  switch (f.kind) {
  case TRACKING_FIELD_I64:
    oss << f.i64;
    break;
  case TRACKING_FIELD_F64:
    oss << f.f64;
    break;
  case TRACKING_FIELD_BOOL:
    oss << (f.i64 != 0 ? "true" : "false");
    break;
  default:
    oss << (f.str != nullptr ? f.str : "");
    break;
  }
  return oss.str();
}

// Event fields by key, in arrival order for prefixed groups.
class Fields {
public:
  explicit Fields(const TrackingEvent &ev) : ev_(ev) {
    for (uint32_t i = 0; i < ev.n_fields; ++i) {
      by_key_[ev.fields[i].key] = &ev.fields[i];
    }
  }
  bool has(const char *key) const { return by_key_.count(key) != 0; }
  std::string str(const char *key) const {
    auto it = by_key_.find(key);
    return it == by_key_.end() ? std::string() : field_value(*it->second);
  }
  int64_t i64(const char *key) const {
    auto it = by_key_.find(key);
    return it == by_key_.end() ? 0 : it->second->i64;
  }
  // "model.n_layers=8" lines for fields "config.model.*" (group "model").
  std::string config_lines(const std::string &group) const {
    const std::string prefix = "config." + group + ".";
    std::string out;
    for (uint32_t i = 0; i < ev_.n_fields; ++i) {
      const std::string key = ev_.fields[i].key;
      if (key.rfind(prefix, 0) != 0) {
        continue;
      }
      if (!out.empty()) {
        out += "\n";
      }
      out += key.substr(std::string("config.").size()) + "=" +
             field_value(ev_.fields[i]);
    }
    return out;
  }

private:
  const TrackingEvent &ev_;
  std::map<std::string, const TrackingField *> by_key_;
};

std::string run_end_details(const Fields &f) {
  const int64_t total_ms = f.i64("time_total_ms");
  const int64_t avg_ms = f.i64("avg_epoch_ms");
  std::ostringstream oss;
  oss << "Status: " << f.str("status") << "\n\n"
      << "Mode: " << f.str("mode") << "\n\n"
      << "Input corpus: " << f.str("input_corpus") << "\n\n"
      << "Dataset: " << f.str("dataset") << "\n\n"
      << "Time total: " << format_duration_ms(total_ms) << " (" << total_ms
      << " ms)\n\n"
      << "Average epoch time: " << format_duration_ms(avg_ms) << " (" << avg_ms
      << " ms over " << f.i64("measured_epochs") << " measured epoch(s))\n\n"
      << "Epochs completed: " << f.str("epochs_completed") << "\n\n"
      << "Global step: " << f.str("global_step");
  if (f.has("checkpoint_latest")) {
    oss << "\n\nCheckpoint latest: " << f.str("checkpoint_latest")
        << "\nBest checkpoint: " << f.str("checkpoint_best");
  }
  if (f.has("best_loss")) {
    oss << "\nBest loss: " << f.str("best_loss")
        << "\nBest epoch: " << f.str("best_epoch");
  }
  if (f.has("stop_reason")) {
    oss << "\nStop reason: " << f.str("stop_reason");
  }
  oss << "\n\nModel config:\n" << f.config_lines("model")
      << "\n\nTraining config:\n" << f.config_lines("training");
  return oss.str();
}

// Labels for OPERATION fields, e.g. training_corpus -> "Training Corpus".
std::string operation_label(const std::string &key) {
  static const std::map<std::string, std::string> labels = {
      {"status", "Status"},
      {"training_corpus", "Training Corpus"},
      {"resolved_corpus", "Resolved Corpus"},
      {"artifacts_dir", "Artifacts Dir"},
      {"tokenizer", "Tokenizer"},
      {"vocab_size", "Vocab Size"},
      {"input_corpus", "Input Corpus"},
      {"output_dataset", "Output Dataset"},
  };
  auto it = labels.find(key);
  return it == labels.end() ? key : it->second;
}

std::string operation_details(const TrackingEvent &ev) {
  std::string out;
  for (uint32_t i = 0; i < ev.n_fields; ++i) {
    if (!out.empty()) {
      out += "\n\n";
    }
    out += operation_label(ev.fields[i].key) + ": " + field_value(ev.fields[i]);
  }
  return out;
}

void append_entry(const std::string &path, int64_t unix_ms,
                  const std::string &op_name, const std::string &details) {
  const std::filesystem::path p(path);
  if (!p.parent_path().empty()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream journal(p, std::ios::app);
  if (!journal) {
    throw std::runtime_error("failed to open journal: " + p.string());
  }
  journal << "[" << timestamp_local(unix_ms) << "] OPERATION: " << op_name
          << "\n\n";
  journal << details << "\n\n";
  if (!journal) {
    throw std::runtime_error("failed to write journal: " + p.string());
  }
}

void *journal_create(const TrackingField *options, uint32_t n_options,
                     char *err, uint32_t err_len) {
  try {
    auto *sink = new JournalSink();
    std::string context_path;
    for (uint32_t i = 0; i < n_options; ++i) {
      const std::string key = options[i].key;
      const std::string value = options[i].str != nullptr ? options[i].str : "";
      if (key == "path") {
        sink->path = value;
      } else if (key == "context.journal_file") {
        context_path = value;
      } else if (key.rfind("context.", 0) != 0) {
        delete sink;
        set_error(err, err_len, "journal: unknown option " + key);
        return nullptr;
      }
    }
    if (sink->path.empty()) {
      sink->path = context_path;
    }
    return sink;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return nullptr;
  }
}

int journal_on_event(void *instance, const TrackingEvent *ev, char *err,
                     uint32_t err_len) {
  try {
    auto *sink = static_cast<JournalSink *>(instance);
    if (sink->path.empty()) {
      return 0;
    }
    if (ev->type == TRACKING_EVENT_OPERATION) {
      append_entry(sink->path, ev->unix_ms, ev->name, operation_details(*ev));
    } else if (ev->type == TRACKING_EVENT_RUN_END) {
      append_entry(sink->path, ev->unix_ms, ev->name,
                   run_end_details(Fields(*ev)));
    }
    return 0;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return 1;
  }
}

int journal_flush(void *) { return 0; }

void journal_destroy(void *instance) {
  delete static_cast<JournalSink *>(instance);
}

const TrackingSinkApiV1 kApi = {
    LITNICE_TRACKING_ABI_VERSION, "journal", &journal_create,
    &journal_on_event,            &journal_flush, &journal_destroy,
};

} // namespace

extern "C" const TrackingSinkApiV1 *litnice_tracking_get_api() { return &kApi; }
