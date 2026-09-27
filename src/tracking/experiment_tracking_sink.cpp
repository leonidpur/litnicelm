#include "experiment_tracking_sink.hpp"

#include <chrono>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
constexpr const char *kDefaultJournalLibrary = "liblitnice_tracking_journal.so";
constexpr uint32_t kErrLen = 512;

std::string make_run_id(const std::string &run_name) {
  const auto now = std::time(nullptr);
  std::tm tmv{};
  localtime_r(&now, &tmv);
  // A random suffix keeps runs started within the same second apart.
  std::random_device rd;
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y%m%d-%H%M%S") << "-" << std::hex
      << std::setw(4) << std::setfill('0') << (rd() & 0xffffu) << std::dec;
  if (!run_name.empty()) {
    oss << "_" << run_name;
  }
  return oss.str();
}

std::filesystem::path executable_dir() {
  std::error_code ec;
  const std::filesystem::path exe =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  return ec ? std::filesystem::path(".") : exe.parent_path();
}

int64_t now_unix_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
} // namespace

TrackingEventBuilder::TrackingEventBuilder(TrackingEventType type,
                                           std::string name)
    : type_(type), name_(std::move(name)) {}

TrackingEventBuilder &TrackingEventBuilder::step(uint64_t value) {
  step_ = value;
  return *this;
}

TrackingEventBuilder &TrackingEventBuilder::epoch(uint32_t value) {
  epoch_ = value;
  return *this;
}

TrackingEventBuilder &TrackingEventBuilder::i64(const std::string &key,
                                                int64_t value) {
  fields_.push_back(Field{key, TRACKING_FIELD_I64, value, 0.0, {}});
  return *this;
}

TrackingEventBuilder &TrackingEventBuilder::f64(const std::string &key,
                                                double value) {
  fields_.push_back(Field{key, TRACKING_FIELD_F64, 0, value, {}});
  return *this;
}

TrackingEventBuilder &TrackingEventBuilder::str(const std::string &key,
                                                const std::string &value) {
  fields_.push_back(Field{key, TRACKING_FIELD_STR, 0, 0.0, value});
  return *this;
}

TrackingEventBuilder &TrackingEventBuilder::boolean(const std::string &key,
                                                    bool value) {
  fields_.push_back(Field{key, TRACKING_FIELD_BOOL, value ? 1 : 0, 0.0, {}});
  return *this;
}

ExperimentTrackingSink::ExperimentTrackingSink(const Config &cfg,
                                               const std::string &run_name)
    : runId_(make_run_id(run_name)) {
  try {
    if (cfg.tracking.sinks.empty()) {
      if (!cfg.paths.journal_file.empty()) {
        load_sink((executable_dir() / kDefaultJournalLibrary).string(),
                  {{"context.journal_file", cfg.paths.journal_file}});
      }
      return;
    }
    for (const std::string &library : cfg.tracking.sinks) {
      std::vector<std::pair<std::string, std::string>> options = {
          {"context.journal_file", cfg.paths.journal_file}};
      // The sink's own options are attached in load_sink once its name is known.
      options.insert(options.end(), cfg.tracking.options.begin(),
                     cfg.tracking.options.end());
      load_sink(library, options);
    }
  } catch (...) {
    unload_all();
    throw;
  }
}

ExperimentTrackingSink::~ExperimentTrackingSink() {
  flush();
  unload_all();
}

void ExperimentTrackingSink::load_sink(
    const std::string &library,
    const std::vector<std::pair<std::string, std::string>> &options) {
  LoadedSink sink;
  sink.library = library;
  sink.handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (sink.handle == nullptr) {
    const char *err = dlerror();
    throw std::runtime_error("ExperimentTrackingSink: failed to load " +
                             library + ": " + (err != nullptr ? err : "?"));
  }
  sinks_.push_back(sink);
  LoadedSink &loaded = sinks_.back();

  auto *get_api = reinterpret_cast<TrackingGetApiFn>(
      dlsym(loaded.handle, "litnice_tracking_get_api"));
  if (get_api == nullptr) {
    throw std::runtime_error("ExperimentTrackingSink: " + library +
                             " does not export litnice_tracking_get_api");
  }
  loaded.api = get_api();
  if (loaded.api == nullptr ||
      loaded.api->abi_version != LITNICE_TRACKING_ABI_VERSION) {
    throw std::runtime_error(
        "ExperimentTrackingSink: tracking ABI mismatch in " + library +
        " (expected " + std::to_string(LITNICE_TRACKING_ABI_VERSION) + ", got " +
        (loaded.api == nullptr ? std::string("none")
                               : std::to_string(loaded.api->abi_version)) +
        ")");
  }

  // Pass context.* options and this sink's own "<name>.<option>" keys, with
  // the "<name>." prefix removed.
  const std::string own_prefix = std::string(loaded.api->name) + ".";
  std::vector<std::pair<std::string, std::string>> selected;
  for (const auto &[key, value] : options) {
    if (key.rfind("context.", 0) == 0) {
      selected.emplace_back(key, value);
    } else if (key.rfind(own_prefix, 0) == 0) {
      selected.emplace_back(key.substr(own_prefix.size()), value);
    }
  }
  std::vector<TrackingField> fields;
  fields.reserve(selected.size());
  for (const auto &[key, value] : selected) {
    fields.push_back(
        TrackingField{key.c_str(), TRACKING_FIELD_STR, 0, 0.0, value.c_str()});
  }

  char err[kErrLen] = {};
  loaded.instance = loaded.api->create(
      fields.data(), static_cast<uint32_t>(fields.size()), err, kErrLen);
  if (loaded.instance == nullptr) {
    throw std::runtime_error("ExperimentTrackingSink: " +
                             std::string(loaded.api->name) +
                             " sink create failed: " + err);
  }
}

void ExperimentTrackingSink::emit(const TrackingEventBuilder &event) {
  std::vector<TrackingField> fields;
  fields.reserve(event.fields_.size());
  for (const auto &f : event.fields_) {
    fields.push_back(TrackingField{f.key.c_str(), static_cast<uint32_t>(f.kind),
                                   f.i64, f.f64, f.str.c_str()});
  }
  TrackingEvent ev{};
  ev.type = static_cast<uint32_t>(event.type_);
  ev.name = event.name_.c_str();
  ev.run_id = runId_.c_str();
  ev.step = event.step_;
  ev.epoch = event.epoch_;
  ev.unix_ms = now_unix_ms();
  ev.fields = fields.data();
  ev.n_fields = static_cast<uint32_t>(fields.size());

  for (LoadedSink &sink : sinks_) {
    if (!sink.enabled) {
      continue;
    }
    char err[kErrLen] = {};
    if (sink.api->on_event(sink.instance, &ev, err, kErrLen) != 0) {
      std::cerr << "[ExperimentTracking][WARN] " << sink.api->name
                << " sink failed on " << event.name_ << " event and is disabled: "
                << err << "\n";
      sink.enabled = false;
    }
  }
}

void ExperimentTrackingSink::flush() {
  for (LoadedSink &sink : sinks_) {
    if (sink.enabled && sink.instance != nullptr) {
      sink.api->flush(sink.instance);
    }
  }
}

size_t ExperimentTrackingSink::active_sinks() const {
  size_t n = 0;
  for (const LoadedSink &sink : sinks_) {
    n += sink.enabled ? 1 : 0;
  }
  return n;
}

void ExperimentTrackingSink::unload_all() {
  for (LoadedSink &sink : sinks_) {
    if (sink.api != nullptr && sink.instance != nullptr) {
      sink.api->destroy(sink.instance);
    }
    if (sink.handle != nullptr) {
      dlclose(sink.handle);
    }
  }
  sinks_.clear();
}
