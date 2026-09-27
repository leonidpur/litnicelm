#pragma once

#include <config.hpp>
#include <tracking_plugin_api.h>

#include <cstdint>
#include <string>
#include <vector>

// One tracking event under construction. Owns its strings, so the
// TrackingField pointers handed to plugins stay valid during emit().
class TrackingEventBuilder {
public:
  TrackingEventBuilder(TrackingEventType type, std::string name);

  TrackingEventBuilder &step(uint64_t value);
  TrackingEventBuilder &epoch(uint32_t value);
  TrackingEventBuilder &i64(const std::string &key, int64_t value);
  TrackingEventBuilder &f64(const std::string &key, double value);
  TrackingEventBuilder &str(const std::string &key, const std::string &value);
  TrackingEventBuilder &boolean(const std::string &key, bool value);

private:
  friend class ExperimentTrackingSink;

  struct Field {
    std::string key;
    TrackingFieldKind kind;
    int64_t i64 = 0;
    double f64 = 0.0;
    std::string str;
  };

  TrackingEventType type_;
  std::string name_;
  uint64_t step_ = 0;
  uint32_t epoch_ = 0;
  std::vector<Field> fields_;
};

// Loads the experiment-tracking plugins named in cfg.tracking.sinks and fans
// every event out to them. With no sinks configured, loads the journal plugin
// (next to the executable) writing to paths.journal_file.
//
// A plugin that fails to load or create aborts construction. A plugin whose
// on_event fails later is reported on stderr and disabled; the run continues.
class ExperimentTrackingSink {
public:
  ExperimentTrackingSink(const Config &cfg, const std::string &run_name);
  ~ExperimentTrackingSink();

  ExperimentTrackingSink(const ExperimentTrackingSink &) = delete;
  ExperimentTrackingSink &operator=(const ExperimentTrackingSink &) = delete;

  const std::string &run_id() const { return runId_; }
  void emit(const TrackingEventBuilder &event);
  void flush();
  size_t active_sinks() const;

private:
  struct LoadedSink {
    std::string library;
    void *handle = nullptr;
    const TrackingSinkApiV1 *api = nullptr;
    void *instance = nullptr;
    bool enabled = true;
  };

  void load_sink(const std::string &library,
                 const std::vector<std::pair<std::string, std::string>> &options);
  void unload_all();

  std::string runId_;
  std::vector<LoadedSink> sinks_;
};
