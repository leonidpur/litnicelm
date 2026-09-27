// Experiment-tracking plugin writing one directory per run:
//
//   <dir>/<run_id>/config.yaml     resolved config (RUN_START config_yaml)
//   <dir>/<run_id>/meta.json       RUN_START / OPERATION metadata
//   <dir>/<run_id>/metrics.jsonl   one line per METRICS event
//   <dir>/<run_id>/events.jsonl    one line per event except METRICS
//   <dir>/<run_id>/summary.json    RUN_END fields
//
// JSONL lines are appended and closed per event, so a crashed run keeps every
// finished line. Options: dir (default "runs").

#include <tracking_plugin_api.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

struct RunsSink {
  fs::path dir = "runs";
};

void set_error(char *err, uint32_t err_len, const std::string &msg) {
  if (err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "%s", msg.c_str());
  }
}

std::string json_string(const char *s) {
  std::string out = "\"";
  for (const char *p = s != nullptr ? s : ""; *p != '\0'; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out.push_back(static_cast<char>(c));
      }
    }
  }
  return out + "\"";
}

// Shortest text that reads back as the same double; NaN/inf have no JSON
// form and become null.
std::string json_number(double v) {
  if (!std::isfinite(v)) {
    return "null";
  }
  char buf[64];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    if (std::strtod(buf, nullptr) == v) {
      break;
    }
  }
  return buf;
}

std::string json_value(const TrackingField &f) {
  switch (f.kind) {
  case TRACKING_FIELD_I64:
    return std::to_string(f.i64);
  case TRACKING_FIELD_F64:
    return json_number(f.f64);
  case TRACKING_FIELD_BOOL:
    return f.i64 != 0 ? "true" : "false";
  default:
    return json_string(f.str);
  }
}

// config.* fields duplicate config.yaml and config_yaml is written there.
bool is_config_field(const TrackingField &f) {
  const std::string key = f.key;
  return key == "config_yaml" || key.rfind("config.", 0) == 0;
}

const char *event_type_name(uint32_t type) {
  switch (type) {
  case TRACKING_EVENT_RUN_START:
    return "run_start";
  case TRACKING_EVENT_OPERATION:
    return "operation";
  case TRACKING_EVENT_METRICS:
    return "metrics";
  case TRACKING_EVENT_CHECKPOINT:
    return "checkpoint";
  case TRACKING_EVENT_RUN_END:
    return "run_end";
  default:
    return "unknown";
  }
}

// {"type":..,"name":..,"run_id":..,"unix_ms":..,"step":..,"epoch":..,<fields>}
std::string event_json(const TrackingEvent &ev) {
  std::string out = "{\"type\":" + json_string(event_type_name(ev.type)) +
                    ",\"name\":" + json_string(ev.name) +
                    ",\"run_id\":" + json_string(ev.run_id) +
                    ",\"unix_ms\":" + std::to_string(ev.unix_ms) +
                    ",\"step\":" + std::to_string(ev.step) +
                    ",\"epoch\":" + std::to_string(ev.epoch);
  for (uint32_t i = 0; i < ev.n_fields; ++i) {
    if (is_config_field(ev.fields[i])) {
      continue;
    }
    out += "," + json_string(ev.fields[i].key) + ":" + json_value(ev.fields[i]);
  }
  return out + "}";
}

void write_file(const fs::path &path, const std::string &content) {
  std::ofstream out(path, std::ios::trunc);
  out << content;
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
}

void append_line(const fs::path &path, const std::string &line) {
  std::ofstream out(path, std::ios::app);
  out << line << "\n";
  if (!out) {
    throw std::runtime_error("failed to append to " + path.string());
  }
}

const TrackingField *find_field(const TrackingEvent &ev, const char *key) {
  for (uint32_t i = 0; i < ev.n_fields; ++i) {
    if (std::string(ev.fields[i].key) == key) {
      return &ev.fields[i];
    }
  }
  return nullptr;
}

void handle_event(const RunsSink &sink, const TrackingEvent &ev) {
  const fs::path run_dir = sink.dir / ev.run_id;
  fs::create_directories(run_dir);
  const std::string line = event_json(ev);

  switch (ev.type) {
  case TRACKING_EVENT_RUN_START:
    if (const TrackingField *yaml = find_field(ev, "config_yaml")) {
      write_file(run_dir / "config.yaml", yaml->str != nullptr ? yaml->str : "");
    }
    write_file(run_dir / "meta.json", line + "\n");
    append_line(run_dir / "events.jsonl", line);
    break;
  case TRACKING_EVENT_OPERATION:
    write_file(run_dir / "meta.json", line + "\n");
    append_line(run_dir / "events.jsonl", line);
    break;
  case TRACKING_EVENT_METRICS:
    append_line(run_dir / "metrics.jsonl", line);
    break;
  case TRACKING_EVENT_RUN_END:
    write_file(run_dir / "summary.json", line + "\n");
    append_line(run_dir / "events.jsonl", line);
    break;
  default:
    append_line(run_dir / "events.jsonl", line);
    break;
  }
}

void *runs_create(const TrackingField *options, uint32_t n_options, char *err,
                  uint32_t err_len) {
  try {
    auto *sink = new RunsSink();
    for (uint32_t i = 0; i < n_options; ++i) {
      const std::string key = options[i].key;
      const std::string value = options[i].str != nullptr ? options[i].str : "";
      if (key == "dir") {
        sink->dir = value;
      } else if (key.rfind("context.", 0) != 0) {
        delete sink;
        set_error(err, err_len, "runs: unknown option " + key);
        return nullptr;
      }
    }
    if (sink->dir.empty()) {
      delete sink;
      set_error(err, err_len, "runs: dir must not be empty");
      return nullptr;
    }
    return sink;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return nullptr;
  }
}

int runs_on_event(void *instance, const TrackingEvent *ev, char *err,
                  uint32_t err_len) {
  try {
    handle_event(*static_cast<RunsSink *>(instance), *ev);
    return 0;
  } catch (const std::exception &e) {
    set_error(err, err_len, std::string("runs: ") + e.what());
    return 1;
  }
}

int runs_flush(void *) { return 0; }

void runs_destroy(void *instance) { delete static_cast<RunsSink *>(instance); }

const TrackingSinkApiV1 kApi = {
    LITNICE_TRACKING_ABI_VERSION, "runs",      &runs_create,
    &runs_on_event,               &runs_flush, &runs_destroy,
};

} // namespace

extern "C" const TrackingSinkApiV1 *litnice_tracking_get_api() { return &kApi; }
