// Experiment-tracking plugin writing TensorBoard scalars, one log directory
// per run: <dir>/<run_id>/events.out.tfevents.<time>.<host>
//
// Scalars (x axis: completed optimizer steps, so resumed runs line up):
//   <metrics name>/<field>   e.g. epoch/train_loss, epoch/epoch_ms,
//                            step/train_loss
//   run_end/<field>          numeric run summary, e.g. run_end/best_loss
//
// Options: dir (default "tensorboard").
// View with: tensorboard --logdir <dir>

#include "tfevents_writer.hpp"

#include <tracking_plugin_api.h>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;

struct TensorBoardSink {
  fs::path dir = "tensorboard";
  std::map<std::string, std::unique_ptr<tfevents::Writer>> writers;
};

void set_error(char *err, uint32_t err_len, const std::string &msg) {
  if (err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "%s", msg.c_str());
  }
}

std::string hostname() {
  char buf[256] = {};
  return gethostname(buf, sizeof(buf) - 1) == 0 ? buf : "localhost";
}

tfevents::Writer &writer_for(TensorBoardSink &sink, const TrackingEvent &ev) {
  auto it = sink.writers.find(ev.run_id);
  if (it != sink.writers.end()) {
    return *it->second;
  }
  const fs::path run_dir = sink.dir / ev.run_id;
  fs::create_directories(run_dir);
  const double wall_time = static_cast<double>(ev.unix_ms) / 1000.0;
  const fs::path file =
      run_dir / ("events.out.tfevents." +
                 std::to_string(ev.unix_ms / 1000) + "." + hostname());
  auto writer = std::make_unique<tfevents::Writer>(file.string(), wall_time);
  return *sink.writers.emplace(ev.run_id, std::move(writer)).first->second;
}

bool is_config_field(const std::string &key) {
  return key.rfind("config.", 0) == 0;
}

// Writes every numeric field of ev as "<prefix>/<key>".
void write_scalars(TensorBoardSink &sink, const TrackingEvent &ev,
                   const std::string &prefix) {
  tfevents::Writer &writer = writer_for(sink, ev);
  const double wall_time = static_cast<double>(ev.unix_ms) / 1000.0;
  for (uint32_t i = 0; i < ev.n_fields; ++i) {
    const TrackingField &f = ev.fields[i];
    float value = 0.0f;
    if (f.kind == TRACKING_FIELD_F64) {
      value = static_cast<float>(f.f64);
    } else if (f.kind == TRACKING_FIELD_I64) {
      value = static_cast<float>(f.i64);
    } else {
      continue;
    }
    if (is_config_field(f.key)) {
      continue;
    }
    writer.write(tfevents::encode_scalar_event(
        wall_time, static_cast<int64_t>(ev.step), prefix + "/" + f.key, value));
  }
}

void *tensorboard_create(const TrackingField *options, uint32_t n_options,
                         char *err, uint32_t err_len) {
  try {
    auto *sink = new TensorBoardSink();
    for (uint32_t i = 0; i < n_options; ++i) {
      const std::string key = options[i].key;
      const std::string value = options[i].str != nullptr ? options[i].str : "";
      if (key == "dir") {
        sink->dir = value;
      } else if (key.rfind("context.", 0) != 0) {
        delete sink;
        set_error(err, err_len, "tensorboard: unknown option " + key);
        return nullptr;
      }
    }
    if (sink->dir.empty()) {
      delete sink;
      set_error(err, err_len, "tensorboard: dir must not be empty");
      return nullptr;
    }
    return sink;
  } catch (const std::exception &e) {
    set_error(err, err_len, e.what());
    return nullptr;
  }
}

int tensorboard_on_event(void *instance, const TrackingEvent *ev, char *err,
                         uint32_t err_len) {
  try {
    auto &sink = *static_cast<TensorBoardSink *>(instance);
    if (ev->type == TRACKING_EVENT_METRICS) {
      write_scalars(sink, *ev, ev->name);
    } else if (ev->type == TRACKING_EVENT_RUN_END) {
      write_scalars(sink, *ev, "run_end");
    }
    return 0;
  } catch (const std::exception &e) {
    set_error(err, err_len, std::string("tensorboard: ") + e.what());
    return 1;
  }
}

int tensorboard_flush(void *) { return 0; }

void tensorboard_destroy(void *instance) {
  delete static_cast<TensorBoardSink *>(instance);
}

const TrackingSinkApiV1 kApi = {
    LITNICE_TRACKING_ABI_VERSION, "tensorboard",       &tensorboard_create,
    &tensorboard_on_event,        &tensorboard_flush, &tensorboard_destroy,
};

} // namespace

extern "C" const TrackingSinkApiV1 *litnice_tracking_get_api() { return &kApi; }
