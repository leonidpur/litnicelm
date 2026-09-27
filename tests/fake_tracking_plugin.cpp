// Test double for ExperimentTrackingSink. Built as several libraries:
//   FAKE_NAME      sink name (options arrive as tracking.<FAKE_NAME>.*)
//   FAKE_ABI       ABI version it reports (a wrong one tests the check)
// Options: log (file receiving one line per option and event),
//          fail_create=1, fail_on=N (the Nth event and later fail).

#include <tracking_plugin_api.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#ifndef FAKE_NAME
#define FAKE_NAME "fake"
#endif
#ifndef FAKE_ABI
#define FAKE_ABI LITNICE_TRACKING_ABI_VERSION
#endif

namespace {

struct FakeSink {
  std::string log;
  long fail_on = 0;
  long events = 0;
};

void append(const std::string &path, const std::string &line) {
  if (!path.empty()) {
    std::ofstream(path, std::ios::app) << line << "\n";
  }
}

void *fake_create(const TrackingField *options, uint32_t n, char *err,
                  uint32_t err_len) {
  auto *sink = new FakeSink();
  bool fail_create = false;
  for (uint32_t i = 0; i < n; ++i) {
    const std::string key = options[i].key;
    const std::string value = options[i].str;
    if (key == "log") {
      sink->log = value;
    } else if (key == "fail_on") {
      sink->fail_on = std::strtol(value.c_str(), nullptr, 10);
    } else if (key == "fail_create") {
      fail_create = value == "1";
    }
  }
  for (uint32_t i = 0; i < n; ++i) {
    append(sink->log, std::string("OPT ") + options[i].key);
  }
  if (fail_create) {
    std::snprintf(err, err_len, "%s", "requested create failure");
    delete sink;
    return nullptr;
  }
  return sink;
}

int fake_on_event(void *instance, const TrackingEvent *ev, char *err,
                  uint32_t err_len) {
  auto *sink = static_cast<FakeSink *>(instance);
  sink->events += 1;
  if (sink->fail_on > 0 && sink->events >= sink->fail_on) {
    std::snprintf(err, err_len, "%s", "requested event failure");
    return 1;
  }
  append(sink->log, std::string("EVENT ") + ev->name + " fields=" +
                        std::to_string(ev->n_fields));
  return 0;
}

int fake_flush(void *) { return 0; }

void fake_destroy(void *instance) { delete static_cast<FakeSink *>(instance); }

const TrackingSinkApiV1 kApi = {FAKE_ABI,       FAKE_NAME,  &fake_create,
                                &fake_on_event, &fake_flush, &fake_destroy};

} // namespace

extern "C" const TrackingSinkApiV1 *litnice_tracking_get_api() { return &kApi; }
