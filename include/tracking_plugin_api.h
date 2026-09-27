#pragma once

// C ABI between the core's ExperimentTrackingSink and experiment-tracking
// plugins (journal, runs, TensorBoard, ...). A plugin is a shared library
// exporting litnice_tracking_get_api().
//
// Events are generic: a type, a name and a list of typed key/value fields.
// New metrics or metadata are new field keys, not ABI changes; plugins ignore
// keys they do not know.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LITNICE_TRACKING_ABI_VERSION 1u

enum TrackingFieldKind {
  TRACKING_FIELD_I64 = 0,
  TRACKING_FIELD_F64 = 1,
  TRACKING_FIELD_STR = 2,
  TRACKING_FIELD_BOOL = 3, // value in i64: 0 or 1
};

typedef struct TrackingField {
  const char *key;
  uint32_t kind;
  int64_t i64;
  double f64;
  const char *str;
} TrackingField;

enum TrackingEventType {
  // A training or dry run begins. name: "TRAIN_RUN" | "DRY_RUN".
  TRACKING_EVENT_RUN_START = 1,
  // A one-shot operation finished. name: e.g. "TOKENIZER_GEN".
  TRACKING_EVENT_OPERATION = 2,
  // Scalar metrics at (step, epoch), e.g. train_loss.
  TRACKING_EVENT_METRICS = 3,
  // Checkpoint saved or loaded. name: "SAVE" | "LOAD".
  TRACKING_EVENT_CHECKPOINT = 4,
  // A run finished. name matches its RUN_START.
  TRACKING_EVENT_RUN_END = 5,
};

typedef struct TrackingEvent {
  uint32_t type;
  const char *name;
  const char *run_id;
  uint64_t step;
  uint32_t epoch;
  int64_t unix_ms;
  const TrackingField *fields;
  uint32_t n_fields;
} TrackingEvent;

typedef struct TrackingSinkApiV1 {
  uint32_t abi_version;
  // Short name; the plugin receives the config keys tracking.<name>.* as
  // options (without the prefix), plus core-provided context.* options.
  const char *name;
  // Returns an instance, or null with a message in err.
  void *(*create)(const TrackingField *options, uint32_t n_options, char *err,
                  uint32_t err_len);
  // Returns 0 on success. Must not throw across the ABI.
  int (*on_event)(void *instance, const TrackingEvent *event, char *err,
                  uint32_t err_len);
  int (*flush)(void *instance);
  void (*destroy)(void *instance);
} TrackingSinkApiV1;

typedef const TrackingSinkApiV1 *(*TrackingGetApiFn)(void);

#ifdef __cplusplus
}
#endif
