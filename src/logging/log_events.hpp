#pragma once

// Catalog of everything the program logs. Components log an EventId plus an
// optional appendix text and optional fields; the logger takes the event's
// type and prefix text from this catalog and routes it by type. The "now:"
// notes mark events not yet logged through the logger.

#include <cstddef>
#include <cstdint>

// What the output is for. The logger routes by type.
enum class EventType : uint8_t {
  Info,     // what the program is doing: stages, loads, saves, ETA
  Warning,
  Error,
  Probe,    // tensor and memory diagnostics
  History,  // the run's record: metrics, checkpoints, decisions
  Count
};

enum class EventId : uint16_t {
  // Startup and configuration
  CliParsed,
  ConfigLoaded,
  EnvOverridesApplied,
  EnvOverridesSkipped,
  EnvOverride,
  CommandOverridesApplied,
  FatalError,

  // Device backend
  BackendLoaded,  // now: device_backend.cpp prints it; that file is also built into the backend plugins

  // Tokenizer
  TokenizerPluginSelected,
  TokenizerRuntimeSelected,
  TokenizerLoaded,
  TokenizerTrainingStart,
  SentencePieceTrainingStart,
  TokenizerTrainingEnd,
  TokenizerNonAsciiBytes,
  TokenizerError,
  TokenizeStart,
  TokenizeStreamStart,
  TokenizeProgress,
  TokenizeEnd,

  // Dataset
  DatasetFileSelected,
  DatasetStats,
  VocabCheckPassed,
  BatchFetched,

  // Trainer setup
  VocabSizeImported,
  ConfigTokenizerDatasetValidated,
  TrainingRuntimeInitialized,
  TrainingEngineInitialized,
  TrainingHyperparams,
  BatchShape,

  // Session: start, resume, end
  FreshInitialization,
  CheckpointNotFound,
  ResumeStart,
  ResumeFailed,
  ResumeSucceeded,
  TrainingStart,        // also: tracking run start
  LrSchedule,
  LrWarmupTooLong,
  EpochStart,           // now: observers on_epoch_start (no text)
  EpochProgress,
  ValidationComplete,
  EpochComplete,        // now: tracking "epoch" metrics
  TrainingEnd,          // also: tracking run end
  NothingToTrain,
  TrainingConfigFile,
  EtaSummary,
  ProfilingSummary,

  // Convergence
  NewBest,
  NoImprovement,
  EarlyStopRequested,
  PersistedStopCleared,
  BestMetricChanged,

  // Checkpoints
  CheckpointSaveSkippedNonFinite,
  CheckpointSaved,                 // also: tracking "SAVE"
  CheckpointSaveFailed,
  BestCheckpointPromoted,
  BestCheckpointCopyFailed,
  CheckpointLoaded,                // now: tracking "LOAD"
  CheckpointLoadFailed,
  CheckpointMismatch,
  CheckpointValidationFailed,
  CheckpointVersionMismatch,
  CheckpointSidecarMissing,
  CheckpointConfVersionTruncated,
  CheckpointLegacyHeader,

  // Probes
  ProbeTensor,
  ProbeLoss,
  ProbeInitTensorsXY,
  ProbeTokEmbPreviewSkipped,

  // Inference
  InferenceStart,
  InferenceEnd,

  // Experiment tracking
  TrackingSinkDisabled,

  Count
};

struct EventDef {
  EventId id;
  const char *name;    // stable machine name, for structured destinations
  EventType type;
  const char *prefix;  // text written before the component's appendix
  const char *color = nullptr;  // ANSI color of the whole line on a console;
                                // unset: Warning yellow, Error red
};

inline constexpr const char *kAnsiRed = "\033[31m";
inline constexpr const char *kAnsiGreen = "\033[32m";
inline constexpr const char *kAnsiYellow = "\033[33m";
inline constexpr const char *kAnsiCyan = "\033[36m";
inline constexpr const char *kAnsiReset = "\033[0m";

inline constexpr EventDef kEventCatalog[] = {
    {EventId::CliParsed, "cli_parsed", EventType::Info, "[Main: Step 1] CLI parsed. Config path:"},
    {EventId::ConfigLoaded, "config_loaded", EventType::Info, "[Main: Step 2] Base YAML loaded."},
    {EventId::EnvOverridesApplied, "env_overrides_applied", EventType::Info, "[Main: Step 3] Applying ENV overrides with prefix:"},
    {EventId::EnvOverridesSkipped, "env_overrides_skipped", EventType::Info, "[Main: Step 3] ENV_PREFIX not set. No ENV overrides applied."},
    {EventId::EnvOverride, "env_override", EventType::Info, "  -> ENV Override:"},
    {EventId::CommandOverridesApplied, "command_overrides_applied", EventType::Info, ""},
    {EventId::FatalError, "fatal_error", EventType::Error, "Fatal error:"},

    {EventId::BackendLoaded, "backend_loaded", EventType::Info, "[DeviceBackend] Loaded backend library:", kAnsiGreen},

    {EventId::TokenizerPluginSelected, "tokenizer_plugin_selected", EventType::Info, "[TokenizerFactory] plugin selection:"},
    {EventId::TokenizerRuntimeSelected, "tokenizer_runtime_selected", EventType::Info, "[TokenizerFactory] runtime selection:"},
    {EventId::TokenizerLoaded, "tokenizer_loaded", EventType::Info, "[TokenizerFactory] runtime load:", kAnsiGreen},
    {EventId::TokenizerTrainingStart, "tokenizer_training_start", EventType::Info, "[Tokenizer training]"},
    {EventId::SentencePieceTrainingStart, "sentencepiece_training_start", EventType::Info, "[Tokenizer][Start]"},
    {EventId::TokenizerTrainingEnd, "tokenizer_training_end", EventType::History, "[Tokenizer][End]"},
    {EventId::TokenizerNonAsciiBytes, "tokenizer_non_ascii_bytes", EventType::Warning, "[OnlySeenCharsTokenizer][Warn]"},
    {EventId::TokenizerError, "tokenizer_error", EventType::Error, "[Tokenizer][Error]"},
    {EventId::TokenizeStart, "tokenize_start", EventType::Info, "[Tokenizer]"},
    {EventId::TokenizeStreamStart, "tokenize_stream_start", EventType::Info, "[Tokenizer][Start]"},
    {EventId::TokenizeProgress, "tokenize_progress", EventType::Info, "[Tokenizer][Progress]"},
    {EventId::TokenizeEnd, "tokenize_end", EventType::Info, "[Tokenizer][End]"},

    {EventId::DatasetFileSelected, "dataset_file_selected", EventType::Info, "[TextDataset] Dataset file to load:"},
    {EventId::DatasetStats, "dataset_stats", EventType::History, "\n[Dataset Stats]"},
    {EventId::VocabCheckPassed, "vocab_check_passed", EventType::Info, "  -> Vocab Check:       Passed"},
    {EventId::BatchFetched, "batch_fetched", EventType::Probe, "[TextDataset][Fetch]"},

    {EventId::VocabSizeImported, "vocab_size_imported", EventType::Info, "[RuntimeConfig] model.vocab_size set to"},
    {EventId::ConfigTokenizerDatasetValidated, "config_tokenizer_dataset_validated", EventType::Info, "[TrainerValidationUtils] Config, tokenizer and dataset checked:"},
    {EventId::TrainingRuntimeInitialized, "training_runtime_initialized", EventType::Info, "[Trainer] Training runtime initialization done."},
    {EventId::TrainingEngineInitialized, "training_engine_initialized", EventType::Info, "\n[Trainer] Engine, memory arenas, tensor store, model, optimizer, and dataset are initialized.\n"},
    {EventId::TrainingHyperparams, "training_hyperparams", EventType::Info, "[ReportingObserver][Start]"},
    {EventId::BatchShape, "batch_shape", EventType::Info, "[ReportingObserver][Batch] Batch->Logit:"},

    {EventId::FreshInitialization, "fresh_initialization", EventType::History, "[TrainingSessionController] Fresh initialization completed for parameters and optimizer state.", kAnsiCyan},
    {EventId::CheckpointNotFound, "checkpoint_not_found", EventType::Info, "[TrainingSessionController] Incremental enabled, but checkpoint not found at:", kAnsiCyan},
    {EventId::ResumeStart, "resume_start", EventType::Info, "[TrainingSessionController] Resuming from", kAnsiCyan},
    {EventId::ResumeFailed, "resume_failed", EventType::Error, "[TrainingSessionController] Error during resume:"},
    {EventId::ResumeSucceeded, "resume_succeeded", EventType::History, "  -> Success. Resumed at Step"},
    {EventId::TrainingStart, "training_start", EventType::Info, "[TrainingSessionController]", kAnsiCyan},
    {EventId::LrSchedule, "lr_schedule", EventType::Info, "[LearningRateSchedule]"},
    {EventId::LrWarmupTooLong, "lr_warmup_too_long", EventType::Warning, "[LearningRateSchedule] Warmup longer than the run:"},
    {EventId::EpochStart, "epoch_start", EventType::Info, "[Epoch start]"},
    {EventId::EpochProgress, "epoch_progress", EventType::Info, "[EtaObserver][Epoch complete]"},
    {EventId::ValidationComplete, "validation_complete", EventType::History, "[TrainingSessionController] Validation", kAnsiCyan},
    {EventId::EpochComplete, "epoch_complete", EventType::History, "[Epoch]"},
    {EventId::TrainingEnd, "training_end", EventType::Info, "[TrainingSessionController]", kAnsiCyan},
    {EventId::NothingToTrain, "nothing_to_train", EventType::Warning, "[TrainingSessionController] Nothing to train:"},
    {EventId::TrainingConfigFile, "training_config_file", EventType::Info, "[Trainer] Config file:"},
    {EventId::EtaSummary, "eta_summary", EventType::Info, "[EtaObserver][End]"},
    {EventId::ProfilingSummary, "profiling_summary", EventType::Info, ""},

    {EventId::NewBest, "new_best", EventType::History, "[ConvergenceMonitor] New best"},
    {EventId::NoImprovement, "no_improvement", EventType::History, "[ConvergenceMonitor] No new best at epoch", kAnsiYellow},
    {EventId::EarlyStopRequested, "early_stop_requested", EventType::History, "[ConvergenceMonitor] Early stop requested:"},
    {EventId::PersistedStopCleared, "persisted_stop_cleared", EventType::Info, "[ConvergenceMonitor] Resumed checkpoint carries a prior early-stop state", kAnsiGreen},
    {EventId::BestMetricChanged, "best_metric_changed", EventType::Warning, "[ConvergenceMonitor] Best is reset:"},

    {EventId::CheckpointSaveSkippedNonFinite, "checkpoint_save_skipped_non_finite", EventType::Warning, "[CheckpointPolicy] Skipping latest checkpoint save because epoch loss is non-finite."},
    {EventId::CheckpointSaved, "checkpoint_saved", EventType::History, "[CheckpointStore] Saved latest:"},
    {EventId::CheckpointSaveFailed, "checkpoint_save_failed", EventType::Error, "[CheckpointStore] Critical: failed to save checkpoint:"},
    {EventId::BestCheckpointPromoted, "best_checkpoint_promoted", EventType::History, "[CheckpointStore]"},
    {EventId::BestCheckpointCopyFailed, "best_checkpoint_copy_failed", EventType::Error, "[CheckpointStore] Failed to copy best checkpoint:"},
    {EventId::CheckpointLoaded, "checkpoint_loaded", EventType::History, "Checkpoint loaded:"},
    {EventId::CheckpointLoadFailed, "checkpoint_load_failed", EventType::Error, "Checkpoint load failed:"},
    {EventId::CheckpointMismatch, "checkpoint_mismatch", EventType::Error, "Checkpoint mismatch:"},
    {EventId::CheckpointValidationFailed, "checkpoint_validation_failed", EventType::Error, "Checkpoint validation failed:"},
    {EventId::CheckpointVersionMismatch, "checkpoint_version_mismatch", EventType::Warning, "Checkpoint warning:"},
    {EventId::CheckpointSidecarMissing, "checkpoint_sidecar_missing", EventType::Warning, "Checkpoint warning: tokenizer fingerprint sidecar missing:"},
    {EventId::CheckpointConfVersionTruncated, "checkpoint_conf_version_truncated", EventType::Warning, "Checkpoint warning: conf.version is too long, truncating to"},
    {EventId::CheckpointLegacyHeader, "checkpoint_legacy_header", EventType::Warning, "Checkpoint note:"},

    {EventId::ProbeTensor, "probe_tensor", EventType::Probe, "[TensorProbe][Step complete]"},
    {EventId::ProbeLoss, "probe_loss", EventType::Probe, "[TensorProbe][Step complete] loss"},
    {EventId::ProbeInitTensorsXY, "probe_init_tensors_xy", EventType::Probe, "[ReportingObserver][Start] Input"},
    {EventId::ProbeTokEmbPreviewSkipped, "probe_tok_emb_preview_skipped", EventType::Probe, "[ReportingObserver][Progress] init_tensors_X_Y: tok_emb preview requires CPU F32."},

    {EventId::InferenceStart, "inference_start", EventType::Info, "[Inference][Start]"},
    {EventId::InferenceEnd, "inference_end", EventType::Info, "[Inference][End]"},

    {EventId::TrackingSinkDisabled, "tracking_sink_disabled", EventType::Warning, "[ExperimentTracking][Warn]"},
};

// The table has one row per event, in enum order.
static_assert(sizeof(kEventCatalog) / sizeof(kEventCatalog[0]) ==
                  static_cast<std::size_t>(EventId::Count),
              "kEventCatalog must have one entry per EventId");

constexpr bool event_catalog_in_enum_order() {
  for (std::size_t i = 0; i < static_cast<std::size_t>(EventId::Count); ++i) {
    if (static_cast<std::size_t>(kEventCatalog[i].id) != i) {
      return false;
    }
  }
  return true;
}
static_assert(event_catalog_in_enum_order(),
              "kEventCatalog rows must follow EventId order");

constexpr const EventDef &event_def(EventId id) {
  return kEventCatalog[static_cast<std::size_t>(id)];
}
