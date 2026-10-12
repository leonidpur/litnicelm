#pragma once

// Events logged by the model component (src/model): trainer, checkpoints,
// dataset, inference and diagnostics.
// Names are "model.<event>".

#include <litnicelm_utils/log.hpp>

namespace model_events {

// checkpoint.cpp
inline constexpr EventDef kCheckpointLoadFailed{
    "model.checkpoint_load_failed",
    EventType::Error,
    "Checkpoint load failed:"};
inline constexpr EventDef kCheckpointMismatch{
    "model.checkpoint_mismatch",
    EventType::Error,
    "Checkpoint mismatch:"};
inline constexpr EventDef kCheckpointValidationFailed{
    "model.checkpoint_validation_failed",
    EventType::Error,
    "Checkpoint validation failed:"};
inline constexpr EventDef kCheckpointVersionMismatch{
    "model.checkpoint_version_mismatch",
    EventType::Warning,
    "Checkpoint warning:"};
inline constexpr EventDef kCheckpointSidecarMissing{
    "model.checkpoint_sidecar_missing",
    EventType::Warning,
    "Checkpoint warning: tokenizer fingerprint sidecar missing:"};
inline constexpr EventDef kCheckpointConfVersionTruncated{
    "model.checkpoint_conf_version_truncated",
    EventType::Warning,
    "Checkpoint warning: conf.version is too long, truncating to"};
inline constexpr EventDef kCheckpointLegacyHeader{
    "model.checkpoint_legacy_header",
    EventType::Warning,
    "Checkpoint note:"};

// dataset.cpp
inline constexpr EventDef kDatasetFileSelected{
    "model.dataset_file_selected",
    EventType::Info,
    "[TextDataset] Dataset file to load:"};
inline constexpr EventDef kBatchFetched{
    "model.batch_fetched",
    EventType::Probe,
    "[TextDataset][Fetch]"};

// inference.cpp
inline constexpr EventDef kInferenceStart{
    "model.inference_start",
    EventType::Info,
    "[Inference][Start]"};
inline constexpr EventDef kInferenceEnd{"model.inference_end", EventType::Info, "[Inference][End]"};

// runtime_config.cpp
inline constexpr EventDef kVocabSizeImported{
    "model.vocab_size_imported",
    EventType::Info,
    "[RuntimeConfig] model.vocab_size set to"};

// tensor_probe.cpp
inline constexpr EventDef kProbeTensor{
    "model.probe_tensor",
    EventType::Probe,
    "[TensorProbe][Step complete]"};
inline constexpr EventDef kProbeLoss{
    "model.probe_loss",
    EventType::Probe,
    "[TensorProbe][Step complete] loss"};

// trainer.cpp
inline constexpr EventDef kTrainingRuntimeInitialized{
    "model.training_runtime_initialized",
    EventType::Info,
    "[Trainer] Training runtime initialization done."};
inline constexpr EventDef kTrainingEngineInitialized{
    "model.training_engine_initialized",
    EventType::Info,
    "\n[Trainer] Engine, memory arenas, tensor store, model, optimizer, and dataset are initialized.\n"};
inline constexpr EventDef kTrainingConfigFile{
    "model.training_config_file",
    EventType::Info,
    "[Trainer] Config file:"};

// trainer_validation_utils.cpp
inline constexpr EventDef kDatasetStats{
    "model.dataset_stats",
    EventType::History,
    "\n[Dataset Stats]"};
inline constexpr EventDef kVocabCheckPassed{
    "model.vocab_check_passed",
    EventType::Info,
    "  -> Vocab Check:       Passed"};
inline constexpr EventDef kConfigTokenizerDatasetValidated{
    "model.config_tokenizer_dataset_validated",
    EventType::Info,
    "[TrainerValidationUtils] Config, tokenizer and dataset checked:"};

} // namespace model_events
