#pragma once

// Events logged by the observers component (src/observers): the session
// controller and the reporting, ETA and profiling observers. Names are
// "observers.<event>".

#include <litnicelm_utils/log.hpp>

namespace observers_events {

// TrainingSessionController
inline constexpr EventDef kFreshInitialization{
    "observers.fresh_initialization",
    EventType::History,
    "[TrainingSessionController] Fresh initialization completed for parameters and optimizer state.",
    kAnsiCyan};
inline constexpr EventDef kCheckpointNotFound{
    "observers.checkpoint_not_found",
    EventType::Info,
    "[TrainingSessionController] Incremental enabled, but checkpoint not found at:",
    kAnsiCyan};
inline constexpr EventDef kResumeStart{
    "observers.resume_start",
    EventType::Info,
    "[TrainingSessionController] Resuming from",
    kAnsiCyan};
inline constexpr EventDef kResumeFailed{
    "observers.resume_failed",
    EventType::Error,
    "[TrainingSessionController] Error during resume:"};
inline constexpr EventDef kResumeSucceeded{
    "observers.resume_succeeded",
    EventType::History,
    "  -> Success. Resumed at Step"};
inline constexpr EventDef kTrainingStart{
    "observers.training_start",
    EventType::Info,
    "[TrainingSessionController]",
    kAnsiCyan};
inline constexpr EventDef kValidationComplete{
    "observers.validation_complete",
    EventType::History,
    "[TrainingSessionController] Validation",
    kAnsiCyan};
inline constexpr EventDef kTrainingEnd{
    "observers.training_end",
    EventType::Info,
    "[TrainingSessionController]",
    kAnsiCyan};
inline constexpr EventDef kNothingToTrain{
    "observers.nothing_to_train",
    EventType::Warning,
    "[TrainingSessionController] Nothing to train:"};

// ReportingObserver
inline constexpr EventDef kTrainingHyperparams{
    "observers.training_hyperparams",
    EventType::Info,
    "[ReportingObserver][Start]"};
inline constexpr EventDef kBatchShape{
    "observers.batch_shape",
    EventType::Info,
    "[ReportingObserver][Batch] Batch->Logit:"};
inline constexpr EventDef kProbeInitTensorsXY{
    "observers.probe_init_tensors_xy",
    EventType::Probe,
    "[ReportingObserver][Start] Input"};
inline constexpr EventDef kProbeTokEmbPreviewSkipped{
    "observers.probe_tok_emb_preview_skipped",
    EventType::Probe,
    "[ReportingObserver][Progress] init_tensors_X_Y: tok_emb preview requires CPU F32."};

// EtaObserver
inline constexpr EventDef kEpochProgress{
    "observers.epoch_progress",
    EventType::Info,
    "[EtaObserver][Epoch complete]"};
inline constexpr EventDef kEtaSummary{
    "observers.eta_summary",
    EventType::Info,
    "[EtaObserver][End]"};

// ProfilingObserver
inline constexpr EventDef kProfilingSummary{"observers.profiling_summary", EventType::Info, ""};

} // namespace observers_events
