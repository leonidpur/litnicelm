#pragma once

// Events logged by the session controller's components
// (src/training_session_controller_components): convergence, checkpoints,
// learning-rate schedule.
// Names are "training_session_controller_components.<event>".

#include <litnicelm_utils/log.hpp>

namespace training_session_controller_components_events {

// checkpoint_policy.cpp
inline constexpr EventDef kCheckpointSaveSkippedNonFinite{
    "training_session_controller_components.checkpoint_save_skipped_non_finite",
    EventType::Warning,
    "[CheckpointPolicy] Skipping latest checkpoint save because epoch loss is non-finite."};

// checkpoint_store.cpp
inline constexpr EventDef kCheckpointSaved{
    "training_session_controller_components.checkpoint_saved",
    EventType::History,
    "[CheckpointStore] Saved latest:"};
inline constexpr EventDef kCheckpointSaveFailed{
    "training_session_controller_components.checkpoint_save_failed",
    EventType::Error,
    "[CheckpointStore] Critical: failed to save checkpoint:"};
inline constexpr EventDef kBestCheckpointPromoted{
    "training_session_controller_components.best_checkpoint_promoted",
    EventType::History,
    "[CheckpointStore]"};
inline constexpr EventDef kBestCheckpointCopyFailed{
    "training_session_controller_components.best_checkpoint_copy_failed",
    EventType::Error,
    "[CheckpointStore] Failed to copy best checkpoint:"};

// convergence_monitor.cpp
inline constexpr EventDef kNewBest{
    "training_session_controller_components.new_best",
    EventType::History,
    "[ConvergenceMonitor] New best"};
inline constexpr EventDef kNoImprovement{
    "training_session_controller_components.no_improvement",
    EventType::History,
    "[ConvergenceMonitor] No new best at epoch",
    kAnsiYellow};
inline constexpr EventDef kEarlyStopRequested{
    "training_session_controller_components.early_stop_requested",
    EventType::History,
    "[ConvergenceMonitor] Early stop requested:"};
inline constexpr EventDef kPersistedStopCleared{
    "training_session_controller_components.persisted_stop_cleared",
    EventType::Info,
    "[ConvergenceMonitor] Resumed checkpoint carries a prior early-stop state",
    kAnsiGreen};
inline constexpr EventDef kBestMetricChanged{
    "training_session_controller_components.best_metric_changed",
    EventType::Warning,
    "[ConvergenceMonitor] Best is reset:"};

// learning_rate_schedule.cpp
inline constexpr EventDef kLrSchedule{
    "training_session_controller_components.lr_schedule",
    EventType::Info,
    "[LearningRateSchedule]"};
inline constexpr EventDef kLrWarmupTooLong{
    "training_session_controller_components.lr_warmup_too_long",
    EventType::Warning,
    "[LearningRateSchedule] Warmup longer than the run:"};

} // namespace training_session_controller_components_events
