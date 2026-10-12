#pragma once

// Events logged by the tracking component (src/tracking).
// Names are "tracking.<event>".

#include <litnicelm_utils/log.hpp>

namespace tracking_events {

// experiment_tracking_sink.cpp
inline constexpr EventDef kTrackingSinkDisabled{
    "tracking.tracking_sink_disabled",
    EventType::Warning,
    "[ExperimentTracking][Warn]"};

} // namespace tracking_events
