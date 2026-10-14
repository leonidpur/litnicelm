#pragma once

// Events logged by the litnicelm_utils component (src/litnicelm_utils).
// Names are "litnicelm_utils.<event>".

#include "litnicelm_utils/log.hpp"

namespace litnicelm_utils_events {

// checks.cpp
inline constexpr EventDef kCheckFailed{"litnicelm_utils.check_failed",
                                       EventType::Error, "[Check]"};

} // namespace litnicelm_utils_events
