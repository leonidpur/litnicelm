#pragma once

// Events logged by the litnicelm_utils component (src/litnicelm_utils).
// Names are "litnicelm_utils.<event>".

#include "litnicelm_utils/log.hpp"

namespace litnicelm_utils_events {

// validation.cpp
inline constexpr EventDef kValidationFailed{"litnicelm_utils.validation_failed",
                                            EventType::Error, "[Validator]"};

} // namespace litnicelm_utils_events
