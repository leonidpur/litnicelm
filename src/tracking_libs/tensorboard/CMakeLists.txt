add_library(litnice_tracking_tensorboard SHARED
    tensorboard_tracking_plugin.cpp
)

target_include_directories(litnice_tracking_tensorboard PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(litnice_tracking_tensorboard PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
