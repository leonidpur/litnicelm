add_library(litnice_tracking_journal SHARED
    journal_tracking_plugin.cpp
)

target_include_directories(litnice_tracking_journal PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/litnicelm_utils/include
)

set_target_properties(litnice_tracking_journal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
