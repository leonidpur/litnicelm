find_library(OPENBLAS_LIBRARY NAMES openblas)
find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h)

if (NOT OPENBLAS_LIBRARY OR NOT OPENBLAS_INCLUDE_DIR)
  message(STATUS "OpenBLAS not found. Skipping litnice_backend_openblas_plugin target.")
  return()
endif()

add_library(litnice_backend_openblas_plugin SHARED
    ../../model/backend/device_backend.cpp
    openblas_plugin.cpp
)

target_include_directories(litnice_backend_openblas_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/model
    ${CMAKE_SOURCE_DIR}/src/config
    ${CMAKE_SOURCE_DIR}/src/utils
    ${OPENBLAS_INCLUDE_DIR}
)

target_link_libraries(litnice_backend_openblas_plugin PRIVATE
    ${OPENBLAS_LIBRARY}
    dl
)

set_target_properties(litnice_backend_openblas_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
