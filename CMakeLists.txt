cmake_minimum_required(VERSION 3.16)
project(litnicelm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Dependencies ---
find_package(PkgConfig REQUIRED)
# pkg-config will now find these since we installed protobuf-lite and libsentencepiece-dev
pkg_check_modules(SPM REQUIRED sentencepiece)

# --- 2. Source Discovery ---

# Main executable source lives under src/
set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

# Group your other sources
file(GLOB MODEL_SOURCES CONFIGURE_DEPENDS
    src/model/*.cpp
    src/model/algo/*.cpp
    src/model/memory/*.cpp
    src/model/backend/*.cpp
    src/observers/*.cpp
    src/training/*.cpp
    src/tracking/*.cpp
    src/profiling/*.cpp
    src/config/*.cpp
    src/utils/*.cpp
)
list(REMOVE_ITEM MODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/backend/backend_plugin_cpu.cpp
)

file(GLOB TOKENIZER_SOURCES CONFIGURE_DEPENDS
    src/tokenizer/*.cpp
)

# --- 3. Build Targets ---

# Build core library
add_library(litnice_core STATIC 
    ${MODEL_SOURCES} 
    ${TOKENIZER_SOURCES}
)

target_include_directories(litnice_core PRIVATE 
    include 
    src/model 
    src/model/algo
    src/model/memory
    src/observers
    src/training
    src/tracking
    src/profiling
    src/config
    src/utils
    src/tokenizer
    ${SPM_INCLUDE_DIRS}
)

# Git commit and dirty flag of this build, recorded by experiment tracking.
add_custom_target(litnice_build_info
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR}
            -DOUT=${CMAKE_BINARY_DIR}/generated/build_info.hpp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/build_info.hpp
)
add_dependencies(litnice_core litnice_build_info)
target_include_directories(litnice_core PRIVATE ${CMAKE_BINARY_DIR}/generated)

# Link SentencePiece to the core library
target_link_libraries(litnice_core PRIVATE ${SPM_LIBRARIES})
target_link_libraries(litnice_core PRIVATE dl)
target_compile_definitions(litnice_core PRIVATE LITNICELM_HAVE_SENTENCEPIECE=1)

add_library(litnice_backend_cpu SHARED
    src/model/backend/device_backend.cpp
    src/model/backend/backend_plugin_cpu.cpp
)
target_include_directories(litnice_backend_cpu PRIVATE
    include
    src/model
    src/model/algo
    src/model/memory
    src/config
    src/utils
)
target_link_libraries(litnice_backend_cpu PRIVATE dl)
set_target_properties(litnice_backend_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_subdirectory(src/backend_libs/openblas_plugin)
add_subdirectory(src/backend_libs/cuda_cublas_plugin)
add_subdirectory(src/tracking_libs/journal)
add_subdirectory(src/tracking_libs/runs)
add_subdirectory(src/tracking_libs/tensorboard)

# Build main executable using the path-referenced main.cpp
add_executable(litnicelm ${MAIN_SOURCE})
target_include_directories(litnicelm PRIVATE
    include
    src/config
    src/utils
    src/model
    src/model/algo
    src/model/memory
    src/observers
    src/tracking
    src/profiling
    src/tokenizer
    ${SPM_INCLUDE_DIRS}
)
target_link_libraries(litnicelm PRIVATE litnice_core)
# Loaded at runtime from the executable's directory when no tracking sinks are
# configured.
add_dependencies(litnicelm litnice_tracking_journal)

# Op-level tests, run once per available backend plugin: ctest --test-dir build
enable_testing()
add_executable(litnice_rope_test tests/rope_ops_test.cpp)
target_include_directories(litnice_rope_test PRIVATE
    include
    src/config
    src/utils
    src/model
    src/model/algo
    src/model/memory
)
target_link_libraries(litnice_rope_test PRIVATE litnice_core)
foreach(plugin litnice_backend_cpu litnice_backend_openblas_plugin litnice_backend_cublas_plugin)
  if(TARGET ${plugin})
    add_test(NAME rope_ops_${plugin}
             COMMAND litnice_rope_test $<TARGET_FILE:${plugin}>)
  endif()
endforeach()

# Resolved-config YAML dump must load back to the same config.
add_executable(litnice_config_yaml_test tests/config_yaml_roundtrip_test.cpp)
target_include_directories(litnice_config_yaml_test PRIVATE
    include
    src/config
    src/utils
)
target_link_libraries(litnice_config_yaml_test PRIVATE litnice_core)
file(GLOB CHECKED_IN_CONFIGS ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.yaml)
add_test(NAME config_yaml_roundtrip
         COMMAND litnice_config_yaml_test
                 ${CMAKE_CURRENT_BINARY_DIR}/config_yaml_test_tmp
                 ${CHECKED_IN_CONFIGS})

# TensorBoard event encoding against published CRC-32C / protobuf vectors.
add_executable(litnice_tfevents_test tests/tfevents_test.cpp)
target_include_directories(litnice_tfevents_test PRIVATE
    src/tracking_libs/tensorboard
)
add_test(NAME tfevents_encoding COMMAND litnice_tfevents_test)

# Experiment-tracking plugin loading, against fake tracking plugins.
foreach(variant fake fake2 fake_bad_abi)
  add_library(litnice_tracking_${variant} SHARED tests/fake_tracking_plugin.cpp)
  target_include_directories(litnice_tracking_${variant} PRIVATE include)
endforeach()
target_compile_definitions(litnice_tracking_fake2 PRIVATE FAKE_NAME="fake2")
target_compile_definitions(litnice_tracking_fake_bad_abi PRIVATE FAKE_ABI=999u)
add_executable(litnice_tracking_sink_test tests/tracking_sink_test.cpp)
target_include_directories(litnice_tracking_sink_test PRIVATE
    include
    src/config
    src/utils
    src/tracking
)
target_link_libraries(litnice_tracking_sink_test PRIVATE litnice_core)
add_test(NAME tracking_sink
         COMMAND litnice_tracking_sink_test
                 $<TARGET_FILE:litnice_tracking_fake>
                 $<TARGET_FILE:litnice_tracking_fake2>
                 $<TARGET_FILE:litnice_tracking_fake_bad_abi>
                 ${CMAKE_CURRENT_BINARY_DIR}/tracking_sink_test_tmp)

set(WORKFLOW_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/conf/char_book_abcdef_cuda.yaml" CACHE FILEPATH
    "YAML config used by workflow targets")
set(WORKFLOW_INFER_PROMPT "" CACHE STRING
    "Prompt passed to the infer workflow target")

add_custom_target(tokenizer_training
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicelm> --tokenizer_trainig --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicelm>" --tokenizer_trainig --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicelm
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(encode
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicelm> --tokenize --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicelm>" --tokenize --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicelm
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(train
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicelm> --train --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicelm>" --train --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicelm
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(dry_run
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicelm> dry_run --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicelm>" dry_run --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicelm
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(infer
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicelm> infer --config ${WORKFLOW_CONFIG} ${WORKFLOW_INFER_PROMPT}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicelm>" infer --config "${WORKFLOW_CONFIG}" "${WORKFLOW_INFER_PROMPT}"
    DEPENDS litnicelm
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)
