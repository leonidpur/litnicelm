cmake_minimum_required(VERSION 3.16)
project(litnicegpt LANGUAGES CXX)

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
    src/profiling
    src/config
    src/utils
    src/tokenizer
    ${SPM_INCLUDE_DIRS}
)

# Link SentencePiece to the core library
target_link_libraries(litnice_core PRIVATE ${SPM_LIBRARIES})
target_link_libraries(litnice_core PRIVATE dl)
target_compile_definitions(litnice_core PRIVATE LITNICEGPT_HAVE_SENTENCEPIECE=1)

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

# Build main executable using the path-referenced main.cpp
add_executable(litnicegpt ${MAIN_SOURCE})
target_include_directories(litnicegpt PRIVATE
    include
    src/config
    src/utils
    src/model
    src/model/algo
    src/model/memory
    src/observers
    src/profiling
    src/tokenizer
    ${SPM_INCLUDE_DIRS}
)
target_link_libraries(litnicegpt PRIVATE litnice_core)

set(WORKFLOW_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/conf/char_book.yaml" CACHE FILEPATH
    "YAML config used by workflow targets")
set(WORKFLOW_INFER_PROMPT "" CACHE STRING
    "Prompt passed to the infer workflow target")

add_custom_target(tokenizer_training
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicegpt> --tokenizer_trainig --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicegpt>" --tokenizer_trainig --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicegpt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(encode
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicegpt> --tokenize --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicegpt>" --tokenize --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicegpt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(train
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicegpt> --train --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicegpt>" --train --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicegpt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(dry_run
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicegpt> dry_run --config ${WORKFLOW_CONFIG}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicegpt>" dry_run --config "${WORKFLOW_CONFIG}"
    DEPENDS litnicegpt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)

add_custom_target(infer
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "${CMAKE_COMMAND}" -E echo "$<TARGET_FILE:litnicegpt> infer --config ${WORKFLOW_CONFIG} ${WORKFLOW_INFER_PROMPT}"
    COMMAND "${CMAKE_COMMAND}" -E echo ""
    COMMAND "$<TARGET_FILE:litnicegpt>" infer --config "${WORKFLOW_CONFIG}" "${WORKFLOW_INFER_PROMPT}"
    DEPENDS litnicegpt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL
)
