#pragma once

// Events logged by the tokenizer component (src/tokenizer): factory, entry
// points, plugins and tokenization.
// Names are "tokenizer.<event>".

#include <litnicelm_utils/log.hpp>

namespace tokenizer_events {

// only_seen_chars_tokenizer_plugin.cpp
inline constexpr EventDef kTokenizerNonAsciiBytes{
    "tokenizer.tokenizer_non_ascii_bytes",
    EventType::Warning,
    "[OnlySeenCharsTokenizer][Warn]"};

// sentencepiece_plugin.cpp
inline constexpr EventDef kSentencePieceTrainingStart{
    "tokenizer.sentencepiece_training_start",
    EventType::Info,
    "[Tokenizer][Start]"};
inline constexpr EventDef kTokenizerError{
    "tokenizer.tokenizer_error",
    EventType::Error,
    "[Tokenizer][Error]"};

// tokenizer_entrypoints.cpp
inline constexpr EventDef kTokenizerTrainingStart{
    "tokenizer.tokenizer_training_start",
    EventType::Info,
    "[Tokenizer training]"};
inline constexpr EventDef kTokenizeStart{
    "tokenizer.tokenize_start",
    EventType::Info,
    "[Tokenizer]"};

// tokenizer_factory.cpp
inline constexpr EventDef kTokenizerPluginSelected{
    "tokenizer.tokenizer_plugin_selected",
    EventType::Info,
    "[TokenizerFactory] plugin selection:"};
inline constexpr EventDef kTokenizerRuntimeSelected{
    "tokenizer.tokenizer_runtime_selected",
    EventType::Info,
    "[TokenizerFactory] runtime selection:"};
inline constexpr EventDef kTokenizerLoaded{
    "tokenizer.tokenizer_loaded",
    EventType::Info,
    "[TokenizerFactory] runtime load:",
    kAnsiGreen};

// tokenizer_utils.cpp
inline constexpr EventDef kTokenizeStreamStart{
    "tokenizer.tokenize_stream_start",
    EventType::Info,
    "[Tokenizer][Start]"};
inline constexpr EventDef kTokenizeProgress{
    "tokenizer.tokenize_progress",
    EventType::Info,
    "[Tokenizer][Progress]"};
inline constexpr EventDef kTokenizeEnd{
    "tokenizer.tokenize_end",
    EventType::Info,
    "[Tokenizer][End]"};

// shared: only_seen_chars_tokenizer_plugin.cpp, sentencepiece_plugin.cpp
inline constexpr EventDef kTokenizerTrainingEnd{
    "tokenizer.tokenizer_training_end",
    EventType::History,
    "[Tokenizer][End]"};

} // namespace tokenizer_events
