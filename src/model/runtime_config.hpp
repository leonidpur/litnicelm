// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>

class Tokenizer;

// A copy of cfg whose model.vocab_size is the loaded tokenizer's actual vocab
// size (the configured tokenizer.target_vocab_size is only the size asked of
// tokenizer training). The model, memory and checkpoints are built from it.
Config with_tokenizer_vocab_size(const Config &cfg, const Tokenizer &tokenizer);
