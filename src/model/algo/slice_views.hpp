// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

#include <cstdint>

// Leading-prefix views of node slices laid out for the largest call. A
// prefix of the full extent is the slice itself; smaller ones keep the
// slice's strides.
namespace slice_views {

// [B, S, C] storage -> [batch_size, seq_len, C].
TensorView batch_seq(const TensorView &slice, int64_t batch_size,
                     int64_t seq_len, const char *who);
// [B, S, S] storage -> [batch_size, seq_len, seq_len].
TensorView batch_seq_square(const TensorView &slice, int64_t batch_size,
                            int64_t seq_len, const char *who);
// [B, H, S] storage -> [batch_size, H, seq_len].
TensorView batch_head_seq(const TensorView &slice, int64_t batch_size,
                          int64_t seq_len, const char *who);
// [H, B, S, S] storage -> [H, batch_size, seq_len, seq_len].
TensorView head_batch_seq_square(const TensorView &slice, int64_t batch_size,
                                 int64_t seq_len, const char *who);

} // namespace slice_views
