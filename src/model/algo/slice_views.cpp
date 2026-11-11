// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "slice_views.hpp"

#include <litnicelm_utils/checks.hpp>

namespace slice_views {

TensorView batch_seq(const TensorView &slice, int64_t batch_size,
                     int64_t seq_len, const char *who) {
  checker().check_rank(slice, 3, who);
  checker().check_prefix(slice, Shape{batch_size, seq_len}, who);
  if (batch_size == slice.dim(0) && seq_len == slice.dim(1)) {
    return slice;
  }
  return slice.slice(0, 0, batch_size).slice(1, 0, seq_len);
}

TensorView batch_head_seq(const TensorView &slice, int64_t batch_size,
                          int64_t seq_len, const char *who) {
  checker().check_rank(slice, 3, who);
  checker().check_prefix(slice, Shape{batch_size, slice.dim(1), seq_len}, who);
  if (batch_size == slice.dim(0) && seq_len == slice.dim(2)) {
    return slice;
  }
  return slice.slice(0, 0, batch_size).slice(2, 0, seq_len);
}

TensorView batch_seq_square(const TensorView &slice, int64_t batch_size,
                            int64_t seq_len, const char *who) {
  checker().check_rank(slice, 3, who);
  checker().check_prefix(slice, Shape{batch_size, seq_len, seq_len}, who);
  if (batch_size == slice.dim(0) && seq_len == slice.dim(1)) {
    return slice;
  }
  return slice.slice(0, 0, batch_size).slice(1, 0, seq_len).slice(2, 0, seq_len);
}

TensorView head_batch_seq_square(const TensorView &slice, int64_t batch_size,
                                 int64_t seq_len, const char *who) {
  checker().check_rank(slice, 4, who);
  checker().check_prefix(slice, Shape{slice.dim(0), batch_size, seq_len, seq_len},
                         who);
  if (batch_size == slice.dim(1) && seq_len == slice.dim(2)) {
    return slice;
  }
  return slice.slice(1, 0, batch_size).slice(2, 0, seq_len).slice(3, 0, seq_len);
}

} // namespace slice_views
