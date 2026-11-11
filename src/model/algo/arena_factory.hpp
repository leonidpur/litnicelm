// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "slice_request.hpp"

#include <litnicelm_utils/layout_slice.hpp>

#include <cstdint>
#include <string>
#include <vector>

// Lays out node SliceRequests inside a temp layout and hands the resulting
// views back to the nodes.
//
// Every request currently gets a slice of its own, so the layout is the one
// the named temp accessors had. SliceLifetime::Op is recorded so that Op
// slices of different nodes can later share memory without touching nodes.
class ArenaFactory {
public:
  // Appends "<prefix><request.name>" for each request at the next aligned
  // offset, in request order; the dtype is precision.dtype(request.item).
  static void append(const std::string &prefix,
                     const std::vector<SliceRequest> &requests,
                     const PrecisionPolicy &precision, uint64_t alignment,
                     std::vector<LayoutSlice> &slices, uint64_t &cursor);

  // The views of those slices in an arena of arena_bytes laid out by
  // `layout` at `base`. Throws (Layout check) when a requested slice is
  // missing or does not hold exactly the requested shape.
  static NodeSlices bind(const std::string &prefix,
                         const std::vector<SliceRequest> &requests,
                         const std::vector<LayoutSlice> &layout, void *base,
                         uint64_t arena_bytes, Device device);
};
