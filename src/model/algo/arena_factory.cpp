// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "arena_factory.hpp"
#include "named_layout.hpp"

#include <litnicelm_utils/checks.hpp>

void ArenaFactory::append(const std::string &prefix,
                          const std::vector<SliceRequest> &requests,
                          const PrecisionPolicy &precision, uint64_t alignment,
                          std::vector<LayoutSlice> &slices, uint64_t &cursor) {
  for (const SliceRequest &request : requests) {
    const DType dtype = precision.dtype(request.item);
    const std::string name = prefix + request.name;
    const uint64_t bytes = NamedLayout::checked_mul(
        request.shape.numel(), static_cast<uint64_t>(dtype_size(dtype)),
        name.c_str());
    cursor = NamedLayout::align_up(cursor, alignment);
    slices.push_back(LayoutSlice{name, cursor, bytes, dtype});
    cursor += bytes;
  }
}

NodeSlices ArenaFactory::bind(const std::string &prefix,
                              const std::vector<SliceRequest> &requests,
                              const std::vector<LayoutSlice> &layout,
                              void *base, uint64_t arena_bytes,
                              Device device) {
  NodeSlices out;
  for (const SliceRequest &request : requests) {
    const std::string name = prefix + request.name;
    const LayoutSlice *found = nullptr;
    for (const LayoutSlice &slice : layout) {
      if (slice.name == name) {
        found = &slice;
        break;
      }
    }
    checker().check(CheckKind::Layout, found != nullptr, "ArenaFactory",
                    [&] { return "layout has no slice " + name; });
    checker().check_slice(*found, request.shape, arena_bytes, "ArenaFactory");
    out.add(request.name,
            TensorView(device, found->dtype,
                       static_cast<uint8_t *>(base) + found->offset,
                       request.shape));
  }
  return out;
}
