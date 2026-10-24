#pragma once

#include "checks.hpp"
#include "tensor_view_struct.h"

#include <cstddef>
#include <cstdint>

// Walking a strided view as blocks: the trailing suffix_rank axes form a
// block (a row for 1, a matrix for 2, an element for 0), and the leading
// axes index the blocks in row-major order.
namespace view_geometry {

// How many blocks the leading ndim - suffix_rank axes hold (1 when none).
inline uint64_t prefix_count(const TensorViewStruct &v, size_t suffix_rank) {
  checker().check(CheckKind::Layout, suffix_rank <= v.ndim, "view_geometry",
                  "suffix rank exceeds tensor rank");
  uint64_t count = 1;
  for (size_t axis = 0; axis + suffix_rank < v.ndim; ++axis) {
    count *= static_cast<uint64_t>(v.dims[axis]);
  }
  return count;
}

// Byte offset from v.ptr to the start of block `prefix`.
inline int64_t prefix_byte_offset(const TensorViewStruct &v, uint64_t prefix,
                                  size_t suffix_rank) {
  checker().check(CheckKind::Layout, suffix_rank <= v.ndim, "view_geometry",
                  "suffix rank exceeds tensor rank");
  int64_t offset = 0;
  for (size_t axis = v.ndim - suffix_rank; axis-- > 0;) {
    const int64_t dim = v.dims[axis];
    checker().check(CheckKind::Layout, dim > 0, "view_geometry",
                    "non-positive dimension");
    offset += static_cast<int64_t>(prefix % static_cast<uint64_t>(dim)) *
              v.strides_bytes[axis];
    prefix /= static_cast<uint64_t>(dim);
  }
  return offset;
}

} // namespace view_geometry
