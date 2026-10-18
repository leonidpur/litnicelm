#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace cuda_cublas_plugin {

inline constexpr int kThreadsPerBlock = 256;
inline constexpr int kTileDim = 16;

// Row-wise elementwise kernels: one (row, column) per thread, 32-bit index
// math. Block x runs over the columns of a row and is the row width rounded
// up to a power of two, at most max_x, so narrow rows (a gradient arena seen
// as N x 1) leave no lanes idle; y fills a 256-thread block. Rows beyond
// grid y's 65535 limit spill into grid z.
struct RowsColsLaunch {
  dim3 grid;
  dim3 block;
};

inline RowsColsLaunch rows_cols_launch(int64_t rows, int64_t cols,
                                       unsigned int max_x = 32) {
  unsigned int bx = max_x;
  while (bx > 1 && static_cast<int64_t>(bx / 2) >= cols) {
    bx /= 2;
  }
  const dim3 block(bx, 256 / bx);
  const int64_t x = std::max<int64_t>((cols + block.x - 1) / block.x, 1);
  const int64_t row_blocks =
      std::max<int64_t>((rows + block.y - 1) / block.y, 1);
  const int64_t y = std::min<int64_t>(row_blocks, 65535);
  const int64_t z = (row_blocks + y - 1) / y;
  return {dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y),
               static_cast<unsigned int>(z)),
          block};
}

// This thread's row under rows_cols_launch.
__device__ inline int grid_row() {
  return (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
}

} // namespace cuda_cublas_plugin
