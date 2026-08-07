#pragma once

namespace cuda_cublas_plugin {

inline constexpr int kThreadsPerBlock = 256;
inline constexpr int kTileDim = 16;

} // namespace cuda_cublas_plugin
