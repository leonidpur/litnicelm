#include "device_backend.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {
bool is_power_of_two(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

struct AllocationHeader {
  void *base = nullptr;
};
}

void *CpuBackend::alloc(uint64_t bytes, uint32_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument("CpuBackend::alloc: invalid alignment");
  }

  const uint64_t extra =
      static_cast<uint64_t>(alignment - 1) + sizeof(AllocationHeader);
  void *base = std::malloc(static_cast<size_t>(bytes + extra));
  if (base == nullptr) {
    throw std::runtime_error("CpuBackend::alloc failed");
  }

  const auto raw =
      reinterpret_cast<std::uintptr_t>(base) + sizeof(AllocationHeader);
  const auto aligned =
      (raw + (alignment - 1)) & ~static_cast<std::uintptr_t>(alignment - 1);
  auto *header =
      reinterpret_cast<AllocationHeader *>(aligned - sizeof(AllocationHeader));
  header->base = base;
  return reinterpret_cast<void *>(aligned);
}

void CpuBackend::free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto *header = reinterpret_cast<AllocationHeader *>(
      reinterpret_cast<std::uintptr_t>(ptr) - sizeof(AllocationHeader));
  std::free(header->base);
}

void CpuBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("CpuBackend::copy_host2device: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void CpuBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("CpuBackend::copy_device2host: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

bool CpuBackend::is_file2device_read_supported() const { return true; }

void CpuBackend::read_file2device(const std::string &path, void *dst,
                                  uint64_t size, uint64_t file_offset) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    throw std::invalid_argument("CpuBackend::read_file2device: null destination");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("CpuBackend::read_file2device: failed to open file");
  }
  in.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
  if (!in) {
    throw std::runtime_error("CpuBackend::read_file2device: failed to seek file");
  }
  in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(size));
  if (in.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("CpuBackend::read_file2device: short read");
  }
}

void *CudaBackend::alloc(uint64_t bytes, uint32_t alignment) {
  (void)bytes;
  (void)alignment;
  throw std::runtime_error("CudaBackend::alloc: GPU backend is not implemented yet");
}

void CudaBackend::free(void *ptr) {
  (void)ptr;
  throw std::runtime_error("CudaBackend::free: GPU backend is not implemented yet");
}

void CudaBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  throw std::runtime_error(
      "CudaBackend::copy_host2device: GPU backend is not implemented yet");
}

void CudaBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  throw std::runtime_error(
      "CudaBackend::copy_device2host: GPU backend is not implemented yet");
}

bool CudaBackend::is_file2device_read_supported() const { return false; }

void CudaBackend::read_file2device(const std::string &path, void *dst,
                                   uint64_t size, uint64_t file_offset) {
  (void)path;
  (void)dst;
  (void)size;
  (void)file_offset;
  throw std::runtime_error(
      "CudaBackend::read_file2device: GPU backend is not implemented yet");
}

std::unique_ptr<DeviceBackend> make_device_backend(Device device) {
  switch (device) {
  case Device::CPU:
    return std::make_unique<CpuBackend>();
  case Device::GPU:
    return std::make_unique<CudaBackend>();
  default:
    throw std::runtime_error("make_device_backend: unsupported device");
  }
}
