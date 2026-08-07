#pragma once

#include <config.hpp>

#include <cstdint>
#include <memory>
#include <string>

class DeviceBackend {
public:
  virtual ~DeviceBackend() = default;
  virtual void *alloc(uint64_t bytes, uint32_t alignment) = 0;
  virtual void free(void *ptr) = 0;
  virtual void copy_host2device(void *dst, const void *src, uint64_t bytes) = 0;
  virtual void copy_device2host(void *dst, const void *src, uint64_t bytes) = 0;
  virtual bool is_file2device_read_supported() const = 0;
  virtual void read_file2device(const std::string &path, void *dst,
                                uint64_t size, uint64_t file_offset) = 0;
};

class CpuBackend final : public DeviceBackend {
public:
  CpuBackend() = default;
  void *alloc(uint64_t bytes, uint32_t alignment) override;
  void free(void *ptr) override;
  void copy_host2device(void *dst, const void *src, uint64_t bytes) override;
  void copy_device2host(void *dst, const void *src, uint64_t bytes) override;
  bool is_file2device_read_supported() const override;
  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override;
};

class CudaBackend final : public DeviceBackend {
public:
  CudaBackend() = default;
  void *alloc(uint64_t bytes, uint32_t alignment) override;
  void free(void *ptr) override;
  void copy_host2device(void *dst, const void *src, uint64_t bytes) override;
  void copy_device2host(void *dst, const void *src, uint64_t bytes) override;
  bool is_file2device_read_supported() const override;
  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override;
};

std::unique_ptr<DeviceBackend> make_device_backend(Device device);
