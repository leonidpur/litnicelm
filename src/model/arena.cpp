#include "arena.h"

#include <cstdint>
#include <stdexcept>

Arena::Arena(DeviceBackend &backend, Device dev, uint64_t bytes,
             uint32_t alignment)
    : backend_(&backend), dev_(dev), bytes_(bytes), ptr_(nullptr) {
  if (bytes_ == 0) {
    return;
  }
  ptr_ = backend.alloc(bytes_, alignment);
}

Arena::~Arena() {
  if (ptr_ == nullptr) {
    return;
  }
  backend_->free(ptr_);
}

void *Arena::ptr() { return ptr_; }

uint64_t Arena::size_bytes() const { return bytes_; }

Device Arena::device() const { return dev_; }
