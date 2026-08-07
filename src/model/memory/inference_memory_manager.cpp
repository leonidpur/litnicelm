#include "inference_memory_manager.hpp"

#include "backend/device_backend.hpp"

#include <stdexcept>

InferenceMemoryManager::InferenceMemoryManager(const Config &cfg,
                                               DeviceBackend &backend)
    : cfg_(cfg),
      backend_(backend),
      param_layout_(NamedLayout::build_param_layout(cfg)),
      temp_layout_(NamedLayout::build_inference_temp_layout(cfg)) {
  const uint64_t param_bytes = param_layout_.total_bytes();
  const uint64_t temp_bytes = temp_layout_.total_bytes();
  const Device device = backend_.device();

  param_arena_ = std::make_unique<Arena>(backend_, device, param_bytes,
                                         cfg_.memory.alignment_bytes);
  adam_arena_ = std::make_unique<Arena>(backend_, device, param_bytes * 2,
                                        cfg_.memory.alignment_bytes);
  temp_arena_ = std::make_unique<Arena>(backend_, device, temp_bytes,
                                        cfg_.memory.alignment_bytes);

  data_view_ = ArenaView{param_arena_->ptr(), param_arena_->size_bytes(),
                         device};
  adam_view_ = AdamStateView{adam_arena_->ptr(), adam_arena_->size_bytes(),
                             device};

  tensor_factory_ = std::make_unique<TensorFactory>(
      cfg_, param_layout_, data_view_.base, data_view_.bytes, data_view_.device,
      temp_layout_, temp_arena_->ptr(), temp_arena_->size_bytes(),
      TensorFactory::TempLayoutKind::Inference);
}

const NamedLayout &InferenceMemoryManager::param_layout() const {
  return param_layout_;
}

const NamedLayout &InferenceMemoryManager::temp_layout() const {
  return temp_layout_;
}

const ArenaView &InferenceMemoryManager::data_arena() const { return data_view_; }

const AdamStateView &InferenceMemoryManager::adam_state() const {
  return adam_view_;
}

void *InferenceMemoryManager::temp_base() const { return temp_arena_->ptr(); }

uint64_t InferenceMemoryManager::temp_bytes() const {
  return temp_arena_->size_bytes();
}

TensorFactory &InferenceMemoryManager::tensor_factory() const {
  if (tensor_factory_ == nullptr) {
    throw std::runtime_error("InferenceMemoryManager: tensor_factory is null");
  }
  return *tensor_factory_;
}
