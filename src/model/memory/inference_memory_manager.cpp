#include "inference_memory_manager.hpp"

#include "adam_state_store.hpp"
#include "backend/device_backend.hpp"

#include <stdexcept>

InferenceMemoryManager::InferenceMemoryManager(
    const Config &cfg, const ModelAlgoFactory &algo,
    DeviceBackend &backend, const PrecisionPolicy &precision)
    : cfg_(cfg),
      backend_(backend),
      precision_(precision),
      param_layout_(NamedLayout::build_param_layout(
          cfg, algo, precision_.dtype(ItemType::Parameter))),
      temp_layout_(NamedLayout::build_inference_temp_layout(
          cfg, algo, precision_)) {
  const uint64_t param_bytes = param_layout_.total_bytes();
  // Checkpoints carry AdamW m and v; inference loads them into this arena.
  const uint64_t adam_bytes = AdamStateStore::arena_bytes(NamedLayout::build_param_layout(
      cfg, algo, precision_.dtype(ItemType::OptimizerState)));
  const uint64_t temp_bytes = temp_layout_.total_bytes();
  const Device device = backend_.device();

  param_arena_ = std::make_unique<Arena>(backend_, device, param_bytes,
                                         cfg_.memory.alignment_bytes);
  adam_arena_ = std::make_unique<Arena>(backend_, device, adam_bytes,
                                        cfg_.memory.alignment_bytes);
  temp_arena_ = std::make_unique<Arena>(backend_, device, temp_bytes,
                                        cfg_.memory.alignment_bytes);

  data_view_ = ArenaView{param_arena_->ptr(), param_arena_->size_bytes(),
                         device};
  adam_view_ = AdamStateView{adam_arena_->ptr(), adam_arena_->size_bytes(),
                             device};

  tensor_store_ = std::make_unique<TensorStore>(
      cfg_, algo, param_layout_, data_view_.base, data_view_.bytes, data_view_.device,
      temp_layout_, temp_arena_->ptr(), temp_arena_->size_bytes(),
      TensorStore::TempLayoutKind::Inference);
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

TensorStore &InferenceMemoryManager::tensor_store() const {
  if (tensor_store_ == nullptr) {
    throw std::runtime_error("InferenceMemoryManager: tensor_store is null");
  }
  return *tensor_store_;
}
