#include "training_memory_manager.hpp"

#include "backend/device_backend.hpp"
#include "training_session_controller.hpp"

#include <stdexcept>

TrainingMemoryManager::TrainingMemoryManager(const Config &cfg,
                                             DeviceBackend &backend,
                                             TrainingSessionController &session_controller)
    : cfg_(cfg),
      backend_(backend),
      param_layout_(NamedLayout::build_param_layout(cfg)),
      temp_layout_(NamedLayout::build_training_temp_layout(cfg)) {
  session_controller.init_config_ready(cfg_, param_layout_, temp_layout_);

  const uint64_t param_bytes = param_layout_.total_bytes();
  const uint64_t temp_bytes = temp_layout_.total_bytes();
  const Device device = backend_.device();

  memory_usage_.device_before_alloc = backend_.memory_info();
  param_arena_ = std::make_unique<Arena>(backend_, device, param_bytes,
                                         cfg_.memory.alignment_bytes);
  grad_arena_ = std::make_unique<Arena>(backend_, device, param_bytes,
                                        cfg_.memory.alignment_bytes);
  adam_arena_ = std::make_unique<Arena>(backend_, device, param_bytes * 2,
                                        cfg_.memory.alignment_bytes);
  temp_arena_ = std::make_unique<Arena>(backend_, device, temp_bytes,
                                        cfg_.memory.alignment_bytes);

  data_view_ = ArenaView{param_arena_->ptr(), param_arena_->size_bytes(),
                         device};
  grad_view_ = ArenaView{grad_arena_->ptr(), grad_arena_->size_bytes(),
                         device};
  adam_view_ = AdamStateView{adam_arena_->ptr(), adam_arena_->size_bytes(),
                             device};
  memory_usage_.param_bytes = data_view_.bytes;
  memory_usage_.grad_bytes = grad_view_.bytes;
  memory_usage_.adam_bytes = adam_view_.bytes;
  memory_usage_.temp_bytes = temp_arena_->size_bytes();
  memory_usage_.total_managed_bytes =
      memory_usage_.param_bytes + memory_usage_.grad_bytes +
      memory_usage_.adam_bytes + memory_usage_.temp_bytes;
  memory_usage_.device_after_alloc = backend_.memory_info();
  session_controller.init_topology_ready(param_layout_, data_view_.base,
                                         data_view_.bytes, grad_view_.base,
                                         grad_view_.bytes, adam_view_.base,
                                         adam_view_.bytes, temp_arena_->ptr(),
                                         temp_arena_->size_bytes());
  session_controller.memory_usage_ready(memory_usage_);

  tensor_store_ = std::make_unique<TensorStore>(
      cfg_, param_layout_, data_view_.base, data_view_.bytes, data_view_.device,
      temp_layout_, temp_arena_->ptr(), temp_arena_->size_bytes(),
      TensorStore::TempLayoutKind::Training);
  session_controller.tensor_store_topology_ready(cfg_, *tensor_store_);
  gradient_store_ = std::make_unique<GradientStore>(
      cfg_, param_layout_, data_view_.base, data_view_.bytes, grad_view_);
  adam_state_store_ = std::make_unique<AdamStateStore>(
      cfg_, param_layout_, data_view_.base, data_view_.bytes, adam_view_);
}

const NamedLayout &TrainingMemoryManager::param_layout() const {
  return param_layout_;
}

const NamedLayout &TrainingMemoryManager::temp_layout() const {
  return temp_layout_;
}

const ArenaView &TrainingMemoryManager::data_arena() const { return data_view_; }

const ArenaView &TrainingMemoryManager::grad_arena() const { return grad_view_; }

const AdamStateView &TrainingMemoryManager::adam_state() const {
  return adam_view_;
}

const TrainingMemoryUsage &TrainingMemoryManager::memory_usage() const {
  return memory_usage_;
}

void *TrainingMemoryManager::temp_base() const { return temp_arena_->ptr(); }

uint64_t TrainingMemoryManager::temp_bytes() const {
  return temp_arena_->size_bytes();
}

TensorStore &TrainingMemoryManager::tensor_store() const {
  if (tensor_store_ == nullptr) {
    throw std::runtime_error("TrainingMemoryManager: tensor_store is null");
  }
  return *tensor_store_;
}

GradientStore &TrainingMemoryManager::gradient_store() const {
  if (gradient_store_ == nullptr) {
    throw std::runtime_error("TrainingMemoryManager: gradient_store is null");
  }
  return *gradient_store_;
}

AdamStateStore &TrainingMemoryManager::adam_state_store() const {
  if (adam_state_store_ == nullptr) {
    throw std::runtime_error(
        "TrainingMemoryManager: adam_state_store is null");
  }
  return *adam_state_store_;
}
