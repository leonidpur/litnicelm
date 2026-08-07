#pragma once

#include "tensor_factory.hpp"
#include "training_observer.hpp"

#include <fcntl.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <memory>
#include <vector>

struct TrainBatch {
  TensorView ids;      // physical shape: [B*S, 1], logical shape: [B, S]
  TensorView targets;  // physical shape: [B*S, 1], logical shape: [B, S]

  int64_t token_count() const { return static_cast<int64_t>(ids.numel()); }
  int64_t batch_size() const {
    return ids.rank() >= 1 ? ids.dim(0) : static_cast<int64_t>(ids.numel());
  }
  int64_t seq_len() const {
    return ids.rank() >= 2 ? ids.dim(1) : static_cast<int64_t>(ids.numel());
  }
};

struct DatasetHeader {
  uint32_t magic = 0;
  uint32_t version = 0;
  uint32_t vocab_size = 0;
  // Derived at load time from payload size; not stored in the on-disk header.
  uint64_t num_tokens = 0;
};

class IDataLoader {
public:
  explicit IDataLoader(class ITrainingObserver *load_observer = nullptr)
      : load_observer_(load_observer) {}
  virtual ~IDataLoader() = default;
  virtual void reset_epoch() = 0;
  bool next(TrainBatch &out, uint64_t global_step = 0) {
    if (load_observer_ != nullptr) {
      load_observer_->on_batch_load_start(global_step);
    }
    const bool has_batch = next_impl(out);
    if (load_observer_ != nullptr) {
      load_observer_->on_batch_load_end(global_step, has_batch);
    }
    return has_batch;
  }
  virtual uint64_t steps_per_epoch() const = 0;

protected:
  virtual bool next_impl(TrainBatch &out) = 0;

private:
  class ITrainingObserver *load_observer_ = nullptr;
};

class TrainingReportSink;
class DeviceBackend;

class DatasetBackend {
public:
  virtual ~DatasetBackend() = default;
  virtual void fill_batch(const int32_t *tokens,
                          const std::vector<uint64_t> &window_starts,
                          uint64_t first_window_index, uint32_t batch_size,
                          uint32_t seq_len,
                          TensorFactory &tensor_factory,
                          DeviceBackend &device_backend, Device device,
                          TrainBatch &out) const = 0;
};

class DatasetCPU final : public DatasetBackend {
public:
  void fill_batch(const int32_t *tokens,
                  const std::vector<uint64_t> &window_starts,
                  uint64_t first_window_index, uint32_t batch_size,
                  uint32_t seq_len,
                  TensorFactory &tensor_factory,
                  DeviceBackend &device_backend, Device device,
                  TrainBatch &out) const override;
};

class DatasetGPU final : public DatasetBackend {
public:
  void fill_batch(const int32_t *tokens,
                  const std::vector<uint64_t> &window_starts,
                  uint64_t first_window_index, uint32_t batch_size,
                  uint32_t seq_len,
                  TensorFactory &tensor_factory,
                  DeviceBackend &device_backend, Device device,
                  TrainBatch &out) const override;
};

// TextDataset:
// - reads a tokenized .bin dataset with DatasetHeader
// - validates header and payload
// - emits sliding windows for next-token prediction
class TextDataset final : public IDataLoader {
public:
  enum class SourceFormat : uint8_t {
    TOKEN_U32 = 1,
  };

  TextDataset(TensorFactory &tensor_factory, DeviceBackend &device_backend,
              const std::string &dataset_path, Device device, uint32_t seq_len,
              uint32_t window_stride, uint32_t batch_size,
              bool shuffle_blocks = false,
              TrainingReportSink *report_sink = nullptr,
              class ITrainingObserver *load_observer = nullptr);
  ~TextDataset() override;
  static DatasetHeader read_header_or_throw(const std::string &path);

  void reset_epoch() override;
  bool next_impl(TrainBatch &out) override;
  uint64_t steps_per_epoch() const override {
    return batch_size_ == 0 ? 0 : block_starts_.size() / batch_size_;
  }

  uint64_t num_tokens() const { return num_tokens_; }
  uint32_t max_token_id() const;
  uint32_t vocab_size() const { return header_.vocab_size; }
  SourceFormat source_format() const { return source_format_; }

private:
  TensorFactory &tensorFactory_;
  DeviceBackend *device_backend_ = nullptr;
  Device device_;

  uint32_t seq_len_ = 0;
  uint32_t window_stride_ = 0;
  uint32_t batch_size_ = 0;
  bool shuffle_blocks_ = false;

  DatasetHeader header_{};
  void *mmap_base_ = nullptr;
  size_t mmap_len_ = 0;
  const int32_t *tokens_ = nullptr;
  uint64_t num_tokens_ = 0;
  uint64_t cursor_ = 0;
  SourceFormat source_format_ = SourceFormat::TOKEN_U32;
  std::unique_ptr<DatasetBackend> backend_;
  TrainingReportSink *report_sink_ = nullptr;
  uint64_t fetch_report_count_ = 0;
  uint64_t fetch_report_limit_ = 1;

  // For simple optional shuffling by sequence windows.
  std::vector<uint64_t> block_starts_;
  uint32_t block_index_ = 0;

  static bool ends_with(const std::string &s, const std::string &suffix);
  static DatasetHeader decode_header_or_throw_(const uint8_t *bytes, size_t len);
  void build_blocks_();
  void maybe_shuffle_blocks_();
};
