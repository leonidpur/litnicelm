#pragma once

#include "tensor_factory.hpp"

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
  TensorView ids;      // shape: [B*T, 1] (flat) OR [T,1] if B=1
  TensorView targets;  // shape: same as ids
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
  virtual ~IDataLoader() = default;
  virtual void reset_epoch() = 0;
  virtual bool next(TrainBatch &out) = 0;
  virtual uint64_t steps_per_epoch() const = 0;
};

class TrainingReportSink;

class DatasetBackend {
public:
  virtual ~DatasetBackend() = default;
  virtual void fill_batch(const int32_t *tokens, uint64_t ids_begin,
                          uint64_t tgt_begin, uint64_t block_span,
                          TensorFactory &tensor_factory, Device device,
                          TrainBatch &out) const = 0;
};

class DatasetCPU final : public DatasetBackend {
public:
  void fill_batch(const int32_t *tokens, uint64_t ids_begin, uint64_t tgt_begin,
                  uint64_t block_span, TensorFactory &tensor_factory, Device device,
                  TrainBatch &out) const override;
};

class DatasetGPU final : public DatasetBackend {
public:
  void fill_batch(const int32_t *tokens, uint64_t ids_begin, uint64_t tgt_begin,
                  uint64_t block_span, TensorFactory &tensor_factory, Device device,
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

  TextDataset(TensorFactory &tensor_factory, const std::string &dataset_path,
              Device device, uint32_t seq_len,
              uint32_t batch_size, bool shuffle_blocks = false,
              TrainingReportSink *report_sink = nullptr);
  ~TextDataset() override;
  static DatasetHeader read_header_or_throw(const std::string &path);

  void reset_epoch() override;
  bool next(TrainBatch &out) override;
  uint64_t steps_per_epoch() const override { return block_starts_.size(); }

  uint64_t num_tokens() const { return num_tokens_; }
  uint32_t max_token_id() const;
  uint32_t vocab_size() const { return header_.vocab_size; }
  SourceFormat source_format() const { return source_format_; }

private:
  TensorFactory &tensorFactory_;
  Device device_;

  uint32_t seq_len_ = 0;
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

  // For simple optional shuffling by blocks (keeps order within a block).
  std::vector<uint64_t> block_starts_;
  uint32_t block_index_ = 0;

  static bool ends_with(const std::string &s, const std::string &suffix);
  static DatasetHeader decode_header_or_throw_(const uint8_t *bytes, size_t len);
  void build_blocks_();
  void maybe_shuffle_blocks_();
};
