#include "dataset.hpp"
#include "backend/device_backend.hpp"
#include "training_report_sink.hpp"
#include <config.hpp>
#include <utils/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {
constexpr uint32_t kDatasetMagic = 0x4C4E4750u; // "LNGP"
constexpr uint32_t kDatasetVersion = 1u;
constexpr size_t kDatasetHeaderBytes = sizeof(uint32_t) * 3;
} // namespace

void DatasetCPU::fill_batch(const int32_t *tokens,
                            const std::vector<uint64_t> &window_starts,
                            uint64_t first_window_index, uint32_t batch_size,
                            uint32_t seq_len,
                            TensorStore &tensor_store,
                            DeviceBackend &device_backend, Device device,
                            TrainBatch &out) const {
  (void)device_backend;
  (void)device;
  TensorView ids_t = tensor_store.temp_ds_ids();
  TensorView tgt_t = tensor_store.temp_ds_targets();

  for (uint32_t b = 0; b < batch_size; ++b) {
    const uint64_t start = window_starts.at(first_window_index + b);
    auto *ids_row = reinterpret_cast<int32_t *>(
        reinterpret_cast<uint8_t *>(ids_t.data()) + b * ids_t.stride_bytes(0));
    auto *tgt_row = reinterpret_cast<int32_t *>(
        reinterpret_cast<uint8_t *>(tgt_t.data()) + b * tgt_t.stride_bytes(0));
    std::memcpy(ids_row, tokens + start,
                static_cast<size_t>(seq_len) * sizeof(int32_t));
    std::memcpy(tgt_row, tokens + start + 1,
                static_cast<size_t>(seq_len) * sizeof(int32_t));
  }

  out.ids = ids_t;
  out.targets = tgt_t;
}

void DatasetGPU::fill_batch(const int32_t *tokens,
                            const std::vector<uint64_t> &window_starts,
                            uint64_t first_window_index, uint32_t batch_size,
                            uint32_t seq_len,
                            TensorStore &tensor_store,
                            DeviceBackend &device_backend, Device device,
                            TrainBatch &out) const {
  if (device != Device::GPU) {
    throw std::runtime_error(
        "DatasetGPU::fill_batch requires GPU device tensors");
  }

  TensorView ids_t = tensor_store.temp_ds_ids();
  TensorView tgt_t = tensor_store.temp_ds_targets();
  const uint64_t row_bytes = static_cast<uint64_t>(seq_len) * sizeof(int32_t);

  for (uint32_t b = 0; b < batch_size; ++b) {
    const uint64_t start = window_starts.at(first_window_index + b);
    auto *ids_row = reinterpret_cast<uint8_t *>(ids_t.data()) +
                    b * ids_t.stride_bytes(0);
    auto *tgt_row = reinterpret_cast<uint8_t *>(tgt_t.data()) +
                    b * tgt_t.stride_bytes(0);
    device_backend.copy_host2device(ids_row, tokens + start, row_bytes);
    device_backend.copy_host2device(tgt_row, tokens + start + 1, row_bytes);
  }

  out.ids = ids_t;
  out.targets = tgt_t;
}

bool TextDataset::ends_with(const std::string &s, const std::string &suffix) {
  if (suffix.size() > s.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

DatasetHeader
TextDataset::decode_header_or_throw_(const uint8_t *bytes, size_t len) {
  if (len < kDatasetHeaderBytes) {
    throw std::runtime_error(
        "TextDataset: dataset header is missing or truncated");
  }

  DatasetHeader header{};
  std::memcpy(&header.magic, bytes, sizeof(header.magic));
  std::memcpy(&header.version, bytes + sizeof(header.magic),
              sizeof(header.version));
  std::memcpy(&header.vocab_size,
              bytes + sizeof(header.magic) + sizeof(header.version),
              sizeof(header.vocab_size));

  if (header.magic != kDatasetMagic) {
    throw std::runtime_error(
        "TextDataset: invalid dataset header magic; expected 0x4C4E4750 (LNGP)");
  }
  if (header.version != kDatasetVersion) {
    throw std::runtime_error("TextDataset: unsupported dataset header version: " +
                             std::to_string(header.version));
  }
  if (header.vocab_size == 0) {
    throw std::runtime_error(
        "TextDataset: invalid dataset header vocab_size=0");
  }

  const size_t payload_bytes = len - kDatasetHeaderBytes;
  if ((payload_bytes % sizeof(uint32_t)) != 0) {
    throw std::runtime_error(
        "TextDataset: dataset payload is not aligned to uint32 tokens");
  }

  header.num_tokens = static_cast<uint64_t>(payload_bytes / sizeof(uint32_t));

  return header;
}
DatasetHeader TextDataset::read_header_or_throw(const std::string &path) {
  if (!ends_with(path, ".bin")) {
    throw std::runtime_error(
        "TextDataset: training input must be a .bin dataset file");
  }

  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("TextDataset: failed to open input file: " + path);
  }
  struct stat sb {};
  if (fstat(fd, &sb) != 0) {
    close(fd);
    throw std::runtime_error("TextDataset: fstat failed for input file: " + path);
  }
  if (sb.st_size <= 0) {
    close(fd);
    throw std::runtime_error("TextDataset: input file is empty: " + path);
  }

  void *mapped =
      mmap(nullptr, static_cast<size_t>(sb.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  const int close_rc = close(fd);
  (void)close_rc;
  if (mapped == MAP_FAILED) {
    throw std::runtime_error("TextDataset: mmap failed for input file: " + path +
                             " (" + std::strerror(errno) + ")");
  }

  DatasetHeader header{};
  try {
    header = decode_header_or_throw_(reinterpret_cast<const uint8_t *>(mapped),
                                     static_cast<size_t>(sb.st_size));
  } catch (...) {
    munmap(mapped, static_cast<size_t>(sb.st_size));
    throw;
  }

  munmap(mapped, static_cast<size_t>(sb.st_size));
  return header;
}

std::string TextDataset::early_evaluate_input(const Config &cfg, bool report) {
  const std::string selected_input_path = cfg.tokenization.output_binary;
  if (report) {
    std::cout << "[TextDataset] Dataset file to load: "
              << selected_input_path << "\n";
  }
  if (selected_input_path.empty()) {
    throw std::runtime_error(
        "TextDataset: tokenization.output_binary is required");
  }
  return selected_input_path;
}

TextDataset::TextDataset(TensorStore &tensor_store,
                         DeviceBackend &device_backend,
                         const Config &cfg,
                         bool shuffle_blocks, TrainingReportSink *report_sink,
                         ITrainingObserver *load_observer)
    : IDataLoader(load_observer),
      tensorStore_(tensor_store),
      device_backend_(&device_backend),
      device_(device_backend.device()),
      seq_len_(cfg.training.train_seq_len),
      window_stride_(cfg.training.window_stride),
      batch_size_(cfg.training.batch_size),
      shuffle_blocks_(shuffle_blocks),
      report_sink_(report_sink) {
  const std::string dataset_path = early_evaluate_input(cfg, false);

  if (device_ == Device::CPU) {
    backend_ = std::make_unique<DatasetCPU>();
  } else {
    backend_ = std::make_unique<DatasetGPU>();
  }

  REQUIRE_DEBUG(seq_len_ > 0, [&]() { return "TextDataset: seq_len must be > 0"; });
  REQUIRE_DEBUG(window_stride_ > 0,
                [&]() { return "TextDataset: window_stride must be > 0"; });
  REQUIRE_DEBUG(batch_size_ > 0, [&]() { return "TextDataset: batch_size must be > 0"; });

  if (!ends_with(dataset_path, ".bin")) {
    throw std::runtime_error(
        "TextDataset: training input must be a .bin dataset file");
  }

  const int fd = open(dataset_path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("TextDataset: failed to open input file: " + dataset_path);
  }

  struct stat sb {};
  if (fstat(fd, &sb) != 0) {
    close(fd);
    throw std::runtime_error("TextDataset: fstat failed for input file: " + dataset_path);
  }
  if (sb.st_size <= 0) {
    close(fd);
    throw std::runtime_error("TextDataset: input file is empty: " + dataset_path);
  }

  mmap_len_ = static_cast<size_t>(sb.st_size);
  mmap_base_ = mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd, 0);
  const int close_rc = close(fd);
  (void)close_rc;
  if (mmap_base_ == MAP_FAILED) {
    mmap_base_ = nullptr;
    mmap_len_ = 0;
    throw std::runtime_error("TextDataset: mmap failed for input file: " + dataset_path +
                             " (" + std::strerror(errno) + ")");
  }

  try {
    const auto *bytes = reinterpret_cast<const uint8_t *>(mmap_base_);
    header_ = decode_header_or_throw_(bytes, mmap_len_);
    tokens_ = reinterpret_cast<const int32_t *>(bytes + kDatasetHeaderBytes);
    num_tokens_ = header_.num_tokens;
  } catch (...) {
    munmap(mmap_base_, mmap_len_);
    mmap_base_ = nullptr;
    mmap_len_ = 0;
    tokens_ = nullptr;
    num_tokens_ = 0;
    throw;
  }
  source_format_ = SourceFormat::TOKEN_U32;

  if (num_tokens_ < static_cast<uint64_t>(seq_len_ + 1)) {
    throw std::runtime_error("TextDataset: dataset is too small for training");
  }

  build_blocks_();
  reset_epoch();
}

TextDataset::~TextDataset() {
  if (mmap_base_ != nullptr && mmap_base_ != MAP_FAILED && mmap_len_ > 0) {
    munmap(mmap_base_, mmap_len_);
  }
  mmap_base_ = nullptr;
  mmap_len_ = 0;
  tokens_ = nullptr;
  num_tokens_ = 0;
}

uint32_t TextDataset::max_token_id() const {
  uint32_t out = 0;
  for (uint64_t i = 0; i < num_tokens_; ++i) {
    const uint32_t u = static_cast<uint32_t>(tokens_[i]);
    if (u > out) {
      out = u;
    }
  }
  return out;
}

void TextDataset::build_blocks_() {
  block_starts_.clear();

  const uint64_t min_needed = static_cast<uint64_t>(seq_len_) + 1;
  const uint64_t N = num_tokens_;
  const uint64_t stride = static_cast<uint64_t>(window_stride_);
  for (uint64_t start = 0; start + min_needed <= N; start += stride) {
    block_starts_.push_back(start);
  }

  if (block_starts_.empty()) {
    throw std::runtime_error(
        "TextDataset: no full blocks fit: "
        "num_tokens=" + std::to_string(N) +
        ", batch_size=" + std::to_string(batch_size_) +
        ", seq_len=" + std::to_string(seq_len_) +
        ", min_needed=seq_len+1=" + std::to_string(min_needed) +
        ", window_stride=" + std::to_string(window_stride_) +
        ". Need at least num_tokens >= seq_len + 1.");
  }
  const uint64_t full_batch_windows =
      (block_starts_.size() / batch_size_) * batch_size_;
  block_starts_.resize(static_cast<size_t>(full_batch_windows));
  if (block_starts_.empty()) {
    throw std::runtime_error(
        "TextDataset: not enough windows for one full batch: windows=" +
        std::to_string(full_batch_windows) +
        ", batch_size=" + std::to_string(batch_size_));
  }
}

void TextDataset::maybe_shuffle_blocks_() {
  if (!shuffle_blocks_) {
    return;
  }

  std::mt19937 rng(12345u);
  std::shuffle(block_starts_.begin(), block_starts_.end(), rng);
}

void TextDataset::reset_epoch() {
  cursor_ = 0;
  block_index_ = 0;
  maybe_shuffle_blocks_();
}

bool TextDataset::next_impl(TrainBatch &out) {
  if (block_index_ + batch_size_ > block_starts_.size()) {
    return false;
  }

  const uint64_t block_span =
      static_cast<uint64_t>(batch_size_) * static_cast<uint64_t>(seq_len_);
  const uint64_t first_window_index = block_index_;
  const uint64_t first_start = block_starts_[static_cast<size_t>(first_window_index)];
  for (uint32_t b = 0; b < batch_size_; ++b) {
    const uint64_t start = block_starts_[static_cast<size_t>(first_window_index + b)];
    REQUIRE_DEBUG(start + static_cast<uint64_t>(seq_len_) < num_tokens_, [&]() {
      return "TextDataset::next: mapped token read would exceed dataset size";
    });
  }
  block_index_ += batch_size_;

  if (report_sink_ != nullptr && fetch_report_count_ < fetch_report_limit_) {
    TrainingFetchReportData data{};
    data.start_index = first_start;
    data.batch_size = batch_size_;
    data.seq_len = seq_len_;
    data.total_tokens = block_span;
    report_sink_->report_fetch(data);
    fetch_report_count_ += 1;
  }

  backend_->fill_batch(tokens_, block_starts_, first_window_index, batch_size_,
                       seq_len_, tensorStore_, *device_backend_, device_, out);
  return true;
}
