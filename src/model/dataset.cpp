#include "dataset.hpp"
#include "training_report_sink.hpp"
#include <utils/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <random>
#include <stdexcept>
#include <string>

namespace {
constexpr uint32_t kDatasetMagic = 0x4C4E4750u; // "LNGP"
constexpr uint32_t kDatasetVersion = 1u;
constexpr size_t kDatasetHeaderBytes = sizeof(uint32_t) * 3;
} // namespace

void DatasetCPU::fill_batch(const int32_t *tokens, uint64_t ids_begin,
                            uint64_t tgt_begin, uint64_t block_span,
                            TensorFactory &tensor_factory, Device device,
                            TrainBatch &out) const {
  const int64_t rows = static_cast<int64_t>(block_span);
  const Shape2D shp{rows, 1};

  (void)device;
  TensorView ids_t = tensor_factory.temp_ds_ids(shp.r);
  TensorView tgt_t = tensor_factory.temp_ds_targets(shp.r);

  auto *ids_ptr = reinterpret_cast<int32_t *>(ids_t.data());
  auto *tgt_ptr = reinterpret_cast<int32_t *>(tgt_t.data());

  std::memcpy(ids_ptr, tokens + ids_begin,
              static_cast<size_t>(block_span) * sizeof(int32_t));
  std::memcpy(tgt_ptr, tokens + tgt_begin,
              static_cast<size_t>(block_span) * sizeof(int32_t));

  out.ids = ids_t;
  out.targets = tgt_t;
}

void DatasetGPU::fill_batch(const int32_t *tokens, uint64_t ids_begin,
                            uint64_t tgt_begin, uint64_t block_span,
                            TensorFactory &tensor_factory, Device device,
                            TrainBatch &out) const {
  (void)tokens;
  (void)ids_begin;
  (void)tgt_begin;
  (void)block_span;
  (void)tensor_factory;
  (void)device;
  (void)out;
  throw std::runtime_error(
      "TextDataset::next: GPU backend path not implemented");
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

TextDataset::TextDataset(TensorFactory &tensor_factory, const std::string &dataset_path,
                         Device device, uint32_t seq_len, uint32_t batch_size,
                         bool shuffle_blocks, TrainingReportSink *report_sink)
    : tensorFactory_(tensor_factory),
      device_(device),
      seq_len_(seq_len),
      batch_size_(batch_size),
      shuffle_blocks_(shuffle_blocks),
      report_sink_(report_sink) {
  if (device_ == Device::CPU) {
    backend_ = std::make_unique<DatasetCPU>();
  } else {
    backend_ = std::make_unique<DatasetGPU>();
  }

  REQUIRE_DEBUG(seq_len_ > 0, [&]() { return "TextDataset: seq_len must be > 0"; });
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

  const uint64_t block_span =
      static_cast<uint64_t>(batch_size_) * static_cast<uint64_t>(seq_len_);
  const uint64_t min_needed = block_span + 1;
  const uint64_t N = num_tokens_;
  for (uint64_t start = 0; start + min_needed <= N; start += block_span) {
    block_starts_.push_back(start);
  }

  if (block_starts_.empty()) {
    throw std::runtime_error(
        "TextDataset: no full blocks fit; reduce batch_size or seq_len");
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

bool TextDataset::next(TrainBatch &out) {
  if (block_index_ >= block_starts_.size()) {
    return false;
  }

  const uint64_t start = block_starts_[block_index_++];
  const uint64_t block_span =
      static_cast<uint64_t>(batch_size_) * static_cast<uint64_t>(seq_len_);
  const uint64_t ids_begin = start;
  const uint64_t tgt_begin = start + 1;
  REQUIRE_DEBUG(tgt_begin + block_span <= num_tokens_, [&]() {
    return "TextDataset::next: mapped token read would exceed dataset size";
  });

  if (report_sink_ != nullptr && fetch_report_count_ < fetch_report_limit_) {
    TrainingFetchReportData data{};
    data.start_index = start;
    data.batch_size = batch_size_;
    data.seq_len = seq_len_;
    data.total_tokens = block_span;
    report_sink_->report_fetch(data);
    fetch_report_count_ += 1;
  }

  backend_->fill_batch(tokens_, ids_begin, tgt_begin, block_span, tensorFactory_,
                       device_, out);
  return true;
}
