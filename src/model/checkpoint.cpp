#include "checkpoint.hpp"
#include "i_position_encoding.hpp"
#include "model_algo_factory.hpp"
#include <litnicelm_utils/hash_utils.hpp>
#include <litnicelm_utils/log.hpp>
#include "model_events.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
constexpr uint32_t kCheckpointMagic = 0x4C474354; // "LGCT" (4 bytes)
constexpr uint32_t kCurrentFormatVersion = 7;
constexpr uint32_t kCurrentAlgoVersion = 1;
constexpr size_t kConfVersionMax = 32;
constexpr size_t kPositionEncodingMax = 16;

// The first part of the v7 header (v6 was this header alone; v6 files are
// no longer loaded).
struct CkptHeaderV6 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 6;
  uint32_t algo_version = kCurrentAlgoVersion;
  char conf_version[kConfVersionMax] = {};
  uint32_t n_layers = 0;
  uint32_t n_heads = 0;
  uint32_t d_model = 0;
  uint32_t d_ff = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;
  uint64_t alignment_bytes = 0;
  uint64_t global_step = 0;
  uint32_t epoch = 0;
  uint32_t reserved = 0;
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
  CheckpointConvergenceState convergence{};
};

// The v7 header: the v6 header plus the position encoding the parameters
// were trained with.
struct CkptHeaderV7 {
  CkptHeaderV6 base{};
  char position_encoding[kPositionEncodingMax] = {};
};

struct CkptPrefix {
  uint32_t magic = 0;
  uint32_t version = 0;
};

class StagingMemory {
public:
  void ensure_bytes(uint64_t bytes) {
    if (bytes > buffer_.size()) {
      buffer_.resize(static_cast<size_t>(bytes));
    }
  }

  void *data() { return buffer_.empty() ? nullptr : buffer_.data(); }
  const void *data() const { return buffer_.empty() ? nullptr : buffer_.data(); }

private:
  std::vector<uint8_t> buffer_;
};

template <typename T>
bool check_eq_u32(const char *name, T got, T expected, std::string *detail) {
  if (got == expected) {
    return true;
  }
  logger().log(model_events::kCheckpointMismatch,
               std::string(name) + " file=" + std::to_string(got) +
                   " config=" + std::to_string(expected));
  if (detail != nullptr) {
    if (!detail->empty()) {
      *detail += "; ";
    }
    *detail += std::string(name) + " file=" + std::to_string(got) +
               " config=" + std::to_string(expected);
  }
  return false;
}

bool check_eq_str(const char *name, const std::string &got,
                  const std::string &expected, std::string *detail) {
  if (got == expected) {
    return true;
  }
  logger().log(model_events::kCheckpointMismatch,
               std::string(name) + " file=" + got + " config=" + expected);
  if (detail != nullptr) {
    if (!detail->empty()) {
      *detail += "; ";
    }
    *detail += std::string(name) + " file=" + got + " config=" + expected;
  }
  return false;
}

template <typename T>
bool check_eq_u64(const char *name, T got, T expected, std::string *detail) {
  if (got == expected) {
    return true;
  }
  logger().log(model_events::kCheckpointMismatch,
               std::string(name) + " file=" + std::to_string(got) +
                   " config=" + std::to_string(expected));
  if (detail != nullptr) {
    if (!detail->empty()) {
      *detail += "; ";
    }
    *detail += std::string(name) + " file=" + std::to_string(got) +
               " config=" + std::to_string(expected);
  }
  return false;
}

bool arena_memory_ok(const void *base, uint64_t bytes) {
  return bytes == 0 || base != nullptr;
}

bool staging_f32_all_finite(const void *data, uint64_t bytes, const char *label,
                            std::string *error_detail) {
  if (bytes == 0) {
    return true;
  }
  if ((bytes % sizeof(float)) != 0) {
    logger().log(model_events::kCheckpointValidationFailed,
                 std::string(label) + " size is not aligned to float32");
    if (error_detail != nullptr) {
      *error_detail = std::string(label) + " size is not aligned to float32";
    }
    return false;
  }

  const float *values = reinterpret_cast<const float *>(data);
  const uint64_t count = bytes / sizeof(float);
  for (uint64_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      logger().log(model_events::kCheckpointValidationFailed,
                   std::string(label) + " contains non-finite value at float index " +
                       std::to_string(i));
      if (error_detail != nullptr) {
        *error_detail = std::string(label) +
                        " contains non-finite value at float index " +
                        std::to_string(i);
      }
      return false;
    }
  }
  return true;
}

bool write_arena_payload(std::ofstream &out, DeviceBackend &backend,
                         StagingMemory &staging_memory, const void *src,
                         uint64_t bytes, const char *label) {
  if (bytes == 0) {
    return true;
  }
  staging_memory.ensure_bytes(bytes);
  backend.copy_device2host(staging_memory.data(), src, bytes);
  if (!staging_f32_all_finite(staging_memory.data(), bytes, label, nullptr)) {
    return false;
  }
  out.write(reinterpret_cast<const char *>(staging_memory.data()),
            static_cast<std::streamsize>(bytes));
  return static_cast<bool>(out);
}

bool read_arena_payload(std::ifstream &in, DeviceBackend &backend,
                        StagingMemory &staging_memory, void *dst,
                        uint64_t bytes, const char *label,
                        std::string *error_detail) {
  if (bytes == 0) {
    return true;
  }
  staging_memory.ensure_bytes(bytes);
  in.read(reinterpret_cast<char *>(staging_memory.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) {
    return false;
  }
  if (!staging_f32_all_finite(staging_memory.data(), bytes, label, error_detail)) {
    return false;
  }
  backend.copy_host2device(dst, staging_memory.data(), bytes);
  return true;
}

// The fingerprint's starting value. Not the standard FNV-1a 64 offset
// basis (14695981039346656037, one digit longer), but every .tokfp sidecar
// was written with it, so it stays.
constexpr uint64_t kFingerprintSeed = 1469598103934665603ull;

void hash_string(uint64_t &hash, const std::string &value) {
  hash = hash_utils::fnv1a64(hash, value.data(), value.size());
  const char sep = '\0';
  hash = hash_utils::fnv1a64(hash, &sep, 1);
}

void hash_file(uint64_t &hash, const std::filesystem::path &path) {
  hash_string(hash, path.filename().string());
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open tokenizer artifact: " +
                             path.string());
  }
  std::vector<char> buffer(64u * 1024u);
  while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
         in.gcount() > 0) {
    hash = hash_utils::fnv1a64(hash, buffer.data(), static_cast<size_t>(in.gcount()));
  }
}

std::string fingerprint_sidecar_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".tokfp";
}
} // namespace

namespace {

std::string checkpoint_tokenizer_fingerprint(const Config &cfg) {
  namespace fs = std::filesystem;
  uint64_t hash = kFingerprintSeed;
  hash_string(hash, "litnicelm-tokenizer-fingerprint-v1");
  hash_string(hash, cfg.tokenizer.type);
  hash_string(hash, std::to_string(cfg.tokenizer.target_vocab_size));
  hash_string(hash, cfg.tokenizer.inter_file_boundary);

  std::vector<fs::path> files;
  if (!cfg.tokenizer.bpe_artifacts_dir.empty() &&
      fs::is_directory(cfg.tokenizer.bpe_artifacts_dir)) {
    for (const auto &entry : fs::directory_iterator(cfg.tokenizer.bpe_artifacts_dir)) {
      if (entry.is_regular_file()) {
        files.push_back(entry.path());
      }
    }
  } else {
    if (!cfg.tokenizer.bpe_vocab_file.empty()) {
      files.push_back(cfg.tokenizer.bpe_vocab_file);
    }
    if (!cfg.tokenizer.bpe_merges_file.empty()) {
      files.push_back(cfg.tokenizer.bpe_merges_file);
    }
  }
  // Tokenizers without artifacts (e.g. "character") are identified by the
  // config fields above alone.
  std::sort(files.begin(), files.end());
  for (const auto &file : files) {
    hash_file(hash, file);
  }

  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

bool write_checkpoint_tokenizer_fingerprint(const std::string &checkpoint_path,
                                            const std::string &fingerprint,
                                            std::string *error_detail) {
  try {
    const std::string sidecar = fingerprint_sidecar_path(checkpoint_path);
    std::filesystem::path path(sidecar);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(sidecar, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("failed to open tokenizer fingerprint sidecar");
    }
    out << fingerprint << "\n";
    return static_cast<bool>(out);
  } catch (const std::exception &e) {
    if (error_detail != nullptr) {
      *error_detail = e.what();
    }
    return false;
  }
}

bool verify_checkpoint_tokenizer_fingerprint(const std::string &checkpoint_path,
                                             const std::string &fingerprint,
                                             std::string *error_detail) {
  try {
    const std::string sidecar = fingerprint_sidecar_path(checkpoint_path);
    std::ifstream in(sidecar, std::ios::binary);
    if (!in) {
      logger().log(model_events::kCheckpointSidecarMissing, sidecar);
      return true;
    }
    std::string stored;
    std::getline(in, stored);
    if (stored == fingerprint) {
      return true;
    }
    if (error_detail != nullptr) {
      *error_detail = "tokenizer fingerprint mismatch file=" + stored +
                      " runtime=" + fingerprint;
    }
    return false;
  } catch (const std::exception &e) {
    if (error_detail != nullptr) {
      *error_detail = e.what();
    }
    return false;
  }
}

bool save_checkpoint(const std::string &path, const ModelConfig &model,
                     const IPositionEncoding &position_encoding,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t optimizer_steps,
                     uint32_t epoch,
                     const CheckpointConvergenceState *convergence_state) {
  if (!arena_memory_ok(data_arena.base, data_arena.bytes) ||
      !arena_memory_ok(adam_state.base, adam_state.bytes)) {
    return false;
  }

  CkptHeaderV7 h7;
  h7.base.version = kCurrentFormatVersion;
  std::snprintf(h7.position_encoding, sizeof(h7.position_encoding), "%s",
                position_encoding.name());
  CkptHeaderV6 &h = h7.base;
  h.algo_version = kCurrentAlgoVersion;
  if (conf_version.size() >= kConfVersionMax) {
    logger().log(model_events::kCheckpointConfVersionTruncated,
                 std::to_string(kConfVersionMax - 1) + " chars");
  }
  std::snprintf(h.conf_version, sizeof(h.conf_version), "%s",
                conf_version.c_str());
  h.n_layers = model.n_layers;
  h.n_heads = model.n_heads;
  h.d_model = model.d_model;
  h.d_ff = model.d_ff;
  h.vocab_size = model.vocab_size;
  h.max_seq_len = model.max_seq_len;
  h.alignment_bytes = alignment_bytes;
  h.global_step = optimizer_steps;
  h.epoch = epoch;
  h.data_bytes = data_arena.bytes;
  h.adam_bytes = adam_state.bytes;
  if (convergence_state != nullptr) {
    h.convergence = *convergence_state;
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  StagingMemory staging_memory;
  out.write(reinterpret_cast<const char *>(&h7), sizeof(h7));
  if (!out) {
    return false;
  }

  if (!write_arena_payload(out, backend, staging_memory, data_arena.base,
                           h.data_bytes, "parameter arena")) {
    return false;
  }
  if (!write_arena_payload(out, backend, staging_memory, adam_state.base,
                           h.adam_bytes, "optimizer arena")) {
    return false;
  }
  return true;
}

void set_error(std::string *error_detail, const std::string &text) {
  if (error_detail != nullptr) {
    *error_detail = text;
  }
}

// Reads the magic, the format version (only kCurrentFormatVersion loads)
// and the v7 header.
bool read_header(std::ifstream &in, const std::string &path, CkptHeaderV7 &h7,
                 std::string *error_detail) {
  CkptPrefix pfx;
  in.read(reinterpret_cast<char *>(&pfx), sizeof(pfx));
  if (!in || pfx.magic != kCheckpointMagic) {
    logger().log(model_events::kCheckpointLoadFailed, "invalid magic in " + path);
    set_error(error_detail, "invalid magic number");
    return false;
  }
  if (pfx.version != kCurrentFormatVersion) {
    const std::string unsupported = "unsupported checkpoint format version " +
                                    std::to_string(pfx.version) + " (only v" +
                                    std::to_string(kCurrentFormatVersion) + " is supported)";
    logger().log(model_events::kCheckpointLoadFailed, unsupported + " in " + path);
    set_error(error_detail, unsupported);
    return false;
  }
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(&h7), sizeof(h7));
  if (!in) {
    logger().log(model_events::kCheckpointLoadFailed, "truncated v7 header in " + path);
    set_error(error_detail, "truncated v7 header");
    return false;
  }
  h7.position_encoding[kPositionEncodingMax - 1] = '\0';
  return true;
}

// The header describes this model and these arenas. Differing algo or conf
// versions only warn.
bool header_matches(const CkptHeaderV7 &h7, const std::string &path, const ModelConfig &model,
                    const IPositionEncoding &position_encoding,
                    const std::string &conf_version, uint64_t alignment_bytes,
                    const ArenaView &data_arena, const AdamStateView &adam_state,
                    std::string *error_detail) {
  const CkptHeaderV6 &h = h7.base;
  if (h.algo_version != kCurrentAlgoVersion) {
    logger().log(model_events::kCheckpointVersionMismatch,
                 "algo_version mismatch file=" + std::to_string(h.algo_version) +
                     " runtime=" + std::to_string(kCurrentAlgoVersion));
  }
  if (std::string(h.conf_version) != conf_version) {
    logger().log(model_events::kCheckpointVersionMismatch,
                 "conf_version mismatch file=" + std::string(h.conf_version) +
                     " config=" + conf_version);
  }
  std::string mismatch;
  bool ok = true;
  ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch) && ok;
  ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch) && ok;
  ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch) && ok;
  ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch) && ok;
  ok = check_eq_u32("model.vocab_size", h.vocab_size, model.vocab_size, &mismatch) && ok;
  ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch) && ok;
  ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes, &mismatch) && ok;
  ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch) && ok;
  ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch) && ok;
  ok = check_eq_str("model_algo.position_encoding", h7.position_encoding,
                    position_encoding.name(), &mismatch) && ok;
  if (!ok) {
    logger().log(model_events::kCheckpointLoadFailed, "incompatible checkpoint " + path);
    set_error(error_detail, mismatch);
  }
  return ok;
}

// The parameter arena, then the optimizer arena, as they follow the header.
bool read_payloads(std::ifstream &in, DeviceBackend &backend, const CkptHeaderV6 &h,
                   const ArenaView &data_arena, const AdamStateView &adam_state,
                   std::string *error_detail) {
  StagingMemory staging_memory;
  const struct {
    void *dst;
    uint64_t bytes;
    const char *label;
  } arenas[] = {{data_arena.base, h.data_bytes, "parameter arena"},
                {adam_state.base, h.adam_bytes, "optimizer arena"}};
  for (const auto &arena : arenas) {
    if (!read_arena_payload(in, backend, staging_memory, arena.dst, arena.bytes,
                            arena.label, error_detail)) {
      logger().log(model_events::kCheckpointLoadFailed,
                   std::string("could not read ") + arena.label);
      if (error_detail != nullptr && error_detail->empty()) {
        *error_detail = std::string("could not read ") + arena.label + " payload";
      }
      return false;
    }
  }
  return true;
}

bool load_checkpoint(const std::string &path, const ModelConfig &model,
                     const IPositionEncoding &position_encoding,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state,
                     uint64_t &restored_step, uint32_t &restored_epoch,
                     CheckpointConvergenceState *restored_convergence_state,
                     std::string *error_detail) {
  if (restored_convergence_state != nullptr) {
    *restored_convergence_state = CheckpointConvergenceState{};
  }
  if (!arena_memory_ok(data_arena.base, data_arena.bytes) ||
      !arena_memory_ok(adam_state.base, adam_state.bytes)) {
    set_error(error_detail, "invalid checkpoint memory buffers");
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    set_error(error_detail, "failed to open checkpoint file");
    return false;
  }
  CkptHeaderV7 h7{};
  if (!read_header(in, path, h7, error_detail) ||
      !header_matches(h7, path, model, position_encoding, conf_version, alignment_bytes,
                      data_arena, adam_state, error_detail) ||
      !read_payloads(in, backend, h7.base, data_arena, adam_state, error_detail)) {
    return false;
  }
  restored_step = h7.base.global_step;
  restored_epoch = h7.base.epoch;
  if (restored_convergence_state != nullptr) {
    *restored_convergence_state = h7.base.convergence;
  }
  return true;
}

} // namespace

CheckpointIO::CheckpointIO(const Config &cfg, const ModelAlgoFactory &algo,
                           DeviceBackend &backend, const ArenaView &params,
                           const AdamStateView &adam)
    : cfg_(cfg), position_encoding_(algo.create_position_encoding(cfg)), backend_(backend),
      params_(params), adam_(adam) {}

CheckpointIO::~CheckpointIO() = default;

void CheckpointIO::save(const std::string &path, uint64_t step, uint32_t epoch,
                        const CheckpointConvergenceState &convergence) const {
  const std::filesystem::path file(path);
  if (file.has_parent_path()) {
    std::filesystem::create_directories(file.parent_path());
  }
  if (!save_checkpoint(path, cfg_.model, *position_encoding_, cfg_.conf_version,
                       cfg_.memory.alignment_bytes, backend_, params_, adam_, step, epoch,
                       &convergence)) {
    throw CheckpointError("cannot save checkpoint " + path + ": write failed");
  }
  std::string error;
  if (!write_checkpoint_tokenizer_fingerprint(path, checkpoint_tokenizer_fingerprint(cfg_),
                                              &error)) {
    throw CheckpointError("cannot save checkpoint " + path +
                          ": tokenizer fingerprint sidecar: " + error);
  }
}

LoadedCheckpointState CheckpointIO::load(const std::string &path) const {
  LoadedCheckpointState state;
  std::string error;
  if (!load_checkpoint(path, cfg_.model, *position_encoding_, cfg_.conf_version,
                       cfg_.memory.alignment_bytes, backend_, params_, adam_, state.step,
                       state.epoch, &state.convergence, &error) ||
      !verify_checkpoint_tokenizer_fingerprint(path, checkpoint_tokenizer_fingerprint(cfg_),
                                               &error)) {
    throw CheckpointError("cannot load checkpoint " + path + ": " + error);
  }
  return state;
}
