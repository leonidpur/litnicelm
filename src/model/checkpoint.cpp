#include "checkpoint.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr uint32_t kCheckpointMagic = 0x4C474354; // "LGCT" (4 bytes)
constexpr uint32_t kCurrentFormatVersion = 6;
constexpr uint32_t kCurrentAlgoVersion = 1;
constexpr size_t kConfVersionMax = 32;

struct CkptHeaderV1 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 1;
  uint32_t n_layers = 0;
  uint32_t n_heads = 0;
  uint32_t d_model = 0;
  uint32_t d_ff = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;
  uint64_t global_step = 0;
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
};

struct CkptHeaderV2 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 2;
  uint32_t n_layers = 0;
  uint32_t n_heads = 0;
  uint32_t d_model = 0;
  uint32_t d_ff = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;
  uint64_t alignment_bytes = 0;
  uint64_t global_step = 0;
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
};

struct CkptHeaderV3 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 3;
  uint32_t algo_version = kCurrentAlgoVersion;
  uint32_t conf_version = 1;
  uint32_t n_layers = 0;
  uint32_t n_heads = 0;
  uint32_t d_model = 0;
  uint32_t d_ff = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;
  uint64_t alignment_bytes = 0;
  uint64_t global_step = 0;
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
};

struct CkptHeaderV4 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 4;
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
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
};

struct CkptHeaderV5 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = 5;
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
};

struct CkptHeaderV6 {
  uint32_t magic = kCheckpointMagic;
  uint32_t version = kCurrentFormatVersion;
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
  std::cerr << "Checkpoint mismatch: " << name << " file=" << got
            << " config=" << expected << "\n";
  if (detail != nullptr) {
    if (!detail->empty()) {
      *detail += "; ";
    }
    *detail += std::string(name) + " file=" + std::to_string(got) +
               " config=" + std::to_string(expected);
  }
  return false;
}

template <typename T>
bool check_eq_u64(const char *name, T got, T expected, std::string *detail) {
  if (got == expected) {
    return true;
  }
  std::cerr << "Checkpoint mismatch: " << name << " file=" << got
            << " config=" << expected << "\n";
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
    std::cerr << "Checkpoint validation failed: " << label
              << " size is not aligned to float32\n";
    if (error_detail != nullptr) {
      *error_detail = std::string(label) + " size is not aligned to float32";
    }
    return false;
  }

  const float *values = reinterpret_cast<const float *>(data);
  const uint64_t count = bytes / sizeof(float);
  for (uint64_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      std::cerr << "Checkpoint validation failed: " << label
                << " contains non-finite value at float index " << i << "\n";
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
} // namespace

bool save_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t global_step,
                     uint32_t epoch,
                     const CheckpointConvergenceState *convergence_state) {
  if (!arena_memory_ok(data_arena.base, data_arena.bytes) ||
      !arena_memory_ok(adam_state.base, adam_state.bytes)) {
    return false;
  }

  CkptHeaderV6 h;
  h.algo_version = kCurrentAlgoVersion;
  if (conf_version.size() >= kConfVersionMax) {
    std::cerr << "Checkpoint warning: conf.version is too long, truncating to "
              << (kConfVersionMax - 1) << " chars\n";
  }
  std::snprintf(h.conf_version, sizeof(h.conf_version), "%s",
                conf_version.c_str());
  h.n_layers = model.n_layers;
  h.n_heads = model.n_heads;
  h.d_model = model.d_model;
  h.d_ff = model.d_ff;
  h.vocab_size = model.target_vocab_size;
  h.max_seq_len = model.max_seq_len;
  h.alignment_bytes = alignment_bytes;
  h.global_step = global_step;
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
  out.write(reinterpret_cast<const char *>(&h), sizeof(h));
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

bool load_checkpoint(const std::string &path, const ModelConfig &model,
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
    if (error_detail != nullptr) {
      *error_detail = "invalid checkpoint memory buffers";
    }
    return false;
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error_detail != nullptr) {
      *error_detail = "failed to open checkpoint file";
    }
    return false;
  }

  CkptPrefix pfx;
  in.read(reinterpret_cast<char *>(&pfx), sizeof(pfx));
  if (!in || pfx.magic != kCheckpointMagic) {
    std::cerr << "Checkpoint load failed: invalid magic in " << path << "\n";
    if (error_detail != nullptr) {
      *error_detail = "invalid magic number";
    }
    return false;
  }
  in.clear();
  in.seekg(0, std::ios::beg);
  StagingMemory staging_memory;

  if (pfx.version == 1) {
    CkptHeaderV1 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v1 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v1 header";
      }
      return false;
    }
    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }
    std::cerr << "Checkpoint note: v1 header has no memory.alignment_bytes; "
                 "alignment check skipped.\n";

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read parameter arena payload";
          }
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read optimizer arena payload";
          }
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = 0;
    return true;
  }

  if (pfx.version == 2) {
    CkptHeaderV2 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v2 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v2 header";
      }
      return false;
    }
    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes,
                      &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read parameter arena payload";
          }
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read optimizer arena payload";
          }
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = 0;
    return true;
  }

  if (pfx.version == 3) {
    CkptHeaderV3 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v3 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v3 header";
      }
      return false;
    }
    if (h.algo_version != kCurrentAlgoVersion) {
      std::cerr << "Checkpoint warning: algo_version mismatch file="
                << h.algo_version << " runtime=" << kCurrentAlgoVersion << "\n";
    }
    const std::string file_conf_version = std::to_string(h.conf_version);
    if (file_conf_version != conf_version) {
      std::cerr << "Checkpoint warning: conf_version mismatch file="
                << file_conf_version << " config=" << conf_version << "\n";
    }

    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes,
                      &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read parameter arena payload";
          }
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read optimizer arena payload";
          }
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = 0;
    return true;
  }

  if (pfx.version == 4) {
    CkptHeaderV4 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v4 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v4 header";
      }
      return false;
    }
    if (h.algo_version != kCurrentAlgoVersion) {
      std::cerr << "Checkpoint warning: algo_version mismatch file="
                << h.algo_version << " runtime=" << kCurrentAlgoVersion << "\n";
    }
    const std::string file_conf_version(h.conf_version);
    if (file_conf_version != conf_version) {
      std::cerr << "Checkpoint warning: conf_version mismatch file="
                << file_conf_version << " config=" << conf_version << "\n";
    }

    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes,
                      &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read parameter arena payload";
          }
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read optimizer arena payload";
          }
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = 0;
    return true;
  }

  if (pfx.version == 6) {
    CkptHeaderV6 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v6 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v6 header";
      }
      return false;
    }
    if (h.algo_version != kCurrentAlgoVersion) {
      std::cerr << "Checkpoint warning: algo_version mismatch file="
                << h.algo_version << " runtime=" << kCurrentAlgoVersion << "\n";
    }
    const std::string file_conf_version(h.conf_version);
    if (file_conf_version != conf_version) {
      std::cerr << "Checkpoint warning: conf_version mismatch file="
                << file_conf_version << " config=" << conf_version << "\n";
    }

    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes,
                      &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr && error_detail->empty()) {
          *error_detail = "could not read parameter arena payload";
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr && error_detail->empty()) {
          *error_detail = "could not read optimizer arena payload";
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = h.epoch;
    if (restored_convergence_state != nullptr) {
      *restored_convergence_state = h.convergence;
    }
    return true;
  }

  if (pfx.version == 5) {
    CkptHeaderV5 h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in) {
      std::cerr << "Checkpoint load failed: truncated v5 header in " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = "truncated v5 header";
      }
      return false;
    }
    if (h.algo_version != kCurrentAlgoVersion) {
      std::cerr << "Checkpoint warning: algo_version mismatch file="
                << h.algo_version << " runtime=" << kCurrentAlgoVersion << "\n";
    }
    const std::string file_conf_version(h.conf_version);
    if (file_conf_version != conf_version) {
      std::cerr << "Checkpoint warning: conf_version mismatch file="
                << file_conf_version << " config=" << conf_version << "\n";
    }

    std::string mismatch_detail;
    bool ok = true;
    ok = check_eq_u32("model.n_layers", h.n_layers, model.n_layers, &mismatch_detail) && ok;
    ok = check_eq_u32("model.n_heads", h.n_heads, model.n_heads, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_model", h.d_model, model.d_model, &mismatch_detail) && ok;
    ok = check_eq_u32("model.d_ff", h.d_ff, model.d_ff, &mismatch_detail) && ok;
    ok = check_eq_u32("model.target_vocab_size", h.vocab_size, model.target_vocab_size, &mismatch_detail) && ok;
    ok = check_eq_u32("model.max_seq_len", h.max_seq_len, model.max_seq_len, &mismatch_detail) && ok;
    ok = check_eq_u64("memory.alignment_bytes", h.alignment_bytes, alignment_bytes,
                      &mismatch_detail) && ok;
    ok = check_eq_u64("data_bytes", h.data_bytes, data_arena.bytes, &mismatch_detail) && ok;
    ok = check_eq_u64("adam_bytes", h.adam_bytes, adam_state.bytes, &mismatch_detail) && ok;
    if (!ok) {
      std::cerr << "Checkpoint load failed: incompatible checkpoint " << path
                << "\n";
      if (error_detail != nullptr) {
        *error_detail = mismatch_detail;
      }
      return false;
    }

    if (!read_arena_payload(in, backend, staging_memory, data_arena.base,
                            h.data_bytes, "parameter arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read parameter arena payload";
          }
        }
        return false;
    }
    if (!read_arena_payload(in, backend, staging_memory, adam_state.base,
                            h.adam_bytes, "optimizer arena", error_detail)) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          if (error_detail->empty()) {
            *error_detail = "could not read optimizer arena payload";
          }
        }
        return false;
    }
    restored_step = h.global_step;
    restored_epoch = h.epoch;
    return true;
  }

  std::cerr << "Checkpoint load failed: unsupported checkpoint version "
            << pfx.version << " in " << path << "\n";
  if (error_detail != nullptr) {
    *error_detail = "unsupported checkpoint format version " + std::to_string(pfx.version);
  }
  return false;
}
