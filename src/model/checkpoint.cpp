#include "checkpoint.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
constexpr uint32_t kCheckpointMagic = 0x4C474354; // "LGCT" (4 bytes)
constexpr uint32_t kCurrentFormatVersion = 4;
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
  uint64_t data_bytes = 0;
  uint64_t adam_bytes = 0;
};

struct CkptPrefix {
  uint32_t magic = 0;
  uint32_t version = 0;
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

bool cpu_memory_ok(const void *base, uint64_t bytes, Device device) {
  return device == Device::CPU && (bytes == 0 || base != nullptr);
}
} // namespace

bool save_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t global_step) {
  if (!cpu_memory_ok(data_arena.base, data_arena.bytes, data_arena.device) ||
      !cpu_memory_ok(adam_state.base, adam_state.bytes, adam_state.device)) {
    return false;
  }

  CkptHeaderV4 h;
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
  h.max_seq_len = model.window_capacity;
  h.alignment_bytes = alignment_bytes;
  h.global_step = global_step;
  h.data_bytes = data_arena.bytes;
  h.adam_bytes = adam_state.bytes;

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char *>(&h), sizeof(h));
  if (!out) {
    return false;
  }

  if (h.data_bytes > 0) {
    out.write(reinterpret_cast<const char *>(data_arena.base),
              static_cast<std::streamsize>(h.data_bytes));
    if (!out) {
      return false;
    }
  }
  if (h.adam_bytes > 0) {
    out.write(reinterpret_cast<const char *>(adam_state.base),
              static_cast<std::streamsize>(h.adam_bytes));
    if (!out) {
      return false;
    }
  }
  return true;
}

bool load_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state,
                     uint64_t &restored_step,
                     std::string *error_detail) {
  if (!cpu_memory_ok(data_arena.base, data_arena.bytes, data_arena.device) ||
      !cpu_memory_ok(adam_state.base, adam_state.bytes, adam_state.device)) {
    if (error_detail != nullptr) {
      *error_detail = "non-CPU or invalid checkpoint memory buffers";
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
    ok = check_eq_u32("model.window_capacity", h.max_seq_len, model.window_capacity, &mismatch_detail) && ok;
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

    if (h.data_bytes > 0) {
      in.read(reinterpret_cast<char *>(data_arena.base),
              static_cast<std::streamsize>(h.data_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read parameter arena payload";
        }
        return false;
      }
    }
    if (h.adam_bytes > 0) {
      in.read(reinterpret_cast<char *>(adam_state.base),
              static_cast<std::streamsize>(h.adam_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read optimizer arena payload";
        }
        return false;
      }
    }
    restored_step = h.global_step;
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
    ok = check_eq_u32("model.window_capacity", h.max_seq_len, model.window_capacity, &mismatch_detail) && ok;
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

    if (h.data_bytes > 0) {
      in.read(reinterpret_cast<char *>(data_arena.base),
              static_cast<std::streamsize>(h.data_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read parameter arena payload";
        }
        return false;
      }
    }
    if (h.adam_bytes > 0) {
      in.read(reinterpret_cast<char *>(adam_state.base),
              static_cast<std::streamsize>(h.adam_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read optimizer arena payload";
        }
        return false;
      }
    }
    restored_step = h.global_step;
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
    ok = check_eq_u32("model.window_capacity", h.max_seq_len, model.window_capacity, &mismatch_detail) && ok;
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

    if (h.data_bytes > 0) {
      in.read(reinterpret_cast<char *>(data_arena.base),
              static_cast<std::streamsize>(h.data_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read parameter arena payload";
        }
        return false;
      }
    }
    if (h.adam_bytes > 0) {
      in.read(reinterpret_cast<char *>(adam_state.base),
              static_cast<std::streamsize>(h.adam_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read optimizer arena payload";
        }
        return false;
      }
    }
    restored_step = h.global_step;
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
    ok = check_eq_u32("model.window_capacity", h.max_seq_len, model.window_capacity, &mismatch_detail) && ok;
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

    if (h.data_bytes > 0) {
      in.read(reinterpret_cast<char *>(data_arena.base),
              static_cast<std::streamsize>(h.data_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read parameter arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read parameter arena payload";
        }
        return false;
      }
    }
    if (h.adam_bytes > 0) {
      in.read(reinterpret_cast<char *>(adam_state.base),
              static_cast<std::streamsize>(h.adam_bytes));
      if (!in) {
        std::cerr << "Checkpoint load failed: could not read optimizer arena\n";
        if (error_detail != nullptr) {
          *error_detail = "could not read optimizer arena payload";
        }
        return false;
      }
    }
    restored_step = h.global_step;
    return true;
  }

  std::cerr << "Checkpoint load failed: unsupported checkpoint version "
            << pfx.version << " in " << path << "\n";
  if (error_detail != nullptr) {
    *error_detail = "unsupported checkpoint format version " + std::to_string(pfx.version);
  }
  return false;
}
