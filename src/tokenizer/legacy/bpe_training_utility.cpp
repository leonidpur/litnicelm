#include "bpe_training_utility.hpp"

#include "bpe_tokenizer.hpp"
#include <config.hpp>
#include "operation_journal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct PairHash {
  size_t operator()(const std::pair<int32_t, int32_t> &p) const noexcept {
    const uint64_t a = static_cast<uint32_t>(p.first);
    const uint64_t b = static_cast<uint32_t>(p.second);
    return static_cast<size_t>((a << 32) ^ b);
  }
};

uint32_t fnv1a32_append(uint32_t hash, const void *data, size_t size) {
  const auto *ptr = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint32_t>(ptr[i]);
    hash *= 16777619u;
  }
  return hash;
}

uint32_t fnv1a32(const std::string &text) {
  uint32_t hash = 2166136261u;
  return fnv1a32_append(hash, text.data(), text.size());
}

std::string read_file_or_throw(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("BPE_TrainingUtility: failed to open corpus: " +
                             path.string());
  }
  std::string data((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  if (data.empty()) {
    throw std::runtime_error("BPE_TrainingUtility: corpus is empty: " +
                             path.string());
  }
  return data;
}

std::pair<int32_t, int32_t> select_best_pair(
    const std::unordered_map<std::pair<int32_t, int32_t>, size_t, PairHash> &pair_counts,
    size_t *best_count_out) {
  size_t best_count = 0;
  std::pair<int32_t, int32_t> best{-1, -1};
  for (const auto &kv : pair_counts) {
    if (kv.second > best_count ||
        (kv.second == best_count && best_count != 0 && kv.first < best)) {
      best = kv.first;
      best_count = kv.second;
    }
  }
  if (best_count_out != nullptr) {
    *best_count_out = best_count;
  }
  if (best_count < 2) {
    return {-1, -1};
  }
  return best;
}

std::string build_vocab_text(const std::vector<std::string> &tokens) {
  std::string out;
  out.reserve(tokens.size() * 8);
  for (size_t i = 0; i < tokens.size(); ++i) {
    out += BPETokenizer::escape_token(tokens[i]);
    out.push_back('\t');
    out += std::to_string(i);
    out.push_back('\n');
  }
  return out;
}

std::string build_merges_text(
    const std::vector<std::pair<int32_t, int32_t>> &merges,
    const std::vector<std::string> &tokens) {
  std::string out;
  out.reserve(merges.size() * 16);
  for (const auto &merge : merges) {
    out += BPETokenizer::escape_token(tokens[static_cast<size_t>(merge.first)]);
    out.push_back(' ');
    out += BPETokenizer::escape_token(tokens[static_cast<size_t>(merge.second)]);
    out.push_back('\n');
  }
  return out;
}

void write_text_file(const fs::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("BPE_TrainingUtility: failed to write file: " +
                             path.string());
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) {
    throw std::runtime_error("BPE_TrainingUtility: write failed for file: " +
                             path.string());
  }
}

void validate_roundtrip_or_throw(const fs::path &vocab_path,
                                 const fs::path &merges_path,
                                 const std::string &corpus,
                                 int32_t bpe_validation_num_threads,
                                 uint32_t bpe_validation_sample_rate,
                                 ReportSink *sink) {
  constexpr size_t kChunkBytes = 8 * 1024;
  const size_t total_size = corpus.size();
  const size_t total_chunks =
      std::max<size_t>(1, (total_size + kChunkBytes - 1) / kChunkBytes);
  const uint32_t sample_rate = std::min<uint32_t>(100u, bpe_validation_sample_rate);
  const size_t sampled_chunks =
      std::max<size_t>(1, (total_chunks * static_cast<size_t>(sample_rate) + 99u) / 100u);
  std::vector<size_t> chunk_indices;
  chunk_indices.reserve(sampled_chunks);
  if (sampled_chunks >= total_chunks) {
    for (size_t i = 0; i < total_chunks; ++i) {
      chunk_indices.push_back(i);
    }
  } else {
    for (size_t i = 0; i < sampled_chunks; ++i) {
      const size_t idx = std::min(
          total_chunks - 1,
          (i * total_chunks) / sampled_chunks);
      if (chunk_indices.empty() || chunk_indices.back() != idx) {
        chunk_indices.push_back(idx);
      }
    }
  }

  const size_t configured_threads =
      (bpe_validation_num_threads == 0 || bpe_validation_num_threads == -1)
          ? std::max<size_t>(1, std::thread::hardware_concurrency())
          : static_cast<size_t>(bpe_validation_num_threads);
  const size_t bpe_validation_num_threads_effective =
      std::min(configured_threads, chunk_indices.size());

  std::atomic<size_t> next_chunk{0};
  std::atomic<size_t> completed_chunks{0};
  std::atomic<bool> stop_reporter{false};
  std::mutex print_mutex;

  report_utils::report_if(sink, ReportPhase::VALIDATION, ReportEvent::START, 0, 0.0f,
            "Validating tokenizer round-trip: chunks=" +
                std::to_string(chunk_indices.size()) +
                ", bpe_validation_num_threads=" +
                std::to_string(bpe_validation_num_threads_effective));

  std::thread reporter([&]() {
    while (!stop_reporter.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      if (stop_reporter.load()) {
        break;
      }

      const size_t current = completed_chunks.load();
      const size_t dispatched = std::min(next_chunk.load(), chunk_indices.size());
      const size_t pct = (current * 100) / chunk_indices.size();
      std::lock_guard<std::mutex> lock(print_mutex);
      report_utils::report_if(sink, ReportPhase::VALIDATION, ReportEvent::PROGRESS,
                static_cast<uint32_t>(current), static_cast<float>(pct),
                "validation dispatched=" + std::to_string(dispatched) + "/" +
                    std::to_string(chunk_indices.size()));
    }
  });

  std::vector<std::future<void>> futures;
  try {
    for (size_t i = 0; i < bpe_validation_num_threads_effective; ++i) {
      futures.push_back(std::async(std::launch::async, [&, i]() {
        BPETokenizer tokenizer;
        if (!tokenizer.load_from_files(vocab_path.string(), merges_path.string())) {
          throw std::runtime_error("Validation: Failed to load artifacts");
        }

        while (true) {
          const size_t sample_index = next_chunk.fetch_add(1);
          if (sample_index >= chunk_indices.size()) {
            break;
          }

          const size_t chunk = chunk_indices[sample_index];
          const size_t start = chunk * kChunkBytes;
          const size_t end = std::min(start + kChunkBytes, total_size);
          const std::string piece = corpus.substr(start, end - start);
          if (tokenizer.decode(tokenizer.encode(piece)) != piece) {
            throw std::runtime_error("BPE Validation failed at chunk " +
                                     std::to_string(chunk));
          }

          const size_t current = ++completed_chunks;
          const size_t pct = (current * 100) / chunk_indices.size();
          std::lock_guard<std::mutex> lock(print_mutex);
          report_utils::report_if(sink, ReportPhase::VALIDATION, ReportEvent::PROGRESS,
                    static_cast<uint32_t>(current), static_cast<float>(pct),
                    "validation progress");
        }
      }));
    }
    for (auto &f : futures) {
      f.get();
    }
  } catch (...) {
    stop_reporter = true;
    if (reporter.joinable()) {
      reporter.join();
    }
    throw;
  }

  stop_reporter = true;
  if (reporter.joinable()) {
    reporter.join();
  }

  report_utils::report_if(sink, ReportPhase::VALIDATION, ReportEvent::END,
            static_cast<uint32_t>(completed_chunks.load()), 100.0f,
            "Validation complete");
}
} // namespace

BPE_TrainingUtility::BPE_TrainingUtility(std::string corpus_path,
                                         std::string artifacts_dir,
                                         uint32_t target_vocab_size,
                                         bool run_validation,
                                         int32_t bpe_validation_num_threads,
                                         uint32_t bpe_validation_sample_rate,
                                         ReportSink *sink)
    : corpus_path_(std::move(corpus_path)),
      artifacts_dir_(std::move(artifacts_dir)),
      target_vocab_size_(target_vocab_size),
      run_validation_(run_validation),
      bpe_validation_num_threads_(bpe_validation_num_threads),
      bpe_validation_sample_rate_(bpe_validation_sample_rate),
      sink_(sink) {}

void BPE_TrainingUtility::run() const {
  if (corpus_path_.empty()) {
    throw std::runtime_error(
        "BPE_TrainingUtility: tokenizer.bpe_corpus_file is required");
  }
  if (artifacts_dir_.empty()) {
    throw std::runtime_error(
        "BPE_TrainingUtility: tokenizer.bpe_artifacts_dir is required");
  }
  if (target_vocab_size_ < 256) {
    throw std::runtime_error(
        "BPE_TrainingUtility: target vocab size must be >= 256 for byte-level BPE");
  }

  const std::string corpus = read_file_or_throw(corpus_path_);

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::START, 0, 0.0f,
            "BPE create params: data source=" + corpus_path_ +
                ", vocab size=" + std::to_string(target_vocab_size_) +
                ", text size=" + std::to_string(corpus.size()) + " bytes");
  struct Node {
    int32_t id = 0;
    Node *prev = nullptr;
    Node *next = nullptr;
    bool deleted = false;
  };

  std::vector<Node> list_storage(corpus.size());
  for (size_t i = 0; i < corpus.size(); ++i) {
    list_storage[i].id = static_cast<int32_t>(static_cast<unsigned char>(corpus[i]));
    list_storage[i].prev = (i > 0) ? &list_storage[i - 1] : nullptr;
    list_storage[i].next =
        (i + 1 < corpus.size()) ? &list_storage[i + 1] : nullptr;
  }

  std::vector<std::string> tokens;
  tokens.reserve(target_vocab_size_);
  std::unordered_map<std::string, int32_t> token_ids;
  token_ids.reserve(target_vocab_size_);
  for (int i = 0; i < 256; ++i) {
    tokens.emplace_back(1, static_cast<char>(i));
    token_ids.emplace(tokens.back(), i);
  }

  std::vector<std::pair<int32_t, int32_t>> merges;
  merges.reserve(target_vocab_size_ - 256);
  using Pair = std::pair<int32_t, int32_t>;
  std::unordered_map<Pair, size_t, PairHash> pair_counts;
  std::unordered_map<Pair, std::unordered_set<Node *>, PairHash> pair_locations;
  pair_counts.reserve(corpus.size());
  pair_locations.reserve(corpus.size());

  auto add_pair = [&](Node *a) {
    if (a == nullptr || a->deleted || a->next == nullptr || a->next->deleted) {
      return;
    }
    const Pair p{a->id, a->next->id};
    pair_counts[p] += 1;
    pair_locations[p].insert(a);
  };

  auto remove_pair = [&](Node *a) {
    if (a == nullptr || a->deleted || a->next == nullptr || a->next->deleted) {
      return;
    }
    const Pair p{a->id, a->next->id};
    auto count_it = pair_counts.find(p);
    if (count_it == pair_counts.end()) {
      return;
    }
    auto loc_it = pair_locations.find(p);
    if (loc_it != pair_locations.end()) {
      loc_it->second.erase(a);
    }
    if (count_it->second > 1) {
      count_it->second -= 1;
      return;
    }
    pair_counts.erase(count_it);
    if (loc_it != pair_locations.end() && loc_it->second.empty()) {
      pair_locations.erase(loc_it);
    }
  };

  for (size_t i = 0; i + 1 < list_storage.size(); ++i) {
    add_pair(&list_storage[i]);
  }
  const uint32_t total_merges =
      (target_vocab_size_ > 256) ? (target_vocab_size_ - 256) : 0;
  uint64_t merge_iterations = 0;
  std::string completion_reason = "target vocab size reached";

  if (total_merges > 0) {
    report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::PROGRESS, 0, 0.0f,
              "BPE merge progress initialized: total_merges=" +
                  std::to_string(total_merges));
  }

  while (tokens.size() < static_cast<size_t>(target_vocab_size_)) {
    const Pair best = select_best_pair(pair_counts, nullptr);
    if (best.first < 0 || best.second < 0) {
      completion_reason = "no more merge pairs with frequency >= 2";
      break;
    }

    auto loc_it = pair_locations.find(best);
    if (loc_it == pair_locations.end() || loc_it->second.empty()) {
      pair_counts.erase(best);
      completion_reason = "pair index exhausted before reaching target vocab size";
      continue;
    }

    std::vector<Node *> current_locs(loc_it->second.begin(), loc_it->second.end());
    pair_locations.erase(loc_it);
    pair_counts.erase(best);
    ++merge_iterations;

    merges.push_back(best);
    const std::string merged_token = tokens[static_cast<size_t>(best.first)] +
                                     tokens[static_cast<size_t>(best.second)];
    int32_t merged_id = 0;
    auto token_it = token_ids.find(merged_token);
    if (token_it == token_ids.end()) {
      merged_id = static_cast<int32_t>(tokens.size());
      tokens.push_back(merged_token);
      token_ids.emplace(tokens.back(), merged_id);
    } else {
      merged_id = token_it->second;
    }

    for (Node *node_a : current_locs) {
      if (node_a == nullptr || node_a->deleted || node_a->next == nullptr ||
          node_a->next->deleted) {
        continue;
      }
      if (node_a->id != best.first || node_a->next->id != best.second) {
        continue;
      }

      Node *node_b = node_a->next;
      Node *node_prev = node_a->prev;
      Node *node_next_next = node_b->next;

      remove_pair(node_prev);
      remove_pair(node_a);
      remove_pair(node_b);

      node_a->id = merged_id;
      node_a->next = node_next_next;
      if (node_next_next != nullptr) {
        node_next_next->prev = node_a;
      }
      node_b->deleted = true;
      node_b->prev = nullptr;
      node_b->next = nullptr;

      add_pair(node_prev);
      add_pair(node_a);
    }

    if (total_merges > 0) {
      const uint32_t completed =
          static_cast<uint32_t>(tokens.size() > 256 ? tokens.size() - 256 : 0);
      const uint32_t pct = (completed * 100u) / total_merges;
      report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::PROGRESS,
                static_cast<uint32_t>(merge_iterations), static_cast<float>(pct),
                "BPE merge progress: " + std::to_string(completed) + "/" +
                    std::to_string(total_merges));
    }
  }

  if (total_merges > 0) {
    report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::STEP_COMPLETE,
              static_cast<uint32_t>(merge_iterations), 100.0f,
              "BPE merge loop complete");
  }

  if (tokens.size() >= static_cast<size_t>(target_vocab_size_)) {
    completion_reason = "target vocab size reached";
  }
  if (tokens.size() != static_cast<size_t>(target_vocab_size_)) {
    throw std::runtime_error(
        "BPE_TrainingUtility: corpus was insufficient to reach target vocab "
        "size. target=" +
        std::to_string(target_vocab_size_) +
        ", actual=" + std::to_string(tokens.size()));
  }

  const fs::path output_dir(artifacts_dir_);
  fs::create_directories(output_dir);

  const fs::path vocab_path = output_dir / "vocab.txt";
  const fs::path merges_path = output_dir / "merges.txt";
  const fs::path checksum_path = output_dir / "bpe_checksum.chs";

  const std::string vocab_text = build_vocab_text(tokens);
  const std::string merges_text = build_merges_text(merges, tokens);
  const uint32_t vocab_hash = fnv1a32(vocab_text);
  const uint32_t merges_hash = fnv1a32(merges_text);
  uint32_t combined_hash = 2166136261u;
  combined_hash = fnv1a32_append(combined_hash, vocab_text.data(), vocab_text.size());
  combined_hash = fnv1a32_append(combined_hash, "\n", 1);
  combined_hash = fnv1a32_append(combined_hash, merges_text.data(), merges_text.size());

  write_text_file(vocab_path, vocab_text);
  write_text_file(merges_path, merges_text);

  const std::string checksum_text =
      "vocab.txt " + std::to_string(vocab_hash) + "\n" +
      "merges.txt " + std::to_string(merges_hash) + "\n" +
      "combined " + std::to_string(combined_hash) + "\n";
  write_text_file(checksum_path, checksum_text);
  if (run_validation_) {
    validate_roundtrip_or_throw(vocab_path, merges_path, corpus,
                                bpe_validation_num_threads_,
                                bpe_validation_sample_rate_, sink_);
  }

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::END, static_cast<uint32_t>(merges.size()),
            static_cast<float>(tokens.size()),
            "Created BPE artifacts in " + output_dir.string() +
                ", corpus=" + corpus_path_ +
                ", vocab=" + vocab_path.string() +
                ", merges=" + merges_path.string() +
                ", checksum=" + checksum_path.string() +
                ", target vocab size=" + std::to_string(target_vocab_size_) +
                ", actual vocab size=" + std::to_string(tokens.size()) +
                ", completion reason=" + completion_reason);
}

int run_create_bpe_mode(const std::string &config_path, ReportSink *sink) {
  const Config cfg = Config::load_from_file(config_path);
  BPE_TrainingUtility utility(cfg.tokenizer.bpe_corpus_file,
                              cfg.tokenizer.bpe_artifacts_dir,
                              cfg.tokenizer.target_vocab_size,
                              cfg.tokenizer.run_validation,
                              cfg.tokenizer.bpe_validation_num_threads,
                              cfg.tokenizer.bpe_validation_sample_rate,
                              sink);
  utility.run();
  const fs::path artifacts_dir(cfg.tokenizer.bpe_artifacts_dir);
  const std::string details =
      "Status: SUCCESS\n\n"
      "Input Corpus: " + cfg.tokenizer.bpe_corpus_file + "\n\n"
      "Artifacts: " + (artifacts_dir / "vocab.txt").string() + ", " +
      (artifacts_dir / "merges.txt").string() + "\n\n"
      "Vocab Size: " + std::to_string(cfg.tokenizer.target_vocab_size);
  log_operation(cfg.paths.journal_file, "BPE_TOKENIZER_GEN", details);
  return 0;
}
