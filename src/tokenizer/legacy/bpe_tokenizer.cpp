#include "bpe_tokenizer.hpp"
#include "backend/device_backend.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
thread_local std::unique_ptr<Arena> tls_arena = nullptr;
const std::thread::id g_main_thread_id = std::this_thread::get_id();
constexpr uint64_t k_tls_arena_size_bytes = 32ull * 1024ull * 1024ull;

Arena &get_tls_arena() {
  static thread_local CpuBackend cpu_backend;
  if (!tls_arena) {
    tls_arena = std::make_unique<Arena>(cpu_backend, Device::CPU,
                                        k_tls_arena_size_bytes, 64);
  }
  return *tls_arena;
}

std::string trim(const std::string &s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  return s.substr(a, b - a);
}

} // namespace

bool BPETokenizer::load_from_files(const std::string &vocab_path,
                                   const std::string &merges_path,
                                   uint32_t expected_target_vocab_size) {
  id_to_token_.clear();
  token_to_id_.clear();
  merge_pairs_storage_.clear();
  merge_rank_.clear();
  merge_rank_view_.clear();

  {
    std::ifstream in(vocab_path);
    if (!in) {
      return false;
    }

    std::string line;
    while (std::getline(in, line)) {
      line = trim(line);
      if (line.empty()) {
        continue;
      }

      const size_t tab = line.find('\t');
      const std::string tok =
          (tab == std::string::npos) ? unescape_token(line)
                                     : unescape_token(trim(line.substr(0, tab)));
      int32_t id = (tab == std::string::npos)
                       ? static_cast<int32_t>(id_to_token_.size())
                       : std::stoi(trim(line.substr(tab + 1)));

      if (id < 0) {
        throw std::runtime_error("BPETokenizer: vocab id < 0");
      }
      if (token_to_id_.find(tok) != token_to_id_.end()) {
        throw std::runtime_error("BPETokenizer: duplicate token in vocab");
      }

      if (id >= static_cast<int32_t>(id_to_token_.size())) {
        id_to_token_.resize(static_cast<size_t>(id) + 1);
      }
      id_to_token_[static_cast<size_t>(id)] = tok;
      token_to_id_[tok] = id;
    }

    for (size_t i = 0; i < id_to_token_.size(); ++i) {
      if (id_to_token_[i].empty()) {
        throw std::runtime_error("BPETokenizer: vocab has empty slot at id " +
                                 std::to_string(i));
      }
    }
    if (id_to_token_.empty()) {
      throw std::runtime_error("BPETokenizer: loaded vocab is empty");
    }
    if (expected_target_vocab_size != 0 &&
        id_to_token_.size() !=
            static_cast<size_t>(expected_target_vocab_size)) {
      throw std::runtime_error(
          "BPETokenizer: vocab size mismatch. expected " +
          std::to_string(expected_target_vocab_size) + ", got " +
          std::to_string(id_to_token_.size()));
    }
  }

  {
    std::ifstream in(merges_path);
    if (!in) {
      return false;
    }

    std::string line;
    int32_t rank = 0;
    while (std::getline(in, line)) {
      line = trim(line);
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream iss(line);
      std::string a;
      std::string b;
      if (iss >> a >> b) {
        merge_pairs_storage_.push_back({unescape_token(a), unescape_token(b)});
        const auto &stored = merge_pairs_storage_.back();
        merge_rank_[stored] = rank;
        merge_rank_view_[{std::string_view(stored.first),
                          std::string_view(stored.second)}] = rank;
        ++rank;
      }
    }
  }

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::STEP_COMPLETE, 0, 0.0f,
            "BPETokenizer loaded. vocab=" + vocab_path + " (" +
                std::to_string(id_to_token_.size()) + " entries), merges=" +
                merges_path + " (" + std::to_string(merge_rank_view_.size()) + " ranks)");

  return true;
}

std::vector<int32_t> BPETokenizer::encode(const std::string &text) const {
  if (id_to_token_.empty()) {
    throw std::runtime_error("BPETokenizer::encode: tokenizer not loaded");
  }
  if (text.empty()) {
    return {};
  }

  Arena &arena = get_tls_arena();
  const uint64_t bytes_needed =
      static_cast<uint64_t>(text.size()) * sizeof(TokenNode);
  if (bytes_needed > arena.size_bytes()) {
    CpuBackend cpu_backend;
    Arena temp(cpu_backend, Device::CPU, bytes_needed, 64);
    try {
      return bpe_encode_optimized(text, temp);
    } catch (const std::out_of_range &) {
      return bpe_encode_fallback(text);
    }
  }

  try {
    return bpe_encode_optimized(text, arena);
  } catch (const std::out_of_range &) {
    return bpe_encode_fallback(text);
  }
}

std::vector<int32_t>
BPETokenizer::bpe_encode_optimized(const std::string &text, Arena &arena) const {
  const size_t n = text.size();
  auto *nodes = reinterpret_cast<TokenNode *>(arena.ptr());
  std::priority_queue<MergeCandidate, std::vector<MergeCandidate>,
                      std::greater<MergeCandidate>>
      pq;

  auto push_pair = [&](TokenNode *left, TokenNode *right) {
    if (!left || !right || left->deleted || right->deleted) {
      return;
    }
    const auto it = merge_rank_view_.find({left->view, right->view});
    if (it != merge_rank_view_.end()) {
      pq.push({it->second, left, right});
    }
  };

  for (size_t i = 0; i < n; ++i) {
    nodes[i].view = std::string_view(text.data() + i, 1);
    nodes[i].prev = (i > 0) ? &nodes[i - 1] : nullptr;
    nodes[i].next = (i + 1 < n) ? &nodes[i + 1] : nullptr;
    nodes[i].deleted = false;
  }

  for (size_t i = 0; i + 1 < n; ++i) {
    push_pair(&nodes[i], &nodes[i + 1]);
  }

  const bool print_progress =
      std::this_thread::get_id() == g_main_thread_id && n > 0;
  size_t current_token_count = n;
  uint64_t total_pops = 0;

  while (!pq.empty()) {
    ++total_pops;
    if (total_pops % 10000 == 0 && print_progress) {
      report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::PROGRESS, static_cast<uint32_t>(total_pops),
                static_cast<float>(current_token_count),
                "Queue pops=" + std::to_string(total_pops) +
                    " tokens=" + std::to_string(current_token_count));
    }

    const MergeCandidate top = pq.top();
    pq.pop();

    if (top.left->deleted || top.right->deleted || top.left->next != top.right) {
      continue;
    }

    top.left->view = std::string_view(top.left->view.data(),
                                      top.left->view.size() +
                                          top.right->view.size());

    TokenNode *to_delete = top.right;
    top.left->next = to_delete->next;
    if (to_delete->next) {
      to_delete->next->prev = top.left;
    }
    to_delete->deleted = true;
    --current_token_count;

    push_pair(top.left->prev, top.left);
    push_pair(top.left, top.left->next);
  }

  if (print_progress && total_pops >= 10000) {
    report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::STEP_COMPLETE, static_cast<uint32_t>(total_pops),
              static_cast<float>(current_token_count), "Queue merge finished");
  }

  std::vector<int32_t> ids;
  ids.reserve(n);
  TokenNode *curr = &nodes[0];
  while (curr) {
    if (!curr->deleted) {
      ids.push_back(token_to_id_.at(std::string(curr->view)));
    }
    curr = curr->next;
  }
  return ids;
}

std::vector<std::string>
BPETokenizer::bytes_to_base_symbols(const std::string &text) {
  std::vector<std::string> out;
  out.reserve(text.size());
  for (unsigned char b : text) {
    out.emplace_back(1, static_cast<char>(b));
  }
  return out;
}

std::vector<std::string>
BPETokenizer::bpe_merge(std::vector<std::string> symbols) const {
  if (symbols.size() < 2) {
    return symbols;
  }

  while (true) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_i = static_cast<size_t>(-1);

    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_rank_.find({symbols[i], symbols[i + 1]});
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }

    if (best_i == static_cast<size_t>(-1)) {
      break;
    }

    symbols[best_i] = symbols[best_i] + symbols[best_i + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
  }

  return symbols;
}

std::vector<int32_t>
BPETokenizer::bpe_encode_fallback(const std::string &text) const {
  std::vector<std::string> symbols = bytes_to_base_symbols(text);
  symbols = bpe_merge(std::move(symbols));

  std::vector<int32_t> ids;
  ids.reserve(symbols.size());
  for (const auto &symbol : symbols) {
    ids.push_back(token_to_id_.at(symbol));
  }
  return ids;
}

std::string BPETokenizer::decode(const std::vector<int32_t> &ids) const {
  if (id_to_token_.empty()) {
    throw std::runtime_error("BPETokenizer::decode: tokenizer not loaded");
  }

  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
      throw std::runtime_error("BPETokenizer::decode: id out of range");
    }
    out += id_to_token_[static_cast<size_t>(id)];
  }
  return out;
}

std::string BPETokenizer::join_tokens(const std::vector<std::string> &toks) {
  std::string out;
  size_t total = 0;
  for (const auto &t : toks) {
    total += t.size();
  }
  out.reserve(total);
  for (const auto &t : toks) {
    out += t;
  }
  return out;
}

std::string BPETokenizer::escape_token(const std::string &s) {
  std::ostringstream oss;
  oss << std::hex << std::uppercase << std::setfill('0');
  for (unsigned char ch : s) {
    if (std::isprint(ch) && !std::isspace(ch) && ch != '\\') {
      oss << static_cast<char>(ch);
    } else {
      oss << "\\x" << std::setw(2) << static_cast<int>(ch);
    }
  }
  return oss.str();
}

std::string BPETokenizer::unescape_token(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 3 < s.size() && s[i + 1] == 'x' &&
        std::isxdigit(static_cast<unsigned char>(s[i + 2])) &&
        std::isxdigit(static_cast<unsigned char>(s[i + 3]))) {
      const std::string hex = s.substr(i + 2, 2);
      const auto value =
          static_cast<unsigned char>(std::stoul(hex, nullptr, 16));
      out.push_back(static_cast<char>(value));
      i += 3;
      continue;
    }
    out.push_back(s[i]);
  }
  return out;
}
