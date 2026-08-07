#pragma once

#include <tokenizer.hpp>
#include <arena.hpp>
#include <report_interface.hpp>

#include <cstdint>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

class BPETokenizer final : public Tokenizer {
public:
  BPETokenizer() = default;

  bool load_from_files(const std::string &vocab_path,
                       const std::string &merges_path,
                       uint32_t expected_target_vocab_size = 0);
  void set_report_sink(ReportSink *sink) { sink_ = sink; }

  std::vector<int32_t> encode(const std::string &text) const override;
  std::string decode(const std::vector<int32_t> &ids) const override;

  int32_t vocab_size() const override {
    return static_cast<int32_t>(id_to_token_.size());
  }
  const char *name() const override { return "BPETokenizer(ranked-merges)"; }

  static std::string escape_token(const std::string &s);
  static std::string unescape_token(const std::string &s);

private:
  struct TokenNode {
    std::string_view view;
    TokenNode *prev = nullptr;
    TokenNode *next = nullptr;
    bool deleted = false;
  };

  struct MergeCandidate {
    int32_t rank = 0;
    TokenNode *left = nullptr;
    TokenNode *right = nullptr;

    bool operator>(const MergeCandidate &other) const {
      return rank > other.rank;
    }
  };

  struct StringViewPairHash {
    size_t operator()(
        const std::pair<std::string_view, std::string_view> &p) const noexcept {
      std::hash<std::string_view> hs;
      const size_t h1 = hs(p.first);
      const size_t h2 = hs(p.second);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::pair<std::string, std::string>> merge_pairs_storage_;
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
      std::hash<std::string> hs;
      const size_t h1 = hs(p.first);
      const size_t h2 = hs(p.second);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };
  std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash>
      merge_rank_;
  std::unordered_map<std::pair<std::string_view, std::string_view>, int32_t,
                     StringViewPairHash>
      merge_rank_view_;
  ReportSink *sink_ = nullptr;

  std::vector<int32_t> bpe_encode_optimized(const std::string &text,
                                            Arena &arena) const;
  std::vector<int32_t> bpe_encode_fallback(const std::string &text) const;
  static std::vector<std::string> bytes_to_base_symbols(const std::string &text);
  std::vector<std::string> bpe_merge(std::vector<std::string> symbols) const;
  static std::string join_tokens(const std::vector<std::string> &toks);
};
