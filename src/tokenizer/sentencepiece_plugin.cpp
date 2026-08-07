#include "sentencepiece_plugin.hpp"

#include <report_interface.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if LITNICEGPT_HAVE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#endif

namespace fs = std::filesystem;

namespace SentencePiecePluginUtils {
#if LITNICEGPT_HAVE_SENTENCEPIECE
std::string model_path_for(const std::string &artifacts_dir) {
  return (fs::path(artifacts_dir) / "spm.model").string();
}

std::string vocab_path_for(const std::string &artifacts_dir) {
  return (fs::path(artifacts_dir) / "spm.vocab").string();
}

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TOKENIZER, event, step, value,
                          message);
}
#endif
} // namespace SentencePiecePluginUtils

struct SentencePiecePlugin::Impl {
#if LITNICEGPT_HAVE_SENTENCEPIECE
  sentencepiece::SentencePieceProcessor processor;
  bool loaded = false;
#endif
};

SentencePiecePlugin::SentencePiecePlugin() : impl_(std::make_unique<Impl>()) {}

SentencePiecePlugin::~SentencePiecePlugin() = default;

void SentencePiecePlugin::train(const std::string &corpus_path,
                                const std::string &artifacts_dir,
                                uint32_t target_vocab_size,
                                ReportSink *sink) {
  if (corpus_path.empty()) {
    throw std::runtime_error("SentencePiecePlugin: corpus_path is required");
  }
  if (artifacts_dir.empty()) {
    throw std::runtime_error("SentencePiecePlugin: artifacts_dir is required");
  }
  if (target_vocab_size == 0) {
    throw std::runtime_error(
        "SentencePiecePlugin: target_vocab_size must be > 0");
  }

#if LITNICEGPT_HAVE_SENTENCEPIECE
  fs::create_directories(artifacts_dir);
  const std::string model_prefix = (fs::path(artifacts_dir) / "spm").string();
  SentencePiecePluginUtils::report_if(
      sink, ReportEvent::START, 0, 0.0f,
      "SentencePiece training started: corpus=" + corpus_path +
          ", artifacts=" + artifacts_dir + ", vocab_size=" +
          std::to_string(target_vocab_size));

  const std::string args = "--input=" + corpus_path +
                           " --model_prefix=" + model_prefix +
                           " --vocab_size=" +
                           std::to_string(target_vocab_size) +
                           " --model_type=bpe --character_coverage=1.0";
  const auto status = sentencepiece::SentencePieceTrainer::Train(args);
  if (!status.ok()) {
    const std::string message =
        "SentencePiece training failed: " + status.ToString();
    SentencePiecePluginUtils::report_if(sink, ReportEvent::ERROR, 0, 0.0f,
                                        message);
    throw std::runtime_error(message);
  }

  SentencePiecePluginUtils::report_if(
      sink, ReportEvent::END, 1, 100.0f,
      "SentencePiece artifacts created: model=" +
          SentencePiecePluginUtils::model_path_for(artifacts_dir) + ", vocab=" +
          SentencePiecePluginUtils::vocab_path_for(artifacts_dir));
#else
  (void)sink;
  throw std::runtime_error(
      "SentencePiecePlugin: SentencePiece support is not available in this build");
#endif
}

bool SentencePiecePlugin::load(const std::string &artifacts_dir) {
#if LITNICEGPT_HAVE_SENTENCEPIECE
  if (artifacts_dir.empty()) {
    return false;
  }

  const auto status =
      impl_->processor.Load(SentencePiecePluginUtils::model_path_for(artifacts_dir));
  impl_->loaded = status.ok();
  return impl_->loaded;
#else
  (void)artifacts_dir;
  return false;
#endif
}

std::vector<int32_t> SentencePiecePlugin::encode(const std::string &text) const {
#if LITNICEGPT_HAVE_SENTENCEPIECE
  if (!impl_->loaded) {
    throw std::runtime_error("SentencePiecePlugin::encode: plugin not loaded");
  }

  std::vector<int> raw_ids;
  const auto status = impl_->processor.Encode(text, &raw_ids);
  if (!status.ok()) {
    throw std::runtime_error("SentencePiecePlugin::encode failed: " +
                             status.ToString());
  }

  return std::vector<int32_t>(raw_ids.begin(), raw_ids.end());
#else
  (void)text;
  throw std::runtime_error(
      "SentencePiecePlugin::encode: SentencePiece support is not available in this build");
#endif
}

std::string SentencePiecePlugin::decode(const std::vector<int32_t> &ids) const {
#if LITNICEGPT_HAVE_SENTENCEPIECE
  if (!impl_->loaded) {
    throw std::runtime_error("SentencePiecePlugin::decode: plugin not loaded");
  }

  std::vector<int> raw_ids(ids.begin(), ids.end());
  std::string text;
  const auto status = impl_->processor.Decode(raw_ids, &text);
  if (!status.ok()) {
    throw std::runtime_error("SentencePiecePlugin::decode failed: " +
                             status.ToString());
  }
  return text;
#else
  (void)ids;
  throw std::runtime_error(
      "SentencePiecePlugin::decode: SentencePiece support is not available in this build");
#endif
}

int32_t SentencePiecePlugin::vocab_size() const {
#if LITNICEGPT_HAVE_SENTENCEPIECE
  return static_cast<int32_t>(impl_->processor.GetPieceSize());
#else
  return 0;
#endif
}

const char *SentencePiecePlugin::name() const {
  return "SentencePiece(BPE)";
}
