// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "model_algo_factory.hpp"

#include "composed_self_attention.hpp"
#include "backend/backend_plugin_api.hpp"
#include "cudnn_flash_attention_core.hpp"
#include "ffn.hpp"
#include "fused_inplace_attention_core.hpp"
#include "fp32_logits_output_head.hpp"
#include "fused_bias_relu_ffn.hpp"
#include "fused_linear_ffn.hpp"
#include "fused_linear_output_projection.hpp"
#include "fused_linear_qkv_projection.hpp"
#include "gemm_output_projection.hpp"
#include "gemm_qkv_projection.hpp"
#include "guarded_self_attention.hpp"
#include "inplace_fused_bias_relu_ffn.hpp"
#include "kv_cache_attention.hpp"
#include "kv_cached_attention_core.hpp"
#include "linear_params.hpp"
#include "position_encoding.hpp"
#include "unfused_attention_core.hpp"
#include "unfused_output_head.hpp"
#include "self_attention_fused_inplace_multistream.hpp"

#include <stdexcept>
#include <utility>

namespace {

// cudnn_flash runs only on BF16 activations and a backend with the fused op.
void check_fused_attention_supported(const Ops &ops, const NodeSlices &slices) {
  if ((ops.fused_ops() & kFusedOpCausalAttentionBf16Act) == 0) {
    throw std::runtime_error(
        "ModelAlgoFactory: model_algo.attention cudnn_flash needs a backend "
        "with fused causal attention (the CUDA plugin built with cuDNN)");
  }
  if (slices.at("attn.qkv").dtype() != DType::BF16) {
    throw std::runtime_error(
        "ModelAlgoFactory: model_algo.attention cudnn_flash needs precision "
        "bf16_mixed");
  }
}

} // namespace

ModelAlgoFactory::ModelAlgoFactory(ModelAlgoConfig cfg, ModelRole role)
    : cfg_(cfg), role_(role) {}

std::unique_ptr<IAttentionCore>
ModelAlgoFactory::make_core(int layer_index, const Config &cfg, const Ops &ops,
                            const NodeSlices &slices) const {
  switch (cfg_.attention_impl) {
  case AttentionImplKind::Unfused:
    return std::make_unique<UnfusedAttentionCore>(layer_index, cfg, ops, slices);
  case AttentionImplKind::FusedInplace:
    return std::make_unique<FusedInplaceAttentionCore>(layer_index, cfg, ops,
                                                       slices);
  case AttentionImplKind::CudnnFlash:
    check_fused_attention_supported(ops, slices);
    return std::make_unique<CudnnFlashAttentionCore>(layer_index, cfg, ops,
                                                     slices);
  default:
    throw std::runtime_error("ModelAlgoFactory: no attention core for this "
                             "attention implementation");
  }
}

std::unique_ptr<IAttentionCore>
ModelAlgoFactory::for_role(std::unique_ptr<IAttentionCore> core, int layer_index,
                           const Config &cfg, const Ops &ops,
                           const NodeSlices &slices) const {
  if (role_ == ModelRole::Training || !cfg.inference.kv_cache) {
    return core;
  }
  return std::make_unique<KvCachedAttentionCore>(
      std::move(core),
      std::make_unique<KvCacheAttention>(layer_index, cfg, ops, slices));
}

std::unique_ptr<ISelfAttention> ModelAlgoFactory::create_attention(
    int layer_index, const Config &cfg, TensorStore &tensor_store,
    GradientStore *gradient_store, const Ops &ops,
    IPositionEncoding &position_encoding) const {
  const NodeSlices slices = tensor_store.attention_slices(layer_index);
  const AttentionParams params =
      attention_params(layer_index, tensor_store, gradient_store);
  std::unique_ptr<ISelfAttention> attention;
  if (cfg_.attention_impl == AttentionImplKind::FusedInplaceMultistream) {
    attention = std::make_unique<SelfAttentionFusedInplaceMultistream>(
        layer_index, cfg, params, ops, position_encoding, slices);
  } else {
    attention = std::make_unique<ComposedSelfAttention>(
        layer_index, cfg, params, position_encoding, slices,
        make_qkv_projection(layer_index, params.qkv, ops),
        for_role(make_core(layer_index, cfg, ops, slices), layer_index, cfg,
                 ops, slices),
        make_output_projection(layer_index, params.out, ops));
  }
  if (!cfg_.debug_guards) {
    return attention;
  }
  return std::make_unique<GuardedSelfAttention>(
      layer_index, std::move(attention),
      unguarded_attention_slice_requests(role_ == ModelRole::Training
                                             ? training_attention_dims(cfg)
                                             : inference_attention_dims(cfg)),
      slices, ops.backend());
}

std::unique_ptr<IQkvProjection>
ModelAlgoFactory::make_qkv_projection(int layer_index, const LinearParams &params,
                                      const Ops &ops) const {
  if (cfg_.projections == ProjectionKind::FusedLinear) {
    return std::make_unique<FusedLinearQkvProjection>(layer_index, params, ops);
  }
  return std::make_unique<GemmQkvProjection>(layer_index, params, ops);
}

std::unique_ptr<IOutputProjection>
ModelAlgoFactory::make_output_projection(int layer_index,
                                         const LinearParams &params,
                                         const Ops &ops) const {
  if (cfg_.projections == ProjectionKind::FusedLinear) {
    return std::make_unique<FusedLinearOutputProjection>(layer_index, params, ops);
  }
  return std::make_unique<GemmOutputProjection>(layer_index, params, ops);
}

std::unique_ptr<IFFN> ModelAlgoFactory::create_ffn(
    int layer_index, const Config &cfg, TensorStore &tensor_store,
    GradientStore *gradient_store, const Ops &ops) const {
  switch (cfg_.ffn_impl) {
  case FFNImplKind::Reference:
    return std::make_unique<FFN>(layer_index, cfg, tensor_store, gradient_store,
                                 ops);
  case FFNImplKind::FusedBiasRelu:
    return std::make_unique<FusedBiasReluFFN>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  case FFNImplKind::InplaceFusedBiasRelu:
    return std::make_unique<InplaceFusedBiasReluFFN>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  case FFNImplKind::FusedLinear:
    return std::make_unique<FusedLinearFFN>(
        layer_index, cfg, tensor_store,
        ffn_params(layer_index, tensor_store, gradient_store), ops);
  }
  throw std::runtime_error("ModelAlgoFactory: unknown FFN implementation");
}

std::unique_ptr<IOutputHead> ModelAlgoFactory::create_output_head(
    const Config &cfg, TensorStore &tensor_store, GradientStore *gradient_store,
    const Ops &ops) const {
  // Inference decodes from FP32 logits: under BF16 activations the head
  // casts the hidden states; under FP32 the configured head's logits are FP32.
  if (role_ == ModelRole::Inference &&
      tensor_store.compute_lm_head_w().dtype() == DType::BF16) {
    return std::make_unique<Fp32LogitsOutputHead>(cfg, tensor_store, ops);
  }
  switch (cfg_.output_head) {
  case OutputHeadKind::Unfused:
    return std::make_unique<UnfusedOutputHead>(cfg, tensor_store,
                                               gradient_store, ops);
  }
  throw std::runtime_error("ModelAlgoFactory: unknown output head");
}

std::unique_ptr<IPositionEncoding>
ModelAlgoFactory::create_position_encoding(const Config &cfg) const {
  switch (cfg_.position_encoding) {
  case PositionEncodingKind::Learned:
    return std::make_unique<LearnedPositionEncoding>(cfg);
  case PositionEncodingKind::None:
    return std::make_unique<NoPositionEncoding>();
  case PositionEncodingKind::Rope:
    return std::make_unique<RopePositionEncoding>(cfg);
  }
  throw std::runtime_error(
      "ModelAlgoFactory: unknown position encoding implementation");
}

std::vector<SliceRequest>
ModelAlgoFactory::attention_slice_requests(const AttentionDims &dims) const {
  std::vector<SliceRequest> requests = unguarded_attention_slice_requests(dims);
  return cfg_.debug_guards ? GuardedSelfAttention::with_guards(requests) : requests;
}

std::vector<SliceRequest> ModelAlgoFactory::unguarded_attention_slice_requests(
    const AttentionDims &dims) const {
  std::vector<SliceRequest> requests;
  switch (cfg_.attention_impl) {
  case AttentionImplKind::Unfused:
    requests = ComposedSelfAttention::slice_requests(dims);
    for (SliceRequest &request : UnfusedAttentionCore::slice_requests(dims)) {
      requests.push_back(std::move(request));
    }
    break;
  case AttentionImplKind::FusedInplace:
    requests = ComposedSelfAttention::slice_requests(dims);
    for (SliceRequest &request : FusedInplaceAttentionCore::slice_requests(dims)) {
      requests.push_back(std::move(request));
    }
    break;
  case AttentionImplKind::CudnnFlash:
    requests = ComposedSelfAttention::slice_requests(dims);
    for (SliceRequest &request : CudnnFlashAttentionCore::slice_requests(dims)) {
      requests.push_back(std::move(request));
    }
    break;
  case AttentionImplKind::FusedInplaceMultistream:
    if (dims.with_kv_cache) {
      throw std::runtime_error(
          "ModelAlgoFactory: inference.kv_cache needs a composed attention "
          "(not fused_inplace_multistream); set inference.kv_cache: false");
    }
    return SelfAttentionFusedInplaceMultistream::slice_requests(dims);
  default:
    throw std::runtime_error("ModelAlgoFactory: unknown attention implementation");
  }
  for (SliceRequest &request : KvCacheAttention::slice_requests(dims)) {
    requests.push_back(std::move(request));
  }
  return requests;
}

bool ModelAlgoFactory::ffn_needs_activation_buffers() const {
  return cfg_.ffn_impl != FFNImplKind::InplaceFusedBiasRelu &&
         cfg_.ffn_impl != FFNImplKind::FusedLinear;
}
