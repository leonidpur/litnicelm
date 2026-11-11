// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "precision_policy.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// What a node keeps in a slice.
enum class SliceRole : uint8_t {
  Saved,      // written in forward, read in backward
  Workspace,  // scratch within one forward or backward call
  Output,     // handed to the next node (owned by the composite)
};

// How long a slice's contents must survive.
enum class SliceLifetime : uint8_t {
  Step,  // from forward to the matching backward
  Op,    // one forward or backward call; may share memory with other Op slices
};

// One tensor a node needs, declared before memory is laid out. Never
// checkpointed. Parameters are not requested here: they belong to the node
// kind and keep their stable names in the parameter layout.
struct SliceRequest {
  std::string name;  // node-local, e.g. "attn.scores"; laid out as "layer{i}." + name
  Shape shape;       // the largest call's shape; calls use leading prefixes
  ItemType item = ItemType::Activation;  // dtype comes from the PrecisionPolicy
  SliceRole role = SliceRole::Workspace;
  SliceLifetime lifetime = SliceLifetime::Op;
};

// The views granted to one node instance, in request order. Each view has
// the requested shape; a call takes the prefix it needs (slice_views.hpp).
class NodeSlices {
public:
  void add(const std::string &name, TensorView view);
  // Throws (Layout check) when the node did not request the name.
  const TensorView &at(const std::string &name) const;
  // An empty view when the name was not requested (backward-only slices of
  // an inference layout).
  TensorView find(const std::string &name) const;
  const std::vector<std::pair<std::string, TensorView>> &views() const {
    return views_;
  }

private:
  std::vector<std::pair<std::string, TensorView>> views_;
};
