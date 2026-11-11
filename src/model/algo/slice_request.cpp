// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "slice_request.hpp"

#include <litnicelm_utils/checks.hpp>

void NodeSlices::add(const std::string &name, TensorView view) {
  views_.emplace_back(name, view);
}

const TensorView &NodeSlices::at(const std::string &name) const {
  for (const auto &[slice_name, view] : views_) {
    if (slice_name == name) {
      return view;
    }
  }
  checker().check(CheckKind::Layout, false, "NodeSlices",
                  [&] { return "no slice was requested as " + name; });
  return views_.front().second;  // unreachable: the check throws
}

TensorView NodeSlices::find(const std::string &name) const {
  for (const auto &[slice_name, view] : views_) {
    if (slice_name == name) {
      return view;
    }
  }
  return TensorView();
}
