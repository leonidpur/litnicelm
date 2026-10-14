#pragma once

#include <litnicelm_utils/layout_slice.hpp>
#include <litnicelm_utils/validation.hpp>

#include <string>
#include <utility>
#include <vector>

// Walks a layout's slices in order. Each next() names the slice the caller
// expects there, and finish() requires that none are left, so a store and
// the layout it reads cannot drift apart silently.
class LayoutCursor {
public:
  // who: the store and the layout, e.g. "TensorStore parameter layout".
  LayoutCursor(const std::vector<LayoutSlice> &slices, std::string who)
      : slices_(slices), who_(std::move(who)) {}

  const LayoutSlice &next(const std::string &expected) {
    validator().validate(Check::Layout, index_ < slices_.size(), who_, [&] {
      return "layout ended before slice " + expected;
    });
    const LayoutSlice &slice = slices_[index_++];
    validator().validate(Check::Layout, slice.name == expected, who_, [&] {
      return "expected slice " + expected + ", got " + slice.name;
    });
    return slice;
  }

  void finish() const {
    validator().validate(Check::Layout, index_ == slices_.size(), who_,
                         "layout has unexpected extra slices");
  }

private:
  const std::vector<LayoutSlice> &slices_;
  std::string who_;
  size_t index_ = 0;
};
