//===-- lib/Parser/operators.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/operators.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

namespace merkurius::parser {

bool OperatorForm::IsPrefix() const {
  return specifier == Specifier::FX || specifier == Specifier::FY ||
      specifier == Specifier::FXY;
}

int OperatorForm::LeftArgumentPriority() const {
  switch (specifier) {
  case Specifier::FXY:
  case Specifier::XFX:
  case Specifier::XFY:
    return priority + 1;
  case Specifier::YFX:
    return priority;
  case Specifier::FX:
  case Specifier::FY:
    break;
  }
  DIE("a prefix operator with one argument has no left argument");
}

int OperatorForm::RightArgumentPriority() const {
  switch (specifier) {
  case Specifier::FX:
  case Specifier::XFX:
  case Specifier::YFX:
    return priority + 1;
  case Specifier::FY:
  case Specifier::FXY:
  case Specifier::XFY:
    return priority;
  }
  llvm_unreachable("unknown specifier");
}

static llvm::StringMap<OperatorDefinition> BuildOperatorTable() {
  llvm::StringMap<OperatorDefinition> table;
  auto add{[&](llvm::StringRef name, Specifier specifier, int priority) {
    OperatorDefinition &def{table[name]};
    def.name = name;
    OperatorForm form{specifier, priority};
    std::optional<OperatorForm> &slot{form.IsPrefix() ? def.prefix : def.infix};
    CHECK(!slot); // at most one prefix and one infix form per name
    slot = form;
  }};
#define MERK_OPERATOR(name, specifier, priority) \
  add(name, Specifier::specifier, priority);
#include "operators.def"
  return table;
}

const OperatorDefinition *FindOperator(llvm::StringRef name) {
  static const llvm::StringMap<OperatorDefinition> table{BuildOperatorTable()};
  auto iter{table.find(name)};
  return iter == table.end() ? nullptr : &iter->second;
}

llvm::StringRef GetSpecifierName(Specifier specifier) {
  switch (specifier) {
  case Specifier::FX:
    return "fx";
  case Specifier::FY:
    return "fy";
  case Specifier::FXY:
    return "fxy";
  case Specifier::XFX:
    return "xfx";
  case Specifier::XFY:
    return "xfy";
  case Specifier::YFX:
    return "yfx";
  }
  llvm_unreachable("unknown specifier");
}

} // namespace merkurius::parser
