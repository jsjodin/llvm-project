//===-- lib/CodeGen/Records.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Records.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"

namespace merkurius::codegen {

RecordRule::RecordRule(mlir::ModuleOp module) {
  for (auto def : module.getOps<merk::TypeDefOp>()) {
    Def &entry{defs_[def.getName()]};
    entry.abstract = def.getAbstract();
    for (auto constructor :
        def.getConstructors().getAsRange<merk::ConstructorAttr>()) {
      entry.constructors.emplace_back(
          constructor.getFields().begin(), constructor.getFields().end());
      entry.hidden.push_back(constructor.getHidden());
    }
  }
}

bool RecordRule::IsRecord(llvm::StringRef name) const {
  llvm::StringSet<> path;
  return IsRecord(name, path);
}

bool RecordRule::IsFlatField(mlir::Type field) const {
  llvm::StringSet<> path;
  return IsFlatField(field, path);
}

bool RecordRule::IsFlatField(mlir::Type field, llvm::StringSet<> &path) const {
  if (mlir::isa<mlir::IntegerType, mlir::FloatType>(field)) {
    return true;
  }
  auto term{mlir::dyn_cast<merk::TermType>(field)};
  if (!term) {
    return false;
  }
  auto it{defs_.find(term.getName())};
  if (it == defs_.end() || it->second.abstract) {
    return false;
  }
  const Def &def{it->second};
  // An enumeration, a word.
  if (llvm::all_of(
          def.constructors, [](const llvm::SmallVector<mlir::Type> &fields) {
            return fields.empty();
          })) {
    return true;
  }
  // A notag, its argument: a number or a word, not a record's cell.
  if (def.constructors.size() == 1 && def.constructors.front().size() == 1) {
    if (!path.insert(term.getName()).second) {
      return false;
    }
    mlir::Type argument{def.constructors.front().front()};
    auto inner{mlir::dyn_cast<merk::TermType>(argument)};
    bool flat{IsFlatField(argument, path) &&
        !(inner && IsRecord(inner.getName(), path))};
    path.erase(term.getName());
    return flat;
  }
  return IsRecord(term.getName(), path);
}

bool RecordRule::IsRecord(llvm::StringRef name, llvm::StringSet<> &path) const {
  auto it{defs_.find(name)};
  if (it == defs_.end() || it->second.abstract) {
    return false;
  }
  const Def &def{it->second};
  if (def.constructors.size() != 1 || def.hidden.front() != 0 ||
      def.constructors.front().size() < 2 || !path.insert(name).second) {
    return false;
  }
  bool record{llvm::all_of(def.constructors.front(),
      [&](mlir::Type field) { return IsFlatField(field, path); })};
  path.erase(name);
  return record;
}

} // namespace merkurius::codegen
