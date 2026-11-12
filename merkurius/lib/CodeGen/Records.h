//===-- lib/CodeGen/Records.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_LIB_CODEGEN_RECORDS_H_
#define MERKURIUS_LIB_CODEGEN_RECORDS_H_

// Which term types are flat records, values by LLVM struct and inline in
// arrays (docs/INPROGRESS.md, Flat records): a type of one constructor of
// two or more fields, with no hidden ones, not exported abstractly, each
// field an integer, a float, a character, an enumeration that is not
// exported abstractly, a notag of one of those, or such a record. The rule
// reads only a module's merk.type_defs, so every module that sees a type's
// constructors decides alike, and MLIRGen and lowering agree; the runtime
// agrees through the descriptors' flags (runtime/record.h).

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/BuiltinOps.h"

namespace merkurius::codegen {

class RecordRule {
public:
  // Reads the module's merk.type_defs, which lowering erases after.
  explicit RecordRule(mlir::ModuleOp module);
  bool IsRecord(llvm::StringRef name) const;
  // Whether a value of the type can be a record's field.
  bool IsFlatField(mlir::Type field) const;

private:
  struct Def {
    bool abstract;
    // Its constructors' fields, the hidden ones first.
    llvm::SmallVector<llvm::SmallVector<mlir::Type>> constructors;
    llvm::SmallVector<unsigned> hidden;
  };
  // `path` has the types whose fields are being checked, whose own term in a
  // field makes a recursive type.
  bool IsRecord(llvm::StringRef name, llvm::StringSet<> &path) const;
  bool IsFlatField(mlir::Type field, llvm::StringSet<> &path) const;
  llvm::StringMap<Def> defs_;
};

} // namespace merkurius::codegen

#endif // MERKURIUS_LIB_CODEGEN_RECORDS_H_
