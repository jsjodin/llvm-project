//===-- lib/Dialect/Mrt/IR/MrtDialect.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Dialect/Mrt/IR/MrtDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace merkurius::mrt;

#include "merkurius/Dialect/Mrt/IR/MrtDialect.cpp.inc"

void MrtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "merkurius/Dialect/Mrt/IR/MrtOps.cpp.inc"
      >();
}

// A field is a word at most.
static llvm::LogicalResult VerifyField(mlir::Operation *op, mlir::Type type) {
  if (auto integer{mlir::dyn_cast<mlir::IntegerType>(type)};
      integer && integer.getWidth() > 64) {
    return op->emitOpError("has a field of type ")
        << type << ", which is wider than a word";
  }
  return llvm::success();
}

llvm::LogicalResult FieldLoadOp::verify() {
  return VerifyField(*this, getValue().getType());
}

llvm::LogicalResult FieldStoreOp::verify() {
  return VerifyField(*this, getValue().getType());
}

#define GET_OP_CLASSES
#include "merkurius/Dialect/Mrt/IR/MrtOps.cpp.inc"
