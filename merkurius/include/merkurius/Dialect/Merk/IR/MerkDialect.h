//===-- include/merkurius/Dialect/Merk/IR/MerkDialect.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_H_
#define MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_H_

// The merk dialect (design document, "MLIR dialects"): its attributes and
// operations.

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "merkurius/Dialect/Merk/IR/MerkDialect.h.inc"
#include "merkurius/Dialect/Merk/IR/MerkEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkTypes.h.inc"

#define GET_OP_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkOps.h.inc"

#endif // MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_H_
