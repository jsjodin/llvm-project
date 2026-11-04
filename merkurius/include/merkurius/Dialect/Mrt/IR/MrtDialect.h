//===-- include/merkurius/Dialect/Mrt/IR/MrtDialect.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_H_
#define MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_H_

// The mrt dialect (design document, "MLIR dialects"): its operations.

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "merkurius/Dialect/Mrt/IR/MrtDialect.h.inc"

#define GET_OP_CLASSES
#include "merkurius/Dialect/Mrt/IR/MrtOps.h.inc"

#endif // MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_H_
