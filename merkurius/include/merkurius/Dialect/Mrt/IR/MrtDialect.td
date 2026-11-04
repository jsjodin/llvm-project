//===-- MrtDialect.td - The mrt dialect --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The mrt dialect and the base class of its operations (design document,
// "MLIR dialects").
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_TD
#define MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_TD

include "mlir/IR/OpBase.td"

def Mrt_Dialect : Dialect {
  let name = "mrt";
  let summary = "Mercury's runtime operations, on concrete layouts";
  let description = [{
    The mrt dialect is merk's terms with their layouts chosen: words and
    pointers to headerless cells, tagged in their low bits, allocated on
    the heap. Its values are the LLVM dialect's, a cell an `!llvm.ptr`.
    Every allocation is an `mrt.alloc`, so that a collector needs to hook
    only its slow path.
  }];
  let cppNamespace = "::merkurius::mrt";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

class Mrt_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mrt_Dialect, mnemonic, traits>;

#endif // MERKURIUS_DIALECT_MRT_IR_MRTDIALECT_TD
