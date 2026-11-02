//===-- MerkDialect.td - The merk dialect ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The merk dialect, its attributes, and the base class of its operations
// (design document, "MLIR dialects").
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_TD
#define MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Merk_Dialect : Dialect {
  let name = "merk";
  let summary = "Mercury procedures, terms and control";
  let description = [{
    The merk dialect carries Mercury's semantics: procedures with their
    determinism, calls, terms by constructor, switches and the other
    goals. Its types are Mercury types, not layouts; layout is chosen when
    it is lowered.
  }];
  let cppNamespace = "::merkurius::merk";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

// A Mercury string: in C, MR_String, a pointer to UTF-8 ending in a NUL
// (reference manual, "C data passing conventions").
def Merk_StringType : TypeDef<Merk_Dialect, "String"> {
  let mnemonic = "string";
  let summary = "a Mercury string";
}

// A procedure's determinism (reference manual, "Determinism categories").
def Merk_Determinism : I32EnumAttr<"Determinism", "a Mercury determinism", [
  I32EnumAttrCase<"Det", 0, "det">,
  I32EnumAttrCase<"Semidet", 1, "semidet">,
  I32EnumAttrCase<"Multi", 2, "multi">,
  I32EnumAttrCase<"Nondet", 3, "nondet">,
  I32EnumAttrCase<"CcMulti", 4, "cc_multi">,
  I32EnumAttrCase<"CcNondet", 5, "cc_nondet">,
  I32EnumAttrCase<"Erroneous", 6, "erroneous">,
  I32EnumAttrCase<"Failure", 7, "failure">
]> {
  let cppNamespace = "::merkurius::merk";
  let genSpecializedAttr = 0;
}

def Merk_DeterminismAttr : EnumAttr<Merk_Dialect, Merk_Determinism,
    "determinism">;

// The target's numbers for the address spaces that Mercury code uses: the
// generic one, which every pointer converts to, and those of global,
// workgroup-shared, constant and per-thread private memory. On a CPU target
// they are all 0. With the triple and the data layout, it is what the
// module records of its target (TargetInfo), so that lowering a module read
// back from its text needs nothing else.
def Merk_AddressSpacesAttr : AttrDef<Merk_Dialect, "AddressSpaces"> {
  let mnemonic = "address_spaces";
  let summary = "the target's address spaces for Mercury code";
  let parameters = (ins "unsigned":$generic, "unsigned":$global,
      "unsigned":$shared, "unsigned":$constant, "unsigned":$private_);
  let assemblyFormat = [{
    `<` `generic` `=` $generic `,` `global` `=` $global `,`
    `shared` `=` $shared `,` `constant` `=` $constant `,`
    `private` `=` $private_ `>`
  }];
}

class Merk_Op<string mnemonic, list<Trait> traits = []>
    : Op<Merk_Dialect, mnemonic, traits>;

#endif // MERKURIUS_DIALECT_MERK_IR_MERKDIALECT_TD
