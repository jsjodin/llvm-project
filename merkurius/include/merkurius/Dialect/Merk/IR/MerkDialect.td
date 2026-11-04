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
    it is lowered. Arithmetic on primitive values is the arith dialect's.
  }];
  let cppNamespace = "::merkurius::merk";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

// A Mercury string: in C, MR_String, a pointer to UTF-8 ending in a NUL
// (reference manual, "C data passing conventions").
def Merk_StringType : TypeDef<Merk_Dialect, "String"> {
  let mnemonic = "string";
  let summary = "a Mercury string";
}

// A Mercury array, array(T): a pointer to its size, a word, followed by its
// elements, a word each, so that one layout serves every element type and
// the elements are the contiguous buffer an OpenMP map maps (design
// document, "Type representation").
def Merk_ArrayType : TypeDef<Merk_Dialect, "Array"> {
  let mnemonic = "array";
  let summary = "a Mercury array";
  let parameters = (ins "mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
}

// A value of a type defined by constructors, or of a tuple type, named by
// the type as written fully qualified, with its arguments:
// !merk.term<"list.list(int)">. The type's constructors are those of the
// module's merk.type_def of that name; its layout is chosen when it is
// lowered (design document, "Type representation").
def Merk_TermType : TypeDef<Merk_Dialect, "Term"> {
  let mnemonic = "term";
  let summary = "a value of a type defined by constructors";
  let parameters = (ins StringRefParameter<"the type">:$name);
  let assemblyFormat = "`<` $name `>`";
}

// A value of a type variable: one word that polymorphic code passes without
// looking inside it (design document, "Type representation").
def Merk_WordType : TypeDef<Merk_Dialect, "Word"> {
  let mnemonic = "word";
  let summary = "a value of a type variable";
}

// A run-time description of a type, a tagged pointer: tag 0 for a descriptor
// of a type of arity 0, tag 1 for a cell of a descriptor and its arguments'
// type_infos (design document, "Type descriptors").
def Merk_TypeInfoType : TypeDef<Merk_Dialect, "TypeInfo"> {
  let mnemonic = "type_info";
  let summary = "a run-time description of a type";
}

// A constructor of a type, as merk.type_def lists them: its name and the
// types of its arguments, none for a constant.
def Merk_ConstructorAttr : AttrDef<Merk_Dialect, "Constructor"> {
  let mnemonic = "constructor";
  let summary = "a constructor and its arguments' types";
  let parameters = (ins StringRefParameter<"its name">:$name,
      OptionalArrayRefParameter<"mlir::Type">:$fields);
  let assemblyFormat = "`<` $name (`(` $fields^ `)`)? `>`";
}

// A constructor named by its name and arity, which together tell it from
// the type's others.
def Merk_FunctorAttr : AttrDef<Merk_Dialect, "Functor"> {
  let mnemonic = "functor";
  let summary = "a constructor's name and arity";
  let parameters = (ins StringRefParameter<"its name">:$name,
      "unsigned":$arity);
  let assemblyFormat = "`<` $name `,` $arity `>`";
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
