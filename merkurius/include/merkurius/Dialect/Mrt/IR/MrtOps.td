//===-- MrtOps.td - The mrt dialect's operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The mrt dialect's operations (design document, "MLIR dialects"). Only
// those that lowering makes so far are defined: allocation and the cells
// of terms. A cell is headerless, a word for each of its fields; a pointer
// to one has a 3-bit primary tag in its low bits, which a word's alignment
// leaves free (design document, "Type representation").
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MRT_IR_MRTOPS_TD
#define MERKURIUS_DIALECT_MRT_IR_MRTOPS_TD

include "merkurius/Dialect/Mrt/IR/MrtDialect.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// A primary tag: 0 to 7.
def Mrt_TagAttr : ConfinedAttr<I64Attr, [IntNonNegative, IntMaxValue<7>]>;

// What a field holds: a word or less. A narrower integer is zero-extended
// to its word, so that polymorphic code can pass the word as it is.
def Mrt_FieldType : AnyTypeOf<[AnySignlessInteger, F64, LLVM_AnyPointer]>;

def Mrt_AllocOp : Mrt_Op<"alloc"> {
  let summary = "allocate a cell on the heap";
  let description = [{
    A cell of `words` words, aligned to a word, whose fields are undefined
    until they are stored. Lowering bumps the heap's pointer, and calls the
    runtime's `merk_alloc_slow` when that passes the heap's limit.

    ```mlir
    %cell = mrt.alloc 2 : !llvm.ptr<1>
    ```
  }];
  let arguments = (ins ConfinedAttr<I64Attr, [IntPositive]>:$words);
  let results = (outs Res<LLVM_AnyPointer, "",
      [MemAlloc<DefaultResource>]>:$cell);
  let assemblyFormat = "$words attr-dict `:` qualified(type($cell))";
}

def Mrt_TagOp : Mrt_Op<"tag", [Pure, AllTypesMatch<["cell", "term"]>]> {
  let summary = "a pointer to a cell, tagged";
  let description = [{
    ```mlir
    %t = mrt.tag %cell, 1
    ```
  }];
  let arguments = (ins LLVM_AnyPointer:$cell, Mrt_TagAttr:$tag);
  let results = (outs LLVM_AnyPointer:$term);
  let assemblyFormat = "$cell `,` $tag attr-dict `:` qualified(type($cell))";
  let builders = [OpBuilder<(ins "::mlir::Value":$cell, "std::int64_t":$tag), [{
    build($_builder, $_state, cell.getType(), cell,
        $_builder.getI64IntegerAttr(tag));
  }]>];
}

def Mrt_UntagOp : Mrt_Op<"untag", [Pure, AllTypesMatch<["cell", "term"]>]> {
  let summary = "the pointer to a cell, from its tagged pointer";
  let description = [{
    The term's tag is `tag`: a switch or a test has found it to be.

    ```mlir
    %cell = mrt.untag %t, 1
    ```
  }];
  let arguments = (ins LLVM_AnyPointer:$term, Mrt_TagAttr:$tag);
  let results = (outs LLVM_AnyPointer:$cell);
  let assemblyFormat = "$term `,` $tag attr-dict `:` qualified(type($term))";
  let builders = [OpBuilder<(ins "::mlir::Value":$term, "std::int64_t":$tag), [{
    build($_builder, $_state, term.getType(), term,
        $_builder.getI64IntegerAttr(tag));
  }]>];
}

def Mrt_PrimaryTagOp : Mrt_Op<"primary_tag", [Pure]> {
  let summary = "the primary tag of a term";
  let description = [{
    The low bits of the term's word, of a pointer to a cell or of a
    constant, whose tag is 0.

    ```mlir
    %tag = mrt.primary_tag %t
    ```
  }];
  let arguments = (ins LLVM_AnyPointer:$term);
  let results = (outs I64:$tag);
  let assemblyFormat = "$term attr-dict `:` qualified(type($term))";
  let builders = [OpBuilder<(ins "::mlir::Value":$term), [{
    build($_builder, $_state, $_builder.getI64Type(), term);
  }]>];
}

def Mrt_FieldLoadOp : Mrt_Op<"field_load"> {
  let summary = "load a cell's field";
  let description = [{
    The field in the cell's word `word`, the first being 0.

    ```mlir
    %h = mrt.field_load %cell[0] : i64
    ```
  }];
  let arguments = (ins Arg<LLVM_AnyPointer, "", [MemRead]>:$cell,
      ConfinedAttr<I64Attr, [IntNonNegative]>:$word);
  let results = (outs Mrt_FieldType:$value);
  let assemblyFormat = "$cell `[` $word `]` attr-dict `:` qualified(type($cell)) `,` type($value)";
  let hasVerifier = 1;
}

def Mrt_FieldStoreOp : Mrt_Op<"field_store"> {
  let summary = "store a cell's field";
  let description = [{
    Only into a cell just allocated: a term is immutable once it is made.

    ```mlir
    mrt.field_store %h, %cell[0] : i64
    ```
  }];
  let arguments = (ins Mrt_FieldType:$value,
      Arg<LLVM_AnyPointer, "", [MemWrite]>:$cell,
      ConfinedAttr<I64Attr, [IntNonNegative]>:$word);
  let assemblyFormat = [{
    $value `,` $cell `[` $word `]` attr-dict `:` type($value) `,` qualified(type($cell))
  }];
  let hasVerifier = 1;
}

#endif // MERKURIUS_DIALECT_MRT_IR_MRTOPS_TD
