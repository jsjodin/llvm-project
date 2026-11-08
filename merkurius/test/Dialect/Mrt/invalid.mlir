// The mrt verifiers. Each module is checked on its own.
//
// RUN: split-file --leading-lines %s %t
// RUN: not merkc -fc1 -emit-merk %t/tag.mlir 2>&1 | FileCheck %s --check-prefix=TAG
// RUN: not merkc -fc1 -emit-merk %t/words.mlir 2>&1 | FileCheck %s --check-prefix=WORDS
// RUN: not merkc -fc1 -emit-merk %t/field.mlir 2>&1 | FileCheck %s --check-prefix=FIELD

//--- tag.mlir
// TAG: tag.mlir:[[@LINE+3]]:10: error: 'mrt.tag' op attribute 'tag' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative whose maximum value is 7
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%c: !llvm.ptr) -> !llvm.ptr attributes {determinism = #merk.determinism<det>} {
    %t = mrt.tag %c, 8 : !llvm.ptr
    merk.succeed %t : !llvm.ptr
  }
}

//--- words.mlir
// WORDS: words.mlir:[[@LINE+3]]:10: error: 'mrt.alloc' op attribute 'words' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() -> !llvm.ptr attributes {determinism = #merk.determinism<det>} {
    %c = mrt.alloc 0 : !llvm.ptr
    merk.succeed %c : !llvm.ptr
  }
}

//--- field.mlir
// FIELD: field.mlir:[[@LINE+3]]:10: error: 'mrt.field_load' op has a field of type 'i128', which is wider than a word
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%c: !llvm.ptr) -> i128 attributes {determinism = #merk.determinism<det>} {
    %x = mrt.field_load %c[0] : !llvm.ptr, i128
    merk.succeed %x : i128
  }
}
