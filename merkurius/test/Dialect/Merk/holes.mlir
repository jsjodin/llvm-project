// The operations of last call modulo constructor (design document,
// Lowering, "Deterministic and semidet code", "Last call modulo
// constructor") print as they are parsed. A
// hole is null until it is filled, merk.field_ref is the address of the
// field in the cell, past the tag, and merk.ref_store stores the value
// there, a narrow integer as its whole word.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @holes attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]

  // CHECK-LABEL: merk.proc @holes.fill_2_0(%arg0: i64, %arg1: !merk.ref)
  // CHECK-NEXT: %[[H:.*]] = merk.hole : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[C:.*]] = merk.construct #merk.functor<"[|]", 2>(%arg0, %[[H]] : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.ref_store %[[C]], %arg1 : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[F:.*]] = merk.field_ref %[[C]], #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[N:.*]] = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.ref_store %[[N]], %[[F]] : !merk.term<"list.list(int)">
  // LLVM-LABEL: define tailcc void @holes.fill_2_0(i64 %0, ptr %1) {
  // LLVM: %[[CELL:.*]] = phi ptr
  // LLVM-NEXT: store i64 %0, ptr %[[CELL]], align 8
  // LLVM-NEXT: %[[HOLE:.*]] = getelementptr i8, ptr %[[CELL]], i32 8
  // LLVM-NEXT: store ptr null, ptr %[[HOLE]], align 8
  // LLVM-NEXT: %[[TERM:.*]] = getelementptr i8, ptr %[[CELL]], i32 1
  // LLVM-NEXT: store ptr %[[TERM]], ptr %1, align 8
  // LLVM-NEXT: %[[UNTAGGED:.*]] = getelementptr i8, ptr %[[TERM]], i32 -1
  // LLVM-NEXT: %[[FIELD:.*]] = getelementptr i8, ptr %[[UNTAGGED]], i32 8
  // LLVM-NEXT: store ptr null, ptr %[[FIELD]], align 8
  // LLVM-NEXT: ret void
  merk.proc @holes.fill_2_0(%x: i64, %r: !merk.ref) attributes {determinism = #merk.determinism<det>} {
    %h = merk.hole : !merk.term<"list.list(int)">
    %c = merk.construct #merk.functor<"[|]", 2>(%x, %h : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    merk.ref_store %c, %r : !merk.term<"list.list(int)">
    %f = merk.field_ref %c, #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
    %n = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
    merk.ref_store %n, %f : !merk.term<"list.list(int)">
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @holes.narrow_2_0(%arg0: i8, %arg1: !merk.ref)
  // CHECK-NEXT: merk.ref_store %arg0, %arg1 : i8
  // LLVM-LABEL: define tailcc void @holes.narrow_2_0(i8 %0, ptr %1) {
  // LLVM-NEXT: %[[WORD:.*]] = zext i8 %0 to i64
  // LLVM-NEXT: store i64 %[[WORD]], ptr %1, align 8
  merk.proc @holes.narrow_2_0(%x: i8, %r: !merk.ref) attributes {determinism = #merk.determinism<det>} {
    merk.ref_store %x, %r : i8
    merk.succeed
  }
}
