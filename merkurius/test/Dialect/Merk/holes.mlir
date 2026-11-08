// The operations of last call modulo constructor (design document,
// Lowering, "Deterministic and semidet code", "Last call modulo
// constructor") print as they are parsed. A
// hole is null until it is filled, merk.field_ref is the term's pointer and the
// field's offset from it, past the tag, and merk.ref_store stores the value
// there, a narrow integer as its whole word.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @holes attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]

  // CHECK-LABEL: merk.proc @holes.fill_3_0(%arg0: i64, %arg1: !merk.ref, %arg2: i64)
  // CHECK-NEXT: %[[H:.*]] = merk.hole : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[C:.*]] = merk.construct #merk.functor<"[|]", 2>(%arg0, %[[H]] : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.ref_store %[[C]], %arg1[%arg2] : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[F:.*]], %[[O:.*]] = merk.field_ref %[[C]], #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[N:.*]] = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.ref_store %[[N]], %[[F]][%[[O]]] : !merk.term<"list.list(int)">
  // LLVM-LABEL: define tailcc void @holes.fill_3_0(i64 %0, ptr addrspace(1) %1, i64 %2) {
  // LLVM: %[[CELL:.*]] = phi ptr addrspace(1)
  // LLVM-NEXT: store i64 %0, ptr addrspace(1) %[[CELL]], align 8
  // LLVM-NEXT: %[[HOLE:.*]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 8
  // LLVM-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[HOLE]], align 8
  // LLVM-NEXT: %[[TERM:.*]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 1
  // LLVM-NEXT: %[[FIRST:.*]] = getelementptr i8, ptr addrspace(1) %1, i64 %2
  // LLVM-NEXT: store ptr addrspace(1) %[[TERM]], ptr addrspace(1) %[[FIRST]], align 8
  // LLVM-NEXT: %[[FIELD:.*]] = getelementptr i8, ptr addrspace(1) %[[TERM]], i64 7
  // LLVM-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[FIELD]], align 8
  // LLVM-NEXT: ret void
  merk.proc @holes.fill_3_0(%x: i64, %r: !merk.ref, %o: i64) attributes {determinism = #merk.determinism<det>} {
    %h = merk.hole : !merk.term<"list.list(int)">
    %c = merk.construct #merk.functor<"[|]", 2>(%x, %h : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    merk.ref_store %c, %r[%o] : !merk.term<"list.list(int)">
    %f, %fo = merk.field_ref %c, #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
    %n = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
    merk.ref_store %n, %f[%fo] : !merk.term<"list.list(int)">
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @holes.narrow_3_0(%arg0: i8, %arg1: !merk.ref, %arg2: i64)
  // CHECK-NEXT: merk.ref_store %arg0, %arg1[%arg2] : i8
  // LLVM-LABEL: define tailcc void @holes.narrow_3_0(i8 %0, ptr addrspace(1) %1, i64 %2) {
  // LLVM-NEXT: %[[WORD:.*]] = zext i8 %0 to i64
  // LLVM-NEXT: %[[AT:.*]] = getelementptr i8, ptr addrspace(1) %1, i64 %2
  // LLVM-NEXT: store i64 %[[WORD]], ptr addrspace(1) %[[AT]], align 8
  merk.proc @holes.narrow_3_0(%x: i8, %r: !merk.ref, %o: i64) attributes {determinism = #merk.determinism<det>} {
    merk.ref_store %x, %r[%o] : i8
    merk.succeed
  }
}
