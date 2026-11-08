// The mrt operations print as they are parsed, and lower to the LLVM
// dialect: mrt.alloc to a bump of the thread's heap pointer, checked
// against its limit, with the runtime's merk_alloc_slow when the cell does
// not fit; a tag to an offset from the cell's address; and a field to its
// word's offset (design document, "Type representation" and "Memory
// management").
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @mrt attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // LLVM: @merk_heap = external thread_local global { ptr, ptr }, align 8

  // CHECK-LABEL: merk.proc @mrt.cons_3_0(%arg0: i64, %arg1: !llvm.ptr<1>) -> !llvm.ptr<1>
  // CHECK-NEXT: %[[CELL:.*]] = mrt.alloc 2 : !llvm.ptr<1>
  // CHECK-NEXT: mrt.field_store %arg0, %[[CELL]][0] : i64, !llvm.ptr<1>
  // CHECK-NEXT: mrt.field_store %arg1, %[[CELL]][1] : !llvm.ptr<1>, !llvm.ptr<1>
  // CHECK-NEXT: %[[T:.*]] = mrt.tag %[[CELL]], 1 : !llvm.ptr<1>
  // CHECK-NEXT: merk.succeed %[[T]]
  // LLVM-LABEL: define tailcc ptr addrspace(1) @mrt.cons_3_0(i64 %0, ptr addrspace(1) %1) {
  // LLVM-NEXT: %[[HEAP:.*]] = call ptr asm sideeffect "movq %fs:0, $0\0A\09leaq merk_heap@tpoff($0), $0", "=r"()
  // LLVM-NEXT: %[[NEXT:.*]] = load ptr addrspace(1), ptr %[[HEAP]], align 8
  // LLVM-NEXT: %[[END:.*]] = getelementptr i8, ptr addrspace(1) %[[NEXT]], i32 16
  // LLVM-NEXT: %[[LIMITP:.*]] = getelementptr { ptr, ptr }, ptr %[[HEAP]], i32 0, i32 1
  // LLVM-NEXT: %[[LIMIT:.*]] = load ptr addrspace(1), ptr %[[LIMITP]], align 8
  // LLVM-NEXT: %[[FITS:.*]] = icmp ule ptr addrspace(1) %[[END]], %[[LIMIT]]
  // LLVM-NEXT: br i1 %[[FITS]], label %[[FAST:.*]], label %[[SLOW:.*]], !prof ![[WEIGHTS:.*]]
  // LLVM: [[FAST]]:
  // LLVM-NEXT: store ptr addrspace(1) %[[END]], ptr %[[HEAP]], align 8
  // LLVM-NEXT: br label %[[JOIN:.*]]
  // LLVM: [[SLOW]]:
  // LLVM-NEXT: %[[NEW:.*]] = call ptr addrspace(1) @merk_alloc_slow(i64 16)
  // LLVM-NEXT: br label %[[JOIN]]
  // LLVM: [[JOIN]]:
  // LLVM-NEXT: %[[C:.*]] = phi ptr addrspace(1) [ %[[NEW]], %[[SLOW]] ], [ %[[NEXT]], %[[FAST]] ]
  // LLVM-NEXT: store i64 %0, ptr addrspace(1) %[[C]], align 8
  // LLVM-NEXT: %[[F1:.*]] = getelementptr i8, ptr addrspace(1) %[[C]], i32 8
  // LLVM-NEXT: store ptr addrspace(1) %1, ptr addrspace(1) %[[F1]], align 8
  // LLVM-NEXT: %[[TAGGED:.*]] = getelementptr i8, ptr addrspace(1) %[[C]], i32 1
  // LLVM-NEXT: ret ptr addrspace(1) %[[TAGGED]]
  merk.proc @mrt.cons_3_0(%h: i64, %t: !llvm.ptr<1>) -> !llvm.ptr<1> attributes {determinism = #merk.determinism<det>} {
    %cell = mrt.alloc 2 : !llvm.ptr<1>
    mrt.field_store %h, %cell[0] : i64, !llvm.ptr<1>
    mrt.field_store %t, %cell[1] : !llvm.ptr<1>, !llvm.ptr<1>
    %l = mrt.tag %cell, 1 : !llvm.ptr<1>
    merk.succeed %l : !llvm.ptr<1>
  }

  // A field narrower than a word is its whole word, zero-extended, whatever
  // its signedness, so that polymorphic code can pass the word as it is.
  // LLVM-LABEL: define tailcc i8 @mrt.narrow_1_0(i8 %0) {
  // LLVM: %[[WORD:.*]] = zext i8 %0 to i64
  // LLVM-NEXT: store i64 %[[WORD]], ptr addrspace(1) %[[N:.*]], align 8
  // LLVM-NEXT: %[[LOADED:.*]] = load i64, ptr addrspace(1) %[[N]], align 8
  // LLVM-NEXT: %[[BACK:.*]] = trunc i64 %[[LOADED]] to i8
  // LLVM-NEXT: ret i8 %[[BACK]]
  merk.proc @mrt.narrow_1_0(%v: i8) -> i8 attributes {determinism = #merk.determinism<det>} {
    %cell = mrt.alloc 1 : !llvm.ptr<1>
    mrt.field_store %v, %cell[0] : i8, !llvm.ptr<1>
    %w = mrt.field_load %cell[0] : !llvm.ptr<1>, i8
    merk.succeed %w : i8
  }

  // CHECK-LABEL: merk.proc @mrt.fields_3_0(%arg0: !llvm.ptr<1>) -> (i64, i64)
  // CHECK-NEXT: %[[TAG:.*]] = mrt.primary_tag %arg0 : !llvm.ptr<1>
  // CHECK-NEXT: %[[CELL:.*]] = mrt.untag %arg0, 1 : !llvm.ptr<1>
  // CHECK-NEXT: %[[H:.*]] = mrt.field_load %[[CELL]][0] : !llvm.ptr<1>, i64
  // CHECK-NEXT: merk.succeed %[[TAG]], %[[H]]
  // LLVM-LABEL: define tailcc { i64, i64 } @mrt.fields_3_0(ptr addrspace(1) %0) {
  // LLVM-NEXT: %[[W:.*]] = ptrtoint ptr addrspace(1) %0 to i64
  // LLVM-NEXT: %[[TAG:.*]] = and i64 %[[W]], 7
  // LLVM-NEXT: %[[CELL:.*]] = getelementptr i8, ptr addrspace(1) %0, i32 -1
  // LLVM-NEXT: %[[H:.*]] = load i64, ptr addrspace(1) %[[CELL]], align 8
  merk.proc @mrt.fields_3_0(%l: !llvm.ptr<1>) -> (i64, i64) attributes {determinism = #merk.determinism<det>} {
    %tag = mrt.primary_tag %l : !llvm.ptr<1>
    %cell = mrt.untag %l, 1 : !llvm.ptr<1>
    %h = mrt.field_load %cell[0] : !llvm.ptr<1>, i64
    merk.succeed %tag, %h : i64, i64
  }

  // LLVM: declare ptr addrspace(1) @merk_alloc_slow(i64)
  // LLVM: ![[WEIGHTS]] = !{!"branch_weights", i32 2000, i32 1}
}
