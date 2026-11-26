; A phi of a field's address in a static term and in a cell, as GVN's PRE
; makes of a loop's frame state, read after a safepoint. It has no cell to
; list: the rewrite gives it a base phi of the cells, null for the static
; term, which the collector reads as a tally, and moves the pointer by its
; offset. Listed in the bundle, as it was, it would be its own base and a
; tally of its own, 24 bytes into the cell, and a collection would copy the
; cell from there and read past it.
;
; RUN: merkc -fc1 -O2 -fdebug-dump-statepoints %s -o - | FileCheck %s
;
; CHECK-LABEL: define tailcc i64 @interior.probe(
; CHECK: %field.base = phi ptr addrspace(1) [ null, %static ], [ %tally, %heap ]
; CHECK-NEXT: %field = phi ptr addrspace(1)
; CHECK: gc.statepoint{{.*}}@merk_alloc_slow{{.*}}"gc-live"({{.*}}%field.base
; CHECK: gc.relocate{{.*}}; (%field.base, %field)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

@__LLVM_StackMaps = external constant i8
; Root type 0 is a tally.
@merk.root_types = private constant { ptr, i64, i64, ptr, i64, ptr } { ptr @__LLVM_StackMaps, i64 2, i64 1, ptr @"merk.type_ctor.interior.tally/0", i64 0, ptr null }, section "merk_root_types", align 8
@"merk.type_ctor.interior.tally/0" = external constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr, ptr, [8 x i16], ptr, ptr }, align 8
; tally([], [], 0, 0, 0)
@merk.term.0 = private unnamed_addr constant { ptr, ptr, i64, i64, i64 } zeroinitializer, align 8
@llvm.used = appending global [1 x ptr] [ptr @merk.root_types], section "llvm.metadata"

define tailcc i64 @interior.probe(i1 %first, ptr addrspace(1) "merk.root_type"="0" %tally) {
entry:
  br i1 %first, label %static, label %heap

static:
  br label %join

heap:
  %in.cell = getelementptr i8, ptr addrspace(1) %tally, i64 24
  br label %join

join:
  %field = phi ptr addrspace(1) [ getelementptr (i8, ptr addrspace(1) addrspacecast (ptr @merk.term.0 to ptr addrspace(1)), i64 24), %static ], [ %in.cell, %heap ]
  %cell = call ptr addrspace(1) @merk_alloc_slow(i64 16) [ "deopt"() ]
  %value = load i64, ptr addrspace(1) %field, align 8
  ret i64 %value
}

declare ptr addrspace(1) @merk_alloc_slow(i64)

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"merk.ir_prepared", i32 1}
