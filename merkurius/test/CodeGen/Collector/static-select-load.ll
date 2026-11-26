; A pointer read from a field of a select of static terms, as SimplifyCFG
; makes of a switch that returns one of several constant terms, is static
; data too, which the collector ignores (-1). Used after a safepoint, it had
; no type there, and compiling stopped. Found by the codegen oracle.
;
; RUN: merkc -fc1 -O2 -fdebug-dump-statepoints %s -o - | FileCheck %s
;
; CHECK-LABEL: define tailcc i64 @static.pick(
; CHECK: %inner = load ptr addrspace(1), ptr addrspace(1) %field
; CHECK: gc.statepoint{{.*}}@merk_alloc_slow{{.*}}[ "gc-live"(ptr addrspace(1) %inner), "deopt"(i64 1, i64 -1) ]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

@__LLVM_StackMaps = external constant i8
@merk.root_types = private constant { ptr, i64 } { ptr @__LLVM_StackMaps, i64 0 }, section "merk_root_types", align 8
@merk.term.0 = private unnamed_addr constant { i64, i64 } { i64 1, i64 2 }, align 8
@merk.term.1 = private unnamed_addr constant { ptr addrspace(1), i64 } { ptr addrspace(1) addrspacecast (ptr @merk.term.0 to ptr addrspace(1)), i64 3 }, align 8
@merk.term.2 = private unnamed_addr constant { ptr addrspace(1), i64 } { ptr addrspace(1) addrspacecast (ptr @merk.term.0 to ptr addrspace(1)), i64 4 }, align 8
@llvm.used = appending global [1 x ptr] [ptr @merk.root_types], section "llvm.metadata"

define tailcc i64 @static.pick(i1 %first) {
  %term = select i1 %first, ptr addrspace(1) getelementptr (i8, ptr addrspace(1) addrspacecast (ptr @merk.term.1 to ptr addrspace(1)), i64 5), ptr addrspace(1) getelementptr (i8, ptr addrspace(1) addrspacecast (ptr @merk.term.2 to ptr addrspace(1)), i64 5)
  %field = getelementptr i8, ptr addrspace(1) %term, i64 -5
  %inner = load ptr addrspace(1), ptr addrspace(1) %field, align 8
  %cell = call ptr addrspace(1) @merk_alloc_slow(i64 16) [ "deopt"() ]
  %value = load i64, ptr addrspace(1) %inner, align 8
  ret i64 %value
}

declare ptr addrspace(1) @merk_alloc_slow(i64)

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"merk.ir_prepared", i32 1}
