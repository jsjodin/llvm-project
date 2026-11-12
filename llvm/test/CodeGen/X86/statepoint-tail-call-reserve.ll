; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; caller's musttail call passes more stack arguments than caller received, so
; it reserves room for them (a negative TCReturnAddrDelta) before it pushes
; the frame pointer. At -O0 the statepoint's gc-live value is spilled, and
; the stack map gives its slot from SP, which StackSize includes the reserve
; in; this used to hit an assertion in getFrameIndexReferencePreferSP.

; CHECK-LABEL: caller:
; CHECK:         subq $16, %rsp
; CHECK:         pushq %rbp
; CHECK:         movq %rsp, %rbp
; CHECK:         subq $16, %rsp
; CHECK-NEXT:    movq %rdi, -8(%rbp)
; CHECK-NEXT:    callq f@PLT

; The slot is rbp - 8 = rsp + 8, in a frame of 40 bytes.
; CHECK-LABEL: __LLVM_StackMaps:
; CHECK:         .quad caller
; CHECK-NEXT:    .quad 40
; CHECK:         .byte 3
; CHECK-NEXT:    .byte 0
; CHECK-NEXT:    .short 8
; CHECK-NEXT:    .short 7
; CHECK-NEXT:    .short 0
; CHECK-NEXT:    .long 8

declare tailcc void @callee(i64, i64, i64, i64, i64, i64, i64, i64, ptr addrspace(1))
declare void @f()

define tailcc void @caller(ptr addrspace(1) %p) #0 gc "statepoint-example" {
  %t = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @f, i32 0, i32 0, i32 0, i32 0) [ "gc-live"(ptr addrspace(1) %p) ]
  %r = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %t, i32 0, i32 0)
  musttail call tailcc void @callee(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, ptr addrspace(1) %r)
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
attributes #0 = { "frame-pointer"="all" }
