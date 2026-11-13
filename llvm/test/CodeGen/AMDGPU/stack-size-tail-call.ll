; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa < %s | FileCheck %s

; A tail call's callee replaces its caller's frame, so a function's private
; segment is the larger of its own frame plus its ordinary callees' largest,
; and its tail callees' largest, not the sum of all of them.

declare void @use(ptr addrspace(5))

; CHECK-LABEL: {{^}}big:
; CHECK: .set .Lbig.private_seg_size, 272
define void @big() {
  %a = alloca [64 x i32], addrspace(5)
  call void @use(ptr addrspace(5) %a)
  ret void
}

; Only a tail call: the callee's frame alone.
; CHECK-LABEL: {{^}}tail_only:
; CHECK: .set .Ltail_only.private_seg_size, max(.Lbig.private_seg_size, {{[0-9]+}})
define void @tail_only() {
  tail call void @big()
  ret void
}

; An ordinary call and a tail call: the ordinary callee nests below the
; caller's frame, the tail callee does not.
; CHECK-LABEL: {{^}}both:
; CHECK: .set .Lboth.private_seg_size, max(.Lbig.private_seg_size, {{[0-9]+}})
define void @both() {
  %a = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %a
  call void @use(ptr addrspace(5) %a)
  tail call void @big()
  ret void
}

; The kernel's figure follows: big's frame, not tail_only's plus big's.
; CHECK-LABEL: {{^}}kernel:
; CHECK: .set .Lkernel.private_seg_size, 0+max(.Ltail_only.private_seg_size)
define amdgpu_kernel void @kernel() {
  call void @tail_only()
  ret void
}
