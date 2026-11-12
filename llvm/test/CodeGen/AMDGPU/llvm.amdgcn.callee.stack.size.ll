; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel -mtriple=amdgpu10.30-amd-amdhsa < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -filetype=obj < %s | llvm-objdump -d --mcpu=gfx1030 - | FileCheck %s --check-prefix=OBJ

; llvm.amdgcn.callee.stack.size is the stack that a function's calls may
; take below its frame: the most that a callee takes with its own callees.
; It is always a literal, as the pseudo is 8 bytes, even when the value would
; fit an inline constant.

declare i32 @llvm.amdgcn.callee.stack.size()

@out = addrspace(1) global i32 0

; A function without calls has none below its frame.
; CHECK-LABEL: leaf:
; CHECK: s_mov_b32 s{{[0-9]+}}, lit(0x0)
; CHECK: .set .Lleaf.private_seg_size, 84
; OBJ-LABEL: <leaf>:
; OBJ: s_mov_b32 s{{[0-9]+}}, lit(0x0) {{.*}} {{[0-9A-F]+}} 00000000
define internal void @leaf() noinline {
  %s = call i32 @llvm.amdgcn.callee.stack.size()
  store volatile i32 %s, ptr addrspace(1) @out
  %a = alloca [20 x i32], align 4, addrspace(5)
  %p = getelementptr i32, ptr addrspace(5) %a, i32 3
  store volatile i32 1, ptr addrspace(5) %p
  ret void
}

; A recursive function: its frame is 64 bytes, and below it are leaf's 84;
; the call back into itself takes none.
; CHECK-LABEL: rec:
; CHECK: s_mov_b32 s{{[0-9]+}}, lit(0x54)
; CHECK: .set .Lrec.private_seg_size, 64+max(.Lleaf.private_seg_size)
; OBJ-LABEL: <rec>:
; OBJ: s_mov_b32 s{{[0-9]+}}, 0x54 {{.*}} {{[0-9A-F]+}} 00000054
define internal i32 @rec(i32 %n) {
  %s = call i32 @llvm.amdgcn.callee.stack.size()
  store volatile i32 %s, ptr addrspace(1) @out
  %a = alloca [10 x i32], align 4, addrspace(5)
  %p = getelementptr i32, ptr addrspace(5) %a, i32 %n
  store volatile i32 %n, ptr addrspace(5) %p
  call void @leaf()
  %c = icmp eq i32 %n, 0
  br i1 %c, label %done, label %more
more:
  %m = sub i32 %n, 1
  %x = call i32 @rec(i32 %m)
  %y = add i32 %x, %n
  ret i32 %y
done:
  ret i32 0
}

; The kernel has no frame, and rec takes 64 + 84 below it.
; CHECK-LABEL: k:
; CHECK: s_mov_b32 s{{[0-9]+}}, lit(0x94)
; OBJ-LABEL: <k>:
; OBJ: s_mov_b32 s{{[0-9]+}}, 0x94 {{.*}} {{[0-9A-F]+}} 00000094
define amdgpu_kernel void @k(i32 %n) {
  %v = call i32 @rec(i32 %n)
  %s = call i32 @llvm.amdgcn.callee.stack.size()
  store volatile i32 %s, ptr addrspace(1) @out
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
