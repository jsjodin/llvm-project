; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck %s

; SGPRs live across a call that clobbers every register but s30-s34 go to
; lanes of a VGPR. SILowerSGPRSpills picks the highest free VGPR, and
; after allocation the VGPR moves to the lowest free one. The call clobbering
; both does not keep it from moving, as the VGPR is spilled around the call
; anyway. Otherwise the lanes stay in v127 on gfx1030, and the kernel counts
; 128 VGPRs. A call counts as using v0-v31 in a kernel. In a function, the
; VGPR's whole-wave save moves with it.

declare void @no_csr() "no_callee_saved_registers"

; CHECK-LABEL: kernel:
; CHECK: v_writelane_b32 v{{[0-9]}}, s{{[0-9]+}}, 0
; CHECK: s_swappc_b64
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]}}, 0
; CHECK: .set .Lkernel.num_vgpr, max(32, amdgpu.max_num_vgpr)
define amdgpu_kernel void @kernel(ptr addrspace(1) %p, i32 %a, i32 %b) {
  call void @no_csr() "no_callee_saved_registers"
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: function:
; CHECK: v_writelane_b32 v{{[0-9]}}, s{{[0-9]+}}, 0
; CHECK: s_swappc_b64
; CHECK: v_readlane_b32 s{{[0-9]+}}, v{{[0-9]}}, 0
; CHECK: s_setpc_b64
; CHECK: .set .Lfunction.num_vgpr, max({{[0-9]}}, amdgpu.max_num_vgpr)
define void @function(ptr addrspace(1) %p, i32 inreg %a, i32 inreg %b) "no_callee_saved_registers" {
  call void @no_csr() "no_callee_saved_registers"
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %p
  ret void
}
