; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck %s

; What IPRA takes a function to keep for its caller includes what frame
; lowering saves apart from the callee-saved registers: the return address,
; the stack pointer, and the frame and base pointers if the function has them;
; but s33 and s34 only then.

declare void @external(ptr addrspace(5))

; Without a frame, s34 is an ordinary register, which amdgpu_gfx_whole_wave
; does not keep.
define amdgpu_gfx_whole_wave void @whole_wave(i1 %active) {
  call void asm sideeffect "; clobber s34", "~{s34}"()
  ret void
}

; So a call of it clobbers s34, which its caller saves.
; CHECK-LABEL: calls_whole_wave:
; CHECK: v_writelane_b32 [[LANES:v[0-9]+]], s34,
; CHECK: s_swappc_b64
; CHECK: v_readlane_b32 s34, [[LANES]],
; CHECK: s_setpc_b64 s[30:31]
define void @calls_whole_wave() {
  call void(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @whole_wave)
  ret void
}

define internal void @has_fp() {
  %a = alloca [16 x i32], align 4, addrspace(5)
  call void @external(ptr addrspace(5) %a)
  ret void
}

; Its callee restores its frame pointer, so it need not save its own.
; CHECK-LABEL: tail_calls_has_fp:
; CHECK-NOT: s33
; CHECK-NOT: s_swappc
; CHECK: s_setpc_b64 s[{{[0-9]+:[0-9]+}}]
; CHECK-NEXT: .Lfunc_end
define internal void @tail_calls_has_fp() {
  tail call void @has_fp()
  ret void
}

define amdgpu_kernel void @kernel() {
  call void @tail_calls_has_fp()
  call void @calls_whole_wave()
  ret void
}
