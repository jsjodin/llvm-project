; A whole-wave function on HSA, called from a compute function: it sets EXEC
; to every lane, keeps the inactive lanes of the VGPRs it uses, and puts them
; and EXEC back. At -O0 its SGPR spills to lanes of a VGPR need an EXEC copy,
; which used to stop frame lowering with an assertion.
; RUN: llc -O2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s
; RUN: llc -O2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s --check-prefix=W32
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s --check-prefix=W32

; CHECK-LABEL: wave_copy:
; CHECK: s_xor_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], -1
; CHECK: buffer_store_dword v0, off, s[0:3], s32
; CHECK: s_mov_b64 exec, -1
; CHECK: flat_load_dwordx2
; CHECK: flat_store_dwordx2
; CHECK: s_xor_b64 exec, s{{\[[0-9]+:[0-9]+\]}}, -1
; CHECK: buffer_load_dword v0, off, s[0:3], s32
; CHECK: s_mov_b64 exec, s{{\[[0-9]+:[0-9]+\]}}
; CHECK: s_setpc_b64
; W32-LABEL: wave_copy:
; W32: s_xor_saveexec_b32 s{{[0-9]+}}, -1
; W32: s_mov_b32 exec_lo, -1
; W32: s_setpc_b64

; CHECK-LABEL: copy:
; CHECK: s_swappc_b64
; W32-LABEL: copy:
; W32: s_swappc_b64

define amdgpu_gfx_whole_wave void @wave_copy(i1 %active, ptr inreg %to,
    ptr inreg %from, i64 inreg %words) {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %id = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %first = zext i32 %id to i64
  br label %loop

loop:
  %i = phi i64 [ %first, %entry ], [ %next, %body ]
  %more = icmp ult i64 %i, %words
  br i1 %more, label %body, label %done

body:
  %s = getelementptr i64, ptr %from, i64 %i
  %v = load i64, ptr %s, align 8
  %d = getelementptr i64, ptr %to, i64 %i
  store i64 %v, ptr %d, align 8
  %next = add i64 %i, 64
  br label %loop

done:
  ret void
}

define void @copy(ptr %to, ptr %from, i64 %words) {
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave.isVoid.p0(ptr @wave_copy,
      ptr inreg %to, ptr inreg %from, i64 inreg %words)
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
declare void @llvm.amdgcn.call.whole.wave.isVoid.p0(ptr, ...)
