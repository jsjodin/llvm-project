; RUN: llc -global-isel=0 -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck %s

; A function with "no_callee_saved_registers" keeps no register for its
; caller, as on X86, but its return address and the stack's pointers
; (s30-s34): it saves no other it uses, and a call of it preserves no other,
; so the caller keeps what it needs across the call itself.

declare void @external()
declare void @external_no_csr() "no_callee_saved_registers"

; CHECK-LABEL: ordinary:
; CHECK-DAG: {{buffer_store_dword|v_accvgpr_write_b32 a[0-9]+,}} v40
; CHECK-DAG: {{buffer_store_dword|v_accvgpr_read_b32 v[0-9]+,}} a32
; CHECK: s_setpc_b64 s[30:31]
define void @ordinary() {
  call void asm sideeffect "; clobber", "~{v40},~{a32}"()
  ret void
}

; CHECK-LABEL: no_csr:
; CHECK-NOT: buffer_store
; CHECK-NOT: v_accvgpr_read
; CHECK: ; clobber
; CHECK-NOT: buffer_load
; CHECK: s_setpc_b64 s[30:31]
define void @no_csr() "no_callee_saved_registers" {
  call void asm sideeffect "; clobber", "~{v40},~{a32}"()
  ret void
}

; It still keeps its return address across its calls.
; CHECK-LABEL: no_csr_calls:
; CHECK-DAG: v_writelane_b32 [[LANES:v[0-9]+]], s30,
; CHECK-DAG: v_writelane_b32 [[LANES]], s31,
; CHECK: s_swappc_b64 s[30:31]
; CHECK-DAG: v_readlane_b32 s30, [[LANES]],
; CHECK-DAG: v_readlane_b32 s31, [[LANES]],
; CHECK: s_setpc_b64 s[30:31]
define void @no_csr_calls() "no_callee_saved_registers" {
  call void @external()
  ret void
}

; A value live across the call is the caller's to keep: here in memory, since
; the call may clobber every register.
; CHECK-LABEL: keeps_across:
; CHECK: global_load_dword [[V:v[0-9]+]],
; CHECK: buffer_store_dword [[V]],
; CHECK: s_swappc_b64 s[30:31]
; CHECK: buffer_load_dword [[R:v[0-9]+]],
; CHECK: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[R]],
define void @keeps_across(ptr addrspace(1) %p) "no_callee_saved_registers" {
  %v = load i32, ptr addrspace(1) %p
  call void @external_no_csr()
  store i32 %v, ptr addrspace(1) %p
  ret void
}

; An ordinary function that calls one saves all its own callee-saved
; registers, the last of each kind here, and does not tail call it...
; CHECK-LABEL: no_tail_call:
; CHECK-DAG: {{buffer_store_dword|v_accvgpr_write_b32 a[0-9]+,}} v255
; CHECK-DAG: {{buffer_store_dword|v_accvgpr_read_b32 v[0-9]+,}} a255
; CHECK-DAG: v_writelane_b32 v{{[0-9]+}}, s105,
; CHECK: s_swappc_b64 s[30:31]
; CHECK: s_setpc_b64 s[30:31]
define void @no_tail_call() {
  tail call void @external_no_csr()
  ret void
}

; ...but one without callee-saved registers does, saving nothing.
; CHECK-LABEL: tail_call:
; CHECK-NOT: buffer_store
; CHECK-NOT: s_swappc
; CHECK: s_setpc_b64 s[{{[0-9]+:[0-9]+}}]
; CHECK-NEXT: .Lfunc_end
define void @tail_call() "no_callee_saved_registers" {
  tail call void @external_no_csr()
  ret void
}
