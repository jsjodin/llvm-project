; Copies and fills by every lane of the wave, for code that runs on one lane
; of it: a GPU process's code, which runs on its wave's first lane while the
; others wait (device.cpp, RunProcess), and a target region's lane that copies
; a message. A whole-wave function (AMDGPUUsage, amdgpu_gfx_whole_wave) sets
; EXEC to every lane in its prologue, keeps the inactive lanes' values of the
; registers it uses on the stack, and puts them and EXEC back in its
; epilogue, so that the waiting lanes go on as they were. Clang has no
; attribute for the calling convention, so the functions are written here,
; and device.cpp calls the entries, which take ordinary arguments: the only
; active lane's, which the whole-wave call passes in scalar registers.
;
; Each lane, numbered by mbcnt (the high half adds 0 on a wave of 32),
; copies or stores the words `lane`, `lane` + 64, ...: four of them
; in flight before their stores, the wave's 256 consecutive words a round,
; then the rest one at a time.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu-amd-amdhsa"

define internal amdgpu_gfx_whole_wave void @wave_copy(i1 %active, ptr inreg %to,
    ptr inreg %from, i64 inreg %words, i64 inreg %lanes) #0 {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %id = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %first = zext i32 %id to i64
  %round = mul i64 %lanes, 4
  br label %four

four:
  %i = phi i64 [ %first, %entry ], [ %i.next, %four.body ]
  %i3 = add i64 %i, %round
  %last = sub i64 %i3, %lanes
  %fits = icmp ult i64 %last, %words
  br i1 %fits, label %four.body, label %one

four.body:
  %i1 = add i64 %i, %lanes
  %i2 = add i64 %i1, %lanes
  %s0 = getelementptr i64, ptr %from, i64 %i
  %s1 = getelementptr i64, ptr %from, i64 %i1
  %s2 = getelementptr i64, ptr %from, i64 %i2
  %s3 = getelementptr i64, ptr %from, i64 %last
  %v0 = load i64, ptr %s0, align 8
  %v1 = load i64, ptr %s1, align 8
  %v2 = load i64, ptr %s2, align 8
  %v3 = load i64, ptr %s3, align 8
  %d0 = getelementptr i64, ptr %to, i64 %i
  %d1 = getelementptr i64, ptr %to, i64 %i1
  %d2 = getelementptr i64, ptr %to, i64 %i2
  %d3 = getelementptr i64, ptr %to, i64 %last
  store i64 %v0, ptr %d0, align 8
  store i64 %v1, ptr %d1, align 8
  store i64 %v2, ptr %d2, align 8
  store i64 %v3, ptr %d3, align 8
  %i.next = add i64 %i, %round
  br label %four

one:
  %j = phi i64 [ %i, %four ], [ %j.next, %one.body ]
  %more = icmp ult i64 %j, %words
  br i1 %more, label %one.body, label %done

one.body:
  %s = getelementptr i64, ptr %from, i64 %j
  %v = load i64, ptr %s, align 8
  %d = getelementptr i64, ptr %to, i64 %j
  store i64 %v, ptr %d, align 8
  %j.next = add i64 %j, %lanes
  br label %one

done:
  ret void
}

define internal amdgpu_gfx_whole_wave void @wave_fill(i1 %active, ptr inreg %to,
    i64 inreg %word, i64 inreg %words, i64 inreg %lanes) #0 {
entry:
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %id = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %first = zext i32 %id to i64
  br label %loop

loop:
  %i = phi i64 [ %first, %entry ], [ %i.next, %body ]
  %more = icmp ult i64 %i, %words
  br i1 %more, label %body, label %done

body:
  %d = getelementptr i64, ptr %to, i64 %i
  store i64 %word, ptr %d, align 8
  %i.next = add i64 %i, %lanes
  br label %loop

done:
  ret void
}

; Copies `words` words from `from` to `to` by every lane of the caller's
; wave. The caller must be the wave's only active lane.
define hidden void @merk_device_wave_copy(ptr %to, ptr %from, i64 %words) #1 {
  %lanes32 = call i32 @llvm.amdgcn.wavefrontsize()
  %lanes = zext i32 %lanes32 to i64
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave.isVoid.p0(ptr @wave_copy,
      ptr inreg %to, ptr inreg %from, i64 inreg %words, i64 inreg %lanes)
  ret void
}

; Stores `word` in `words` words from `to` by every lane of the caller's
; wave. The caller must be the wave's only active lane.
define hidden void @merk_device_wave_fill(ptr %to, i64 %word, i64 %words) #1 {
  %lanes32 = call i32 @llvm.amdgcn.wavefrontsize()
  %lanes = zext i32 %lanes32 to i64
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave.isVoid.p0(ptr @wave_fill,
      ptr inreg %to, i64 inreg %word, i64 inreg %words, i64 inreg %lanes)
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)
declare i32 @llvm.amdgcn.wavefrontsize()
declare void @llvm.amdgcn.call.whole.wave.isVoid.p0(ptr, ...)

; No AGPRs: a whole-wave function keeps the inactive lanes of the VGPRs it
; uses, not of AGPRs, which at -O0 held a spilled VGPR on gfx90a.
;
; Norecurse, said here since the whole-wave call intrinsic is not
; nocallback, so LLVM cannot find it: a function that calls one not
; norecurse is not norecurse either, and a GPU process kernel above it gets
; a dynamic stack (AMDGPUResourceUsageAnalysis), whose scratch can keep some
; of its teams from starting. The entries are not inlined, so that no
; caller holds the intrinsic's call.
attributes #0 = { norecurse "amdgpu-agpr-alloc"="0" }
attributes #1 = { noinline norecurse }
