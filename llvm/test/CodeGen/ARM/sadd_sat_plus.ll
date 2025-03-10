; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=thumbv6m-none-eabi | FileCheck %s --check-prefix=CHECK-T1
; RUN: llc < %s -mtriple=thumbv7m-none-eabi | FileCheck %s --check-prefix=CHECK-T2 --check-prefix=CHECK-T2NODSP
; RUN: llc < %s -mtriple=thumbv7em-none-eabi | FileCheck %s --check-prefix=CHECK-T2 --check-prefix=CHECK-T2DSP
; RUN: llc < %s -mtriple=armv8a-none-eabi | FileCheck %s --check-prefix=CHECK-ARM

declare i4 @llvm.sadd.sat.i4(i4, i4)
declare i8 @llvm.sadd.sat.i8(i8, i8)
declare i16 @llvm.sadd.sat.i16(i16, i16)
declare i32 @llvm.sadd.sat.i32(i32, i32)
declare i64 @llvm.sadd.sat.i64(i64, i64)

define i32 @func32(i32 %x, i32 %y, i32 %z) nounwind {
; CHECK-T1-LABEL: func32:
; CHECK-T1:       @ %bb.0:
; CHECK-T1-NEXT:    muls r1, r2, r1
; CHECK-T1-NEXT:    adds r0, r0, r1
; CHECK-T1-NEXT:    bvc .LBB0_2
; CHECK-T1-NEXT:  @ %bb.1:
; CHECK-T1-NEXT:    asrs r1, r0, #31
; CHECK-T1-NEXT:    movs r0, #1
; CHECK-T1-NEXT:    lsls r0, r0, #31
; CHECK-T1-NEXT:    eors r0, r1
; CHECK-T1-NEXT:  .LBB0_2:
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2NODSP-LABEL: func32:
; CHECK-T2NODSP:       @ %bb.0:
; CHECK-T2NODSP-NEXT:    mla r1, r1, r2, r0
; CHECK-T2NODSP-NEXT:    mov.w r2, #-2147483648
; CHECK-T2NODSP-NEXT:    cmp r1, r0
; CHECK-T2NODSP-NEXT:    it vs
; CHECK-T2NODSP-NEXT:    eorvs.w r1, r2, r1, asr #31
; CHECK-T2NODSP-NEXT:    mov r0, r1
; CHECK-T2NODSP-NEXT:    bx lr
;
; CHECK-T2DSP-LABEL: func32:
; CHECK-T2DSP:       @ %bb.0:
; CHECK-T2DSP-NEXT:    muls r1, r2, r1
; CHECK-T2DSP-NEXT:    qadd r0, r0, r1
; CHECK-T2DSP-NEXT:    bx lr
;
; CHECK-ARM-LABEL: func32:
; CHECK-ARM:       @ %bb.0:
; CHECK-ARM-NEXT:    mul r1, r1, r2
; CHECK-ARM-NEXT:    qadd r0, r0, r1
; CHECK-ARM-NEXT:    bx lr
  %a = mul i32 %y, %z
  %tmp = call i32 @llvm.sadd.sat.i32(i32 %x, i32 %a)
  ret i32 %tmp
}

define i64 @func64(i64 %x, i64 %y, i64 %z) nounwind {
; CHECK-T1-LABEL: func64:
; CHECK-T1:       @ %bb.0:
; CHECK-T1-NEXT:    .save {r4, lr}
; CHECK-T1-NEXT:    push {r4, lr}
; CHECK-T1-NEXT:    ldr r3, [sp, #12]
; CHECK-T1-NEXT:    mov r2, r1
; CHECK-T1-NEXT:    eors r2, r3
; CHECK-T1-NEXT:    ldr r4, [sp, #8]
; CHECK-T1-NEXT:    adds r4, r0, r4
; CHECK-T1-NEXT:    adcs r3, r1
; CHECK-T1-NEXT:    eors r1, r3
; CHECK-T1-NEXT:    bics r1, r2
; CHECK-T1-NEXT:    asrs r0, r3, #31
; CHECK-T1-NEXT:    movs r2, #1
; CHECK-T1-NEXT:    lsls r2, r2, #31
; CHECK-T1-NEXT:    eors r2, r0
; CHECK-T1-NEXT:    cmp r1, #0
; CHECK-T1-NEXT:    bpl .LBB1_3
; CHECK-T1-NEXT:  @ %bb.1:
; CHECK-T1-NEXT:    bpl .LBB1_4
; CHECK-T1-NEXT:  .LBB1_2:
; CHECK-T1-NEXT:    mov r1, r2
; CHECK-T1-NEXT:    pop {r4, pc}
; CHECK-T1-NEXT:  .LBB1_3:
; CHECK-T1-NEXT:    mov r0, r4
; CHECK-T1-NEXT:    bmi .LBB1_2
; CHECK-T1-NEXT:  .LBB1_4:
; CHECK-T1-NEXT:    mov r2, r3
; CHECK-T1-NEXT:    mov r1, r2
; CHECK-T1-NEXT:    pop {r4, pc}
;
; CHECK-T2-LABEL: func64:
; CHECK-T2:       @ %bb.0:
; CHECK-T2-NEXT:    ldr r2, [sp]
; CHECK-T2-NEXT:    ldr.w r12, [sp, #4]
; CHECK-T2-NEXT:    adds r0, r0, r2
; CHECK-T2-NEXT:    adc.w r2, r1, r12
; CHECK-T2-NEXT:    eor.w r3, r1, r12
; CHECK-T2-NEXT:    eors r1, r2
; CHECK-T2-NEXT:    bics r1, r3
; CHECK-T2-NEXT:    it mi
; CHECK-T2-NEXT:    asrmi r0, r2, #31
; CHECK-T2-NEXT:    mov.w r1, #-2147483648
; CHECK-T2-NEXT:    it mi
; CHECK-T2-NEXT:    eormi.w r2, r1, r2, asr #31
; CHECK-T2-NEXT:    mov r1, r2
; CHECK-T2-NEXT:    bx lr
;
; CHECK-ARM-LABEL: func64:
; CHECK-ARM:       @ %bb.0:
; CHECK-ARM-NEXT:    ldr r12, [sp]
; CHECK-ARM-NEXT:    ldr r2, [sp, #4]
; CHECK-ARM-NEXT:    adds r0, r0, r12
; CHECK-ARM-NEXT:    eor r3, r1, r2
; CHECK-ARM-NEXT:    adc r2, r1, r2
; CHECK-ARM-NEXT:    eor r1, r1, r2
; CHECK-ARM-NEXT:    bics r1, r1, r3
; CHECK-ARM-NEXT:    asrmi r0, r2, #31
; CHECK-ARM-NEXT:    mov r1, #-2147483648
; CHECK-ARM-NEXT:    eormi r2, r1, r2, asr #31
; CHECK-ARM-NEXT:    mov r1, r2
; CHECK-ARM-NEXT:    bx lr
  %a = mul i64 %y, %z
  %tmp = call i64 @llvm.sadd.sat.i64(i64 %x, i64 %z)
  ret i64 %tmp
}

define signext i16 @func16(i16 signext %x, i16 signext %y, i16 signext %z) nounwind {
; CHECK-T1-LABEL: func16:
; CHECK-T1:       @ %bb.0:
; CHECK-T1-NEXT:    muls r1, r2, r1
; CHECK-T1-NEXT:    sxth r1, r1
; CHECK-T1-NEXT:    adds r0, r0, r1
; CHECK-T1-NEXT:    ldr r1, .LCPI2_0
; CHECK-T1-NEXT:    cmp r0, r1
; CHECK-T1-NEXT:    blt .LBB2_2
; CHECK-T1-NEXT:  @ %bb.1:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB2_2:
; CHECK-T1-NEXT:    ldr r1, .LCPI2_1
; CHECK-T1-NEXT:    cmp r0, r1
; CHECK-T1-NEXT:    bgt .LBB2_4
; CHECK-T1-NEXT:  @ %bb.3:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB2_4:
; CHECK-T1-NEXT:    bx lr
; CHECK-T1-NEXT:    .p2align 2
; CHECK-T1-NEXT:  @ %bb.5:
; CHECK-T1-NEXT:  .LCPI2_0:
; CHECK-T1-NEXT:    .long 32767 @ 0x7fff
; CHECK-T1-NEXT:  .LCPI2_1:
; CHECK-T1-NEXT:    .long 4294934528 @ 0xffff8000
;
; CHECK-T2NODSP-LABEL: func16:
; CHECK-T2NODSP:       @ %bb.0:
; CHECK-T2NODSP-NEXT:    muls r1, r2, r1
; CHECK-T2NODSP-NEXT:    sxth r1, r1
; CHECK-T2NODSP-NEXT:    add r0, r1
; CHECK-T2NODSP-NEXT:    ssat r0, #16, r0
; CHECK-T2NODSP-NEXT:    bx lr
;
; CHECK-T2DSP-LABEL: func16:
; CHECK-T2DSP:       @ %bb.0:
; CHECK-T2DSP-NEXT:    muls r1, r2, r1
; CHECK-T2DSP-NEXT:    qadd16 r0, r0, r1
; CHECK-T2DSP-NEXT:    sxth r0, r0
; CHECK-T2DSP-NEXT:    bx lr
;
; CHECK-ARM-LABEL: func16:
; CHECK-ARM:       @ %bb.0:
; CHECK-ARM-NEXT:    smulbb r1, r1, r2
; CHECK-ARM-NEXT:    qadd16 r0, r0, r1
; CHECK-ARM-NEXT:    sxth r0, r0
; CHECK-ARM-NEXT:    bx lr
  %a = mul i16 %y, %z
  %tmp = call i16 @llvm.sadd.sat.i16(i16 %x, i16 %a)
  ret i16 %tmp
}

define signext i8 @func8(i8 signext %x, i8 signext %y, i8 signext %z) nounwind {
; CHECK-T1-LABEL: func8:
; CHECK-T1:       @ %bb.0:
; CHECK-T1-NEXT:    muls r1, r2, r1
; CHECK-T1-NEXT:    sxtb r1, r1
; CHECK-T1-NEXT:    adds r0, r0, r1
; CHECK-T1-NEXT:    movs r1, #127
; CHECK-T1-NEXT:    cmp r0, #127
; CHECK-T1-NEXT:    blt .LBB3_2
; CHECK-T1-NEXT:  @ %bb.1:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB3_2:
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    cmp r0, r1
; CHECK-T1-NEXT:    bgt .LBB3_4
; CHECK-T1-NEXT:  @ %bb.3:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB3_4:
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2NODSP-LABEL: func8:
; CHECK-T2NODSP:       @ %bb.0:
; CHECK-T2NODSP-NEXT:    muls r1, r2, r1
; CHECK-T2NODSP-NEXT:    sxtb r1, r1
; CHECK-T2NODSP-NEXT:    add r0, r1
; CHECK-T2NODSP-NEXT:    ssat r0, #8, r0
; CHECK-T2NODSP-NEXT:    bx lr
;
; CHECK-T2DSP-LABEL: func8:
; CHECK-T2DSP:       @ %bb.0:
; CHECK-T2DSP-NEXT:    muls r1, r2, r1
; CHECK-T2DSP-NEXT:    qadd8 r0, r0, r1
; CHECK-T2DSP-NEXT:    sxtb r0, r0
; CHECK-T2DSP-NEXT:    bx lr
;
; CHECK-ARM-LABEL: func8:
; CHECK-ARM:       @ %bb.0:
; CHECK-ARM-NEXT:    smulbb r1, r1, r2
; CHECK-ARM-NEXT:    qadd8 r0, r0, r1
; CHECK-ARM-NEXT:    sxtb r0, r0
; CHECK-ARM-NEXT:    bx lr
  %a = mul i8 %y, %z
  %tmp = call i8 @llvm.sadd.sat.i8(i8 %x, i8 %a)
  ret i8 %tmp
}

define signext i4 @func4(i4 signext %x, i4 signext %y, i4 signext %z) nounwind {
; CHECK-T1-LABEL: func4:
; CHECK-T1:       @ %bb.0:
; CHECK-T1-NEXT:    muls r1, r2, r1
; CHECK-T1-NEXT:    lsls r1, r1, #28
; CHECK-T1-NEXT:    asrs r1, r1, #28
; CHECK-T1-NEXT:    adds r0, r0, r1
; CHECK-T1-NEXT:    movs r1, #7
; CHECK-T1-NEXT:    cmp r0, #7
; CHECK-T1-NEXT:    blt .LBB4_2
; CHECK-T1-NEXT:  @ %bb.1:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB4_2:
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    cmp r0, r1
; CHECK-T1-NEXT:    bgt .LBB4_4
; CHECK-T1-NEXT:  @ %bb.3:
; CHECK-T1-NEXT:    mov r0, r1
; CHECK-T1-NEXT:  .LBB4_4:
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2NODSP-LABEL: func4:
; CHECK-T2NODSP:       @ %bb.0:
; CHECK-T2NODSP-NEXT:    muls r1, r2, r1
; CHECK-T2NODSP-NEXT:    lsls r1, r1, #28
; CHECK-T2NODSP-NEXT:    add.w r0, r0, r1, asr #28
; CHECK-T2NODSP-NEXT:    ssat r0, #4, r0
; CHECK-T2NODSP-NEXT:    bx lr
;
; CHECK-T2DSP-LABEL: func4:
; CHECK-T2DSP:       @ %bb.0:
; CHECK-T2DSP-NEXT:    muls r1, r2, r1
; CHECK-T2DSP-NEXT:    lsls r0, r0, #28
; CHECK-T2DSP-NEXT:    lsls r1, r1, #28
; CHECK-T2DSP-NEXT:    qadd r0, r0, r1
; CHECK-T2DSP-NEXT:    asrs r0, r0, #28
; CHECK-T2DSP-NEXT:    bx lr
;
; CHECK-ARM-LABEL: func4:
; CHECK-ARM:       @ %bb.0:
; CHECK-ARM-NEXT:    smulbb r1, r1, r2
; CHECK-ARM-NEXT:    lsl r0, r0, #28
; CHECK-ARM-NEXT:    lsl r1, r1, #28
; CHECK-ARM-NEXT:    qadd r0, r0, r1
; CHECK-ARM-NEXT:    asr r0, r0, #28
; CHECK-ARM-NEXT:    bx lr
  %a = mul i4 %y, %z
  %tmp = call i4 @llvm.sadd.sat.i4(i4 %x, i4 %a)
  ret i4 %tmp
}
