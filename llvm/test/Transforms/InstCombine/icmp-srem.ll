; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i1 @icmp_ugt_sremsmin_smin(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmin_smin(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[X]], -2147483648
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ugt i32 %r, -2147483648
  ret i1 %c
}

define i1 @icmp_ugt_sremsmin_sminp1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmin_sminp1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], -2147483647
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ugt i32 %r, -2147483647
  ret i1 %c
}

define i1 @icmp_ugt_sremsmin_smaxm1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmin_smaxm1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], 2147483646
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ugt i32 %r, 2147483646
  ret i1 %c
}

define i1 @icmp_ugt_sremsmin_smax(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmin_smax(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[X]], -2147483648
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ugt i32 %r, 2147483647
  ret i1 %c
}

define i1 @icmp_ult_sremsmin_smin(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmin_smin(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ult i32 %r, -2147483648
  ret i1 %c
}

define i1 @icmp_ult_sremsmin_sminp1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmin_sminp1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ult i32 %r, -2147483647
  ret i1 %c
}

define i1 @icmp_ult_sremsmin_sminp2(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmin_sminp2(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], -2147483646
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ult i32 %r, -2147483646
  ret i1 %c
}

define i1 @icmp_ult_sremsmin_smax(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmin_smax(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], -2147483648
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], 2147483647
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, -2147483648
  %c = icmp ult i32 %r, 2147483647
  ret i1 %c
}

define i1 @icmp_ugt_srem5_smin(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_smin(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, -2147483648
  ret i1 %c
}

define i1 @icmp_ugt_srem5_m5(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_m5(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, -5
  ret i1 %c
}

define i1 @icmp_ugt_srem5_m4(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_m4(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], -4
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, -4
  ret i1 %c
}

define i1 @icmp_ugt_srem5_3(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_3(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], 3
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, 3
  ret i1 %c
}

define i1 @icmp_ugt_srem5_4(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_4(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, 4
  ret i1 %c
}

define i1 @icmp_ugt_srem5_smaxm1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_srem5_smaxm1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ugt i32 %r, 2147483646
  ret i1 %c
}

define i1 @icmp_ult_srem5_sminp1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_sminp1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, -2147483647
  ret i1 %c
}

define i1 @icmp_ult_srem5_m4(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_m4(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, -4
  ret i1 %c
}

define i1 @icmp_ult_srem5_m3(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_m3(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], -3
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, -3
  ret i1 %c
}

define i1 @icmp_ult_srem5_4(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_4(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], 4
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, 4
  ret i1 %c
}

define i1 @icmp_ult_srem5_5(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_5(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, 5
  ret i1 %c
}

define i1 @icmp_ult_srem5_smax(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_srem5_smax(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 5
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 5
  %c = icmp ult i32 %r, 2147483647
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_smin(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_smin(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, -2147483648
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_sminp1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_sminp1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, -2147483647
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_sminp2(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_sminp2(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], -2147483646
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, -2147483646
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_smaxm2(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_smaxm2(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp ugt i32 [[R]], 2147483645
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, 2147483645
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_smaxm1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_smaxm1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, 2147483646
  ret i1 %c
}

define i1 @icmp_ugt_sremsmax_smax(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ugt_sremsmax_smax(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[R]], 0
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ugt i32 %r, 2147483647
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_smin(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_smin(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, -2147483648
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_sminp1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_sminp1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, -2147483647
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_sminp2(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_sminp2(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, -2147483646
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_sminp3(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_sminp3(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], -2147483645
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, -2147483645
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_smaxm1(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_smaxm1(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[R]], 2147483646
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, 2147483646
  ret i1 %c
}

define i1 @icmp_ult_sremsmax_smax(i32 %x) {
; CHECK-LABEL: define i1 @icmp_ult_sremsmax_smax(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = srem i32 [[X]], 2147483647
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 [[R]], -1
; CHECK-NEXT:    ret i1 [[C]]
;
  %r = srem i32 %x, 2147483647
  %c = icmp ult i32 %r, 2147483647
  ret i1 %c
}
