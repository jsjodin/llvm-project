; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

; Eliminating a tail call with an accumulator, or one whose return value is
; not the call's, rewrites every other return of the function. A return
; that follows a musttail call must stay next to it, so such calls are kept
; in a function that has one.

declare i64 @other(i64)

; Accumulator recursion: the musttail call's return must not combine the
; accumulator.
define i64 @accumulator(i64 %n) {
; CHECK-LABEL: define i64 @accumulator(
; CHECK:         [[REC:%.*]] = tail call i64 @accumulator(
; CHECK-NEXT:    [[MUL:%.*]] = mul i64 [[REC]], 2
; CHECK-NEXT:    ret i64 [[MUL]]
; CHECK:         [[OTHER:%.*]] = musttail call i64 @other(i64 %n)
; CHECK-NEXT:    ret i64 [[OTHER]]
entry:
  %base = icmp eq i64 %n, 0
  br i1 %base, label %done, label %recurse

recurse:
  %m = sub i64 %n, 1
  %rec = call i64 @accumulator(i64 %m)
  %mul = mul i64 %rec, 2
  ret i64 %mul

done:
  %other = musttail call i64 @other(i64 %n)
  ret i64 %other
}

; A known return value: the musttail call's return must not select it.
define i64 @known(i64 %n) {
; CHECK-LABEL: define i64 @known(
; CHECK:         [[REC:%.*]] = tail call i64 @known(
; CHECK:         [[OTHER:%.*]] = musttail call i64 @other(i64 %n)
; CHECK-NEXT:    ret i64 [[OTHER]]
entry:
  %base = icmp eq i64 %n, 0
  br i1 %base, label %done, label %recurse

recurse:
  %m = sub i64 %n, 1
  %rec = call i64 @known(i64 %m)
  ret i64 7

done:
  %other = musttail call i64 @other(i64 %n)
  ret i64 %other
}

; A call whose result is the return value needs no rewriting, and is still
; a loop.
define i64 @plain(i64 %n) {
; CHECK-LABEL: define i64 @plain(
; CHECK:       tailrecurse:
; CHECK-NOT:     call i64 @plain(
; CHECK:         [[OTHER:%.*]] = musttail call i64 @other(i64
; CHECK-NEXT:    ret i64 [[OTHER]]
entry:
  %base = icmp eq i64 %n, 0
  br i1 %base, label %done, label %recurse

recurse:
  %m = sub i64 %n, 1
  %rec = call i64 @plain(i64 %m)
  ret i64 %rec

done:
  %other = musttail call i64 @other(i64 %n)
  ret i64 %other
}
