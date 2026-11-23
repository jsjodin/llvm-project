; RUN: opt < %s -passes='cgscc(function-attrs)' -S | FileCheck %s

; Inline assembly might call a function, so a function with some is not
; norecurse, unless the function is nocallback, so that the assembly calls
; back into no function of the module, or the function or the call promises
; with the assumption "ompx_no_call_asm" that the assembly calls none, as the
; Attributor's call edges take it.

; CHECK: define void @asm_call() {
define void @asm_call() {
  call void asm sideeffect "", ""()
  ret void
}

; CHECK: define void @asm_function_promise() #[[PROMISED:[0-9]+]] {
define void @asm_function_promise() #0 {
  call void asm sideeffect "", ""()
  ret void
}

; CHECK: define void @asm_call_promise() #[[NORECURSE:[0-9]+]] {
define void @asm_call_promise() {
  call void asm sideeffect "", ""() #0
  ret void
}

; CHECK: define void @asm_nocallback() #[[NOCALLBACK:[0-9]+]] {
define void @asm_nocallback() nocallback {
  call void asm sideeffect "", ""()
  ret void
}

; A caller of one with the promise is norecurse as well.
; CHECK: define void @caller() #[[NORECURSE]] {
define void @caller() {
  call void @asm_function_promise()
  ret void
}

attributes #0 = { "llvm.assume"="ompx_no_call_asm" }

; CHECK-DAG: attributes #[[PROMISED]] = { norecurse {{.*}}"llvm.assume"="ompx_no_call_asm" }
; CHECK-DAG: attributes #[[NORECURSE]] = { norecurse }
; CHECK-DAG: attributes #[[NOCALLBACK]] = { nocallback norecurse }
