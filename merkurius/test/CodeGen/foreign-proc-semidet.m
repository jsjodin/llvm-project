% A semidet foreign_proc sets SUCCESS_INDICATOR. Its C function returns
% it, as C's int, and stores every output through a pointer; the
% procedure's merk.foreign_call has can_fail, and its first result is the
% flag, which the procedure tests.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s -o %t.mlir
% RUN: FileCheck %s --check-prefix=MERK < %t.mlir
% RUN: FileCheck %s --check-prefix=C < %t.mlir
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% C: merk.foreign_code = "{{.*}}typedef int MR_bool;\0A#define MR_TRUE 1\0A#define MR_FALSE 0\0A
% C-SAME: \0AMR_bool merk_foreign_foreign__proc__semidet_Dhalf__2__0(MR_Integer X, MR_Integer *merk_out_Y) {\0A  MR_bool SUCCESS_INDICATOR = MR_TRUE;\0A  MR_Integer Y;\0A
% C-SAME: *merk_out_Y = Y;\0A  return SUCCESS_INDICATOR;\0A}\0A
% C-SAME: \0AMR_bool merk_foreign_foreign__proc__semidet_Dpositive__1__0(MR_Integer X) {\0A  MR_bool SUCCESS_INDICATOR = MR_TRUE;\0A
% C-SAME: return SUCCESS_INDICATOR;\0A}\0A

% MERK-LABEL: merk.proc private @foreign_proc_semidet.half_2_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: %[[R:.*]]:2 = merk.foreign_call @merk_foreign_foreign__proc__semidet_Dhalf__2__0(%arg0) can_fail : (i64) -> (i1, i64)
% MERK-NEXT: merk.guard %[[R]]#0
% MERK-NEXT: merk.succeed %[[R]]#1 : i64
% MERK-LABEL: merk.proc private @foreign_proc_semidet.positive_1_0(%arg0: i64) attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: %[[OK:.*]] = merk.foreign_call @merk_foreign_foreign__proc__semidet_Dpositive__1__0(%arg0) can_fail : (i64) -> i1
% MERK-NEXT: merk.guard %[[OK]]
% MERK-NEXT: merk.succeed

% LLVM-LABEL: define internal tailcc { i1, i64 } @foreign_proc_semidet.half_2_0(i64 %0) {
% LLVM-NEXT: %[[SLOT:.*]] = alloca i64
% LLVM-NEXT: %[[FLAG:.*]] = call i32 @merk_foreign_foreign__proc__semidet_Dhalf__2__0(i64 %0, ptr %[[SLOT]])
% LLVM-NEXT: %[[OK:.*]] = icmp ne i32 %[[FLAG]], 0
% LLVM-NEXT: %[[Y:.*]] = load i64, ptr %[[SLOT]]
% LLVM-NEXT: br i1 %[[OK]], label %[[YES:.*]], label %[[NO:.*]]
% LLVM: [[YES]]:
% LLVM-NEXT: %[[S:.*]] = insertvalue { i1, i64 } { i1 true, i64 undef }, i64 %[[Y]], 1
% LLVM-NEXT: ret { i1, i64 } %[[S]]
% LLVM: [[NO]]:
% LLVM-NEXT: ret { i1, i64 } { i1 false, i64 undef }
% LLVM-LABEL: define internal tailcc i1 @foreign_proc_semidet.positive_1_0(i64 %0) {
% LLVM-NEXT: %[[FLAG:.*]] = call i32 @merk_foreign_foreign__proc__semidet_Dpositive__1__0(i64 %0)

% OUTPUT:3 positive
% OUTPUT-NEXT:odd positive
% OUTPUT-NEXT:-2
% OUTPUT-NOT:{{.}}

:- module foreign_proc_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    test(6, !IO),
    test(7, !IO),
    test(-4, !IO).

:- pred test(int::in, io::di, io::uo) is det.
test(N, !IO) :-
    ( if half(N, H) then io.write_int(H, !IO) else io.write_string("odd", !IO) ),
    ( if positive(N) then io.write_string(" positive", !IO) else true ),
    io.nl(!IO).

:- pred half(int::in, int::out) is semidet.
:- pragma foreign_proc("C",
    half(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"
    SUCCESS_INDICATOR = X % 2 == 0;
    Y = X / 2;
").

:- pred positive(int::in) is semidet.
:- pragma foreign_proc("C",
    positive(X::in),
    [will_not_call_mercury, promise_pure],
"SUCCESS_INDICATOR = X > 0 ? MR_TRUE : MR_FALSE;").
