% Switches on ints and characters, semidet and failure procedures, and
% negation. A procedure that can fail returns its success flag before its
% outputs, as { i1, i64 } for s/2 and i1 for t/1; one that fails returns
% false and undefined outputs. A switch's case can have more constants than
% one, as e/2's second. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc private @switch.s_2_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: %[[Y:.*]] = merk.switch %arg0 : i64 -> i64 cases {{\[}}[-2], [0], [3]] can_fail {
% MERK-NEXT: %[[C:.*]] = merk.constant 7 : i64
% MERK-NEXT: merk.yield %[[C]] : i64
% MERK-NEXT: }, {
% MERK: merk.succeed %[[Y]] : i64
% MERK-LABEL: merk.proc private @switch.c_2_0(%arg0: i32) -> i64
% MERK-NEXT: merk.switch %arg0 : i32 -> i64 cases {{\[}}[97 : i32], [98 : i32], [233 : i32]] can_fail {
% MERK-LABEL: merk.proc private @switch.e_2_0(%arg0: i64) -> i64
% MERK-NEXT: merk.switch %arg0 : i64 -> i64 cases {{\[}}[0], [1, 2]] can_fail {
% MERK-LABEL: merk.proc private @switch.t_1_0(%arg0: i64) attributes {determinism = #merk.determinism<semidet>} {
% MERK: merk.guard
% MERK-NEXT: merk.not {
% MERK-NEXT: %[[FIVE:.*]] = merk.constant 5 : i64
% MERK-NEXT: %[[EQ:.*]] = arith.cmpi eq, %arg0, %[[FIVE]] : i64
% MERK-NEXT: merk.guard %[[EQ]]
% MERK-NEXT: merk.yield
% MERK-NEXT: }
% MERK-NEXT: merk.not {
% MERK: merk.succeed
% MERK-LABEL: merk.proc private @switch.ff_1_0(%arg0: i64) attributes {determinism = #merk.determinism<failure>} {
% MERK-NEXT: merk.fail

% LLVM-LABEL: define internal tailcc { i1, i64 } @switch.s_2_0(i64 %0) {
% LLVM-NEXT: switch i64 %0, label %[[NO:.*]] [
% LLVM-NEXT: i64 -2, label %{{.*}}
% LLVM-NEXT: i64 0, label %{{.*}}
% LLVM-NEXT: i64 3, label %{{.*}}
% LLVM-NEXT: ]
% LLVM: ret { i1, i64 } { i1 true, i64 7 }
% LLVM: [[NO]]:
% LLVM-NEXT: ret { i1, i64 } { i1 false, i64 undef }
% LLVM-LABEL: define internal tailcc i64 @switch.d_f1_0(i64 %0) {
% LLVM: %[[R:.*]] = call tailcc { i1, i64 } @switch.e_2_0(i64 %0)
% LLVM-NEXT: %[[V:.*]] = extractvalue { i1, i64 } %[[R]], 1
% LLVM-NEXT: %[[OK:.*]] = extractvalue { i1, i64 } %[[R]], 0
% LLVM-NEXT: br i1 %[[OK]], label %{{.*}}, label %[[ELSE:.*]]
% LLVM: [[ELSE]]:
% LLVM-NEXT: ret i64 0
% LLVM-LABEL: define internal tailcc i1 @switch.t_1_0(i64 %0) {
% LLVM: ret i1 true
% LLVM: ret i1 false
% LLVM-LABEL: define internal tailcc i1 @switch.ff_1_0(i64 %0) {
% LLVM: ret i1 false

% OUTPUT:7 - 0
% OUTPUT-NEXT:10 - 100
% OUTPUT-NEXT:no - 200
% OUTPUT-NEXT:no - 200
% OUTPUT-NEXT:4 t 0
% OUTPUT-NEXT:no - 0
% OUTPUT-NEXT:no - 0
% OUTPUT-NEXT:no t 0
% OUTPUT-NEXT:1
% OUTPUT-NEXT:2
% OUTPUT-NEXT:3
% OUTPUT-NEXT:no
% OUTPUT-NOT:{{.}}

:- module switch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, char.
:- pred s(int::in, int::out) is semidet.
s(0, 10).
s(3, 4).
s(-2, 7).
:- pred c(char::in, int::out) is semidet.
c('a', 1).
c('b', 2).
c('é', 3).
:- pred e(int::in, int::out) is semidet.
e(N, R) :-
    ( N = 0, R = 100
    ; ( N = 1 ; N = 2 ), R = 200
    ).
:- func d(int) = int.
d(N) = ( if e(N, R) then R else 0 ).
:- pred t(int::in) is semidet.
t(X) :- X > 2, not X = 5, X \= 7.
:- pred ff(int::in) is failure.
ff(_) :- fail.
:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :-
    ( if s(X, Y) then io.write_int(Y, !IO) else io.write_string("no", !IO) ),
    io.write_char(' ', !IO),
    ( if t(X) then io.write_string("t", !IO) else io.write_string("-", !IO) ),
    ( if ff(X) then io.write_string("ff", !IO) else true ),
    io.write_char(' ', !IO),
    io.write_int(d(X), !IO),
    io.nl(!IO).
:- pred showc(char::in, io::di, io::uo) is det.
showc(C, !IO) :-
    ( if c(C, N) then io.write_int(N, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO).
main(!IO) :-
    show(-2, !IO), show(0, !IO), show(1, !IO), show(2, !IO), show(3, !IO),
    show(5, !IO), show(7, !IO), show(8, !IO),
    showc('a', !IO), showc('b', !IO), showc('é', !IO), showc('z', !IO).
