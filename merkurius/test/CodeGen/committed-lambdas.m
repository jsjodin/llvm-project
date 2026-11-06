% The body of a committed choice lambda expression, whose first solution
% is the lambda expression's, is in merk.commit when it has more than one
% solution, as a committed choice procedure's is. The output is mmc's for
% the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module committed_lambdas.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int.

:- pred small(int::out) is multi.
small(3).
small(4).

:- pred run(pred(int)::in(pred(out) is cc_multi), int::out) is cc_multi.
run(P, X) :- P(X).

:- pred try(pred(int)::in(pred(out) is cc_nondet), int::out) is cc_nondet.
try(P, X) :- P(X).

:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).

main(!IO) :-
    % A call of a multi procedure.
    P = (pred(X::out) is cc_multi :- small(X)),
    run(P, A), show(A, !IO),
    % A conjunction with such a call.
    Q = (pred(X::out) is cc_multi :- small(Y), X = Y * 10),
    run(Q, B), show(B, !IO),
    % A disjunction whose first branch has more than one solution.
    R = (pred(X::out) is cc_multi :- ( small(X) ; X = 0 )),
    run(R, C), show(C, !IO),
    % One that can fail.
    S = (pred(X::out) is cc_nondet :- small(X), X > 3),
    ( if try(S, D) then show(D, !IO) else io.write_string("none\n", !IO) ),
    T = (pred(X::out) is cc_nondet :- small(X), X > 4),
    ( if try(T, E) then show(E, !IO) else io.write_string("none\n", !IO) ).

% MERK-LABEL: merk.proc private @committed_lambdas.main_2_0$lambda0() -> !merk.word attributes {determinism = #merk.determinism<cc_multi>} {
% MERK-NEXT: %[[X:.*]] = merk.commit -> i64 {
% MERK-NEXT: merk.nondet_call @committed_lambdas.small_1_0() {
% MERK-LABEL: merk.proc private @committed_lambdas.main_2_0$lambda2() -> !merk.word attributes {determinism = #merk.determinism<cc_multi>} {
% MERK-NEXT: %[[Y:.*]] = merk.commit -> i64 {
% MERK-NEXT: merk.each {
% MERK-NEXT: merk.disj {
% MERK-LABEL: merk.proc private @committed_lambdas.main_2_0$lambda3() -> !merk.word attributes {determinism = #merk.determinism<cc_nondet>} {
% MERK-NEXT: %[[Z:.*]] = merk.commit -> i64 {

% OUTPUT:3
% OUTPUT-NEXT:30
% OUTPUT-NEXT:3
% OUTPUT-NEXT:4
% OUTPUT-NEXT:none
