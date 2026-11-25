% MANUAL: Semantics
% Function application is strict and depth-first, left to right: every
% argument is evaluated, inner calls before outer ones, even an argument the
% function does not use and a call whose result is not used. Only the branch
% of a conditional expression that is taken is evaluated.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"left"
% OUTPUT-NEXT:"inner"
% OUTPUT-NEXT:"unused argument"
% OUTPUT-NEXT:"unused result"
% OUTPUT-NEXT:"used in one branch"
% OUTPUT-NEXT:"head"
% OUTPUT-NEXT:"else"
% OUTPUT-NEXT:"argument"
% OUTPUT-NOT:{{.}}

:- module run_function_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string, univ.

% Throws S, but is det: no pass may treat a call of it as erroneous.
:- func th(string, int) = int.
th(S, N) = ( if N > -1000 then throw(S) else N ).

:- pred show(exception_result(int)::in, io::di, io::uo) is det.
show(succeeded(N), !IO) :- io.format("succeeded %d\n", [i(N)], !IO).
show(failed, !IO) :- io.write_string("failed\n", !IO).
show(exception(E), !IO) :- io.write(univ_value(E), !IO), io.nl(!IO).

:- func f(int, int) = int.
f(X, Y) = X + Y.

:- func g(int) = int.
g(X) = X.

:- func first(int, int) = int.
first(X, _) = X.

:- pred left_to_right(int::out) is det.
left_to_right(R) :- R = f(th("left", 1), th("right", 1)).

:- pred inner_first(int::out) is det.
inner_first(R) :- R = f(g(th("inner", 1)), th("second", 1)).

:- pred unused_argument(int::out) is det.
unused_argument(R) :- R = first(1, th("unused argument", 1)).

:- pred unused_result(int::out) is det.
unused_result(R) :- _ = th("unused result", 1), R = 3.

% X is used in one branch only, but is evaluated before the if-then-else.
:- pred one_branch(int::out) is det.
one_branch(R) :-
    X = th("used in one branch", 1),
    ( if 1 = R0, R0 = 2 then R = X else R = 4 ).

:- pred list_elements(int::out) is det.
list_elements(R) :-
    L = [th("head", 1), th("tail", 1)],
    R = list.length(L).

:- pred branch_taken(int::out) is det.
branch_taken(R) :- R = ( if 1 = 2 then th("then", 1) else th("else", 1) ).

% A higher-order call evaluates its argument before the call.
:- pred closure_argument(int::out) is det.
closure_argument(R) :-
    F = (func(X) = th("closure", X)),
    R = F(th("argument", 1)).

main(!IO) :-
    try(left_to_right, R1), show(R1, !IO),
    try(inner_first, R2), show(R2, !IO),
    try(unused_argument, R3), show(R3, !IO),
    try(unused_result, R4), show(R4, !IO),
    try(one_branch, R5), show(R5, !IO),
    try(list_elements, R6), show(R6, !IO),
    try(branch_taken, R7), show(R7, !IO),
    try(closure_argument, R8), show(R8, !IO).
