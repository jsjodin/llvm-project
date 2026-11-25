% DESIGN: Loops / State variables
% A state variable of the clause that the body updates is threaded through
% the iterations: the first gets !.S before the loop, each later one the
% value the previous one gave, and !.S after the loop is the last one's. A
% body that only reads !.S shares it. io's state is threaded as any.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/threaded_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{124, 20}
% OUTPUT-NEXT:1
% OUTPUT-NEXT:2
% OUTPUT-NOT:{{.}}
:- module run_threaded.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    sums([3, 4, 5], 100, Total, 2, Scaled),
    io.write({Total, Scaled}, !IO), io.nl(!IO),
    ( foreach(X, [1, 2]) do io.write_int(X, !IO), io.nl(!IO) ).

:- pred sums(list(int)::in, int::in, int::out, int::in, int::out) is det.
sums(Xs, !Total, Factor, Scaled) :-
    some [!F] (
        !:F = Factor,
        ( foreach(X, Xs) do !:Total = !.Total + X * !.F ),
        Scaled = !.F * 10
    ).
