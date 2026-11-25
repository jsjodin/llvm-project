% MANUAL: State variables
% State variables: !X in a head, !.X and !:X, threading through
% if-then-else, disjunction and negation, a lambda's own !.S and !:S, some
% [!X], and a field update.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{15, 20, r(1)}
% OUTPUT-NOT:{{.}}

:- module run_state_variables.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type r ---> r(count :: int).
:- pred add(int::in, int::in, int::out) is det.
add(N, X0, X) :- X = X0 + N.
:- pred step(int::in, int::out) is det.
step(!X) :-
    add(1, !X),
    ( if !.X > 3 then add(10, !X) else true ),
    ( if not (add(100, !.X, Y), Y = 5) then true else true ),
    some [!Y] ( !:Y = !.X, add(1, !Y), !:X = !.Y ).
:- pred bump(r::in, r::out) is det.
bump(!R) :- !R ^ count := !.R ^ count + 1.

main(!IO) :-
    step(3, A),
    F = (pred(N::in, !.S::in, !:S::out) is det :- add(N, !S)),
    F(5, A, B),
    bump(r(0), R),
    io.write({A, B, R}, !IO), io.nl(!IO).
