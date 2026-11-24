% MANUAL: Interfacing nondeterministic code with the real world
% A nondet goal without non-local outputs is semidet; solutions collects
% every solution; a cc_multi or cc_nondet predicate commits to its first
% solution; promise_equivalent_solutions makes a cc_multi goal det and a
% cc_nondet one semidet; and a cc_multi goal whose outputs are ignored is
% det.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:a big digit
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:1
% OUTPUT-NOT:{{.}}

:- module run_real_world.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred digit(int::out) is multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

:- pred big_digit(int::out) is nondet.
big_digit(X) :-
    digit(X),
    X > 1.

:- pred any_digit(int::out) is cc_multi.
any_digit(X) :-
    digit(X).

:- pred any_big(int::out) is cc_nondet.
any_big(X) :-
    big_digit(X).

:- pred has_big is semidet.
has_big :-
    big_digit(_).

:- pred smallest(int::out) is det.
smallest(Min) :-
    promise_equivalent_solutions [Min] (
        any_digit(D),
        solutions(digit, Ds),
        Min = list.foldl(int.min, Ds, D)
    ).

:- pred big_exists(int::out) is semidet.
big_exists(B) :-
    promise_equivalent_solutions [B] (
        any_big(_),
        B = 1
    ).

:- pred ignored is det.
ignored :-
    any_digit(_).

main(!IO) :-
    ( if has_big then
        io.write_string("a big digit\n", !IO)
    else
        io.write_string("no big digit\n", !IO)
    ),
    solutions(digit, Ds),
    io.write_line(Ds, !IO),
    smallest(Min),
    io.write_line(Min, !IO),
    ( if big_exists(B) then
        io.write_line(B, !IO)
    else
        io.write_string("none\n", !IO)
    ),
    ignored.
