% The call in a nondet disjunction's last branch passes its continuation
% on, as a tail call, so a search through five million list elements runs
% in constant stack at every optimization level, as in mmc. The output is
% mmc's.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module nondet_tail_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list.

:- pred member2(int::out, list(int)::in) is nondet.
member2(X, [H | T]) :-
    ( X = H ; member2(X, T) ).

:- pred build(int::in, list(int)::in, list(int)::out) is det.
build(N, L0, L) :-
    ( if N = 0 then L = L0 else build(N - 1, [N | L0], L) ).

:- pred first_over(int::in, list(int)::in, int::out) is cc_nondet.
first_over(Min, L, X) :- member2(X, L), X > Min.

main(!IO) :-
    build(5000000, [], L),
    ( if first_over(4999998, L, X) then io.write_int(X, !IO), io.nl(!IO) else io.write_string("none\n", !IO) ),
    ( if first_over(5000000, L, Y) then io.write_int(Y, !IO), io.nl(!IO) else io.write_string("none\n", !IO) ).

% CHECK:4999999
% CHECK-NEXT:none
