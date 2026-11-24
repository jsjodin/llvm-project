% MANUAL: Semantics
% Disjunctions run depth-first, left to right: all the solutions of a multi
% predicate in that order, with a disjunct's own disjunction in its place;
% the first solution of a committed choice; and the left disjunct's throw
% before the right one's solution.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[3, 1, 9, 7, 2]
% OUTPUT-NEXT:first above 5: 9
% OUTPUT-NEXT:"left"
% OUTPUT-NOT:{{.}}

:- module run_disjunction_order.
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
:- import_module solutions.

:- pred p(int::out) is multi.
p(X) :- ( X = 3 ; X = 1 ; q(X) ; X = 2 ).

:- pred q(int::out) is multi.
q(X) :- ( X = 9 ; X = 7 ).

:- pred first(int::out) is cc_nondet.
first(X) :- p(X), X > 5.

:- pred add(int::in, list(int)::in, list(int)::out) is det.
add(X, L, [X | L]).

:- pred left_throws(int::out) is cc_nondet.
left_throws(R) :- ( R = th("left", 1) ; R = 2 ), R > 0.

main(!IO) :-
    unsorted_aggregate(p, add, [], L),
    list.reverse(L, InOrder),
    io.write(InOrder, !IO),
    io.nl(!IO),
    ( if first(F) then
        io.format("first above 5: %d\n", [i(F)], !IO)
    else
        io.write_string("none above 5\n", !IO)
    ),
    try(left_throws, R), show(R, !IO).
