% MANUAL: Exception handling
% A try goal in a recursive predicate, whose goal throws on some calls,
% with state variables updated in the then part and the catch.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:sum 69
% OUTPUT-NOT:{{.}}

:- module run_recursion.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- pred loop(int::in, int::in, int::out) is cc_multi.
loop(N, !S) :-
    ( if N = 0 then
        true
    else
        ( try [] ( if N mod 2 = 0 then throw(N) else V = N )
        then
            !:S = !.S + V
        catch E ->
            !:S = !.S + 10 * E
        ),
        loop(N - 1, !S)
    ).

main(!IO) :-
    loop(5, 0, S),
    io.format("sum %d\n", [i(S)], !IO).
