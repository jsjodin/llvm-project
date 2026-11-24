% MANUAL: Exception handling
% If GOAL fails, ELSEGOAL is executed: a semidet goal, a cc_nondet one, and
% one that throws instead, which a catch handles.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:5 is positive
% OUTPUT-NEXT:-5 is not positive
% OUTPUT-NEXT:0: zero
% OUTPUT-NEXT:found 2
% OUTPUT-NEXT:none
% OUTPUT-NOT:{{.}}

:- module run_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- pred positive(int::in) is semidet.
positive(X) :-
    ( if X = 0 then throw("zero") else X > 0 ).

:- pred test(int::in, io::di, io::uo) is cc_multi.
test(N, !IO) :-
    ( try [] positive(N)
    then
        io.format("%d is positive\n", [i(N)], !IO)
    else
        io.format("%d is not positive\n", [i(N)], !IO)
    catch S ->
        io.format("%d: %s\n", [i(N), s(S)], !IO)
    ).

main(!IO) :-
    test(5, !IO),
    test(-5, !IO),
    test(0, !IO),
    ( try [] ( list.member(X, [1, 2, 3]), X > 1 )
    then
        io.format("found %d\n", [i(X)], !IO)
    else
        io.write_string("none\n", !IO)
    ),
    ( try [] ( list.member(Y, [1, 2, 3]), Y > 5 )
    then
        io.format("found %d\n", [i(Y)], !IO)
    else
        io.write_string("none\n", !IO)
    ).
