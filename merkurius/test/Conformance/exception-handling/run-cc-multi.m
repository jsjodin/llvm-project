% MANUAL: Exception handling
% GOAL may be cc_multi: only one of its solutions is taken.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:one solution
% OUTPUT-NOT:{{.}}

:- module run_cc_multi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- pred one_of(int::out) is multi.
one_of(1).
one_of(2).

main(!IO) :-
    ( try [] one_of(X)
    then
        ( if X = 1 ; X = 2 then
            io.write_string("one solution\n", !IO)
        else
            io.write_string("wrong\n", !IO)
        )
    catch_any _ ->
        io.write_string("caught\n", !IO)
    ).
