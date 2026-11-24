% MANUAL: Exception handling
% On entering a try goal, GOAL is executed; if it succeeds without throwing
% an exception, THENGOAL is executed, and sees the variables GOAL binds.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:then 6 42
% OUTPUT-NOT:{{.}}

:- module run_then.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] ( X = 6, Y = X * 7 )
    then
        io.format("then %d %d\n", [i(X), i(Y)], !IO)
    catch_any _ ->
        io.write_string("caught\n", !IO)
    ).
