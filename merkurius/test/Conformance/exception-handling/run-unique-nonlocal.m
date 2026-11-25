% MANUAL: Exception handling
% The manual says GOAL's nonlocal variables must not be unique unless they
% are io.state; mmc accepts a unique array that GOAL destructively updates,
% and so does merkc (design document, "Where mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:7
% OUTPUT-NOT:{{.}}

:- module run_unique_nonlocal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module array, exception, int, list, string.

main(!IO) :-
    A0 = array.init(3, 0),
    ( try [] array.set(0, 7, A0, A1)
    then
        io.format("%d\n", [i(array.lookup(A1, 0))], !IO)
    catch_any _ ->
        io.write_string("caught\n", !IO)
    ).
