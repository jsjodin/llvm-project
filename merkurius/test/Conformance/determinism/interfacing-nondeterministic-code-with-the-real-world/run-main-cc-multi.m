% MANUAL: Interfacing nondeterministic code with the real world
% main/2 may be cc_multi: the program commits to its first solution.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NOT:{{.}}

:- module run_main_cc_multi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- pred digit(int::out) is multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

main(!IO) :-
    digit(D),
    io.write_line(D, !IO).
