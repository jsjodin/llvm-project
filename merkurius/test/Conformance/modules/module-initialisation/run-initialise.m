% MANUAL: Module initialisation
% initialise directives: their predicates run before main, in the order of
% the directives, a pred(io::di, io::uo) and an impure predicate without
% arguments, det and cc_multi, with initialize as a synonym.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:first
% OUTPUT-NEXT:second
% OUTPUT-NEXT:1
% OUTPUT-NEXT:main
% OUTPUT-NOT:{{.}}

:- module run_initialise.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- initialise first/2.
:- initialize second/0.
:- initialise third/2.

:- pred first(io::di, io::uo) is det.
first(!IO) :-
    io.write_string("first\n", !IO).

:- impure pred second is det.
second :-
    impure impure_true,
    trace [io(!IO)] (
        io.write_string("second\n", !IO)
    ).

:- pred third(io::di, io::uo) is cc_multi.
third(!IO) :-
    ( X = 1 ; X = 2 ),
    io.write_line(X, !IO).

main(!IO) :-
    io.write_string("main\n", !IO).
