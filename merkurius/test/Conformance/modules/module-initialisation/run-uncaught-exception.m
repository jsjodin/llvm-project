% MANUAL: Module initialisation
% An initialiser that ends with an uncaught exception aborts the program:
% the initialisers after it, main and the finalisers do not run.
% MERKC-FLAGS: -fmercury-messages
%
% RUN: merkc -O0 -fmercury-messages %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fmercury-messages %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:first
% OUTPUT-NEXT:Uncaught Mercury exception:
% OUTPUT-NEXT:Software Error: first failed
% OUTPUT-NOT:{{.}}

:- module run_uncaught_exception.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module require.

:- initialise first/2.
:- initialise second/2.
:- finalise last/2.

:- pred first(io::di, io::uo) is det.
first(!IO) :-
    io.write_string("first\n", !IO),
    io.flush_output(!IO),
    error("first failed").

:- pred second(io::di, io::uo) is det.
second(!IO) :-
    io.write_string("second\n", !IO).

:- pred last(io::di, io::uo) is det.
last(!IO) :-
    io.write_string("last\n", !IO).

main(!IO) :-
    io.write_string("main\n", !IO).
