% MANUAL: Module finalisation
% If main ends with an uncaught exception, no finaliser runs.
% MERKC-FLAGS: -fmercury-messages
%
% RUN: merkc -O0 -fmercury-messages %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fmercury-messages %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:main
% OUTPUT-NEXT:Uncaught Mercury exception:
% OUTPUT-NEXT:Software Error: main failed
% OUTPUT-NOT:{{.}}

:- module run_main_exception.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module require.

:- finalise last/2.

:- pred last(io::di, io::uo) is det.
last(!IO) :-
    io.write_string("last\n", !IO).

main(!IO) :-
    io.write_string("main\n", !IO),
    io.flush_output(!IO),
    error("main failed").
