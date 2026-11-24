% MANUAL: An example module
% The manual's example module, used through its abstract type and its
% exported predicates.
% MODULES: Inputs/example_queue.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/example_queue.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/example_queue.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{1, 2}
% OUTPUT-NOT:{{.}}

:- module run_queue.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module example_queue.

main(!IO) :-
    empty_queue(Q0),
    put(Q0, 1, Q1),
    put(Q1, 2, Q2),
    ( if get(Q2, X, Q3), get(Q3, Y, Q4), empty_queue(Q4) then
        io.write_line({X, Y}, !IO)
    else
        io.write_string("wrong\n", !IO)
    ).
