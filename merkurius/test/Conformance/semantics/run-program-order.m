% MANUAL: Semantics
% A program runs its module's initialisation goals, then main/2, then its
% finalisation goals.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:initialisation
% OUTPUT-NEXT:main
% OUTPUT-NEXT:finalisation
% OUTPUT-NOT:{{.}}

:- module run_program_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- initialise init/2.
:- finalise final/2.

:- pred init(io::di, io::uo) is det.
init(!IO) :- io.write_string("initialisation\n", !IO).

:- pred final(io::di, io::uo) is det.
final(!IO) :- io.write_string("finalisation\n", !IO).

main(!IO) :- io.write_string("main\n", !IO).
