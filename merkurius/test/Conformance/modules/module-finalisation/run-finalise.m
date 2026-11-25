% MANUAL: Module finalisation
% finalise directives: their predicates run after main, in the order of the
% directives, with finalize as a synonym; an impure one without arguments,
% and one that sets the exit status.
%
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:main
% OUTPUT-NEXT:first
% OUTPUT-NEXT:second
% OUTPUT-NEXT:third
% OUTPUT-NOT:{{.}}

:- module run_finalise.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- finalise first/2.
:- finalize second/0.
:- finalise third/2.

:- pred first(io::di, io::uo) is det.
first(!IO) :-
    io.write_string("first\n", !IO).

:- impure pred second is cc_multi.
second :-
    impure impure_true,
    trace [io(!IO)] (
        io.write_string("second\n", !IO)
    ).
:- pragma no_determinism_warning(second/0).

:- pred third(io::di, io::uo) is det.
third(!IO) :-
    io.write_string("third\n", !IO),
    io.set_exit_status(3, !IO).

main(!IO) :-
    io.write_string("main\n", !IO).
