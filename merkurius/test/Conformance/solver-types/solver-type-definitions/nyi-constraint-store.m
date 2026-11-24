% MANUAL: Solver type definitions
% A constraint store of mutables, which are separate mutable
% declarations.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: desugaring mutable declarations

:- module nyi_constraint_store.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type w
    where representation is int,
          constraint_store is [mutable(next, int, 0, ground, [untrailed])].

main(!IO) :-
    promise_pure ( semipure get_next(N) ),
    io.write_int(N, !IO), io.nl(!IO).
