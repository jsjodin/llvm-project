% MANUAL: Existentially typed data types
% An existentially quantified variable must occur in the constructor's
% arguments, or in its constraints in the range of a functional dependency:
% T2 does not, without one.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_undetermined_variable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass xable(T1, T2) where [ func x(T1) = T2 ].

:- type t3                                          % ERROR
    --->    some [T1, T2] f5(T1) => xable(T1, T2).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
