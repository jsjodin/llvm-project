% MANUAL: Predicate and function type declarations
% Either every argument of a declaration has a mode or none does.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_some_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::in, int).                            % ERROR

p(_, _).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
