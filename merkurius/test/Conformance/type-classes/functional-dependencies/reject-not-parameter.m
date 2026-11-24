% MANUAL: Functional dependencies
% Each variable of a functional dependency must be a parameter of the
% class.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_not_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass baz(A, B) <= (A -> C) where [          % ERROR
    func name(A, B) = string
].

main(!IO) :-
    io.write_string("unreachable\n", !IO).
