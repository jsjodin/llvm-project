% MANUAL: Predicate and function mode declarations
% Mode analysis checks that each output argument is bound at the end.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_output_not_bound.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred p(int::in, int::out) is det.
p(_, _).                                            % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
