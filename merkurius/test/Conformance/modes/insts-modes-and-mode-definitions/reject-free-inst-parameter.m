% MANUAL: Insts modes and mode definitions
% The inst variables of an inst definition's body must be its parameters.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_free_inst_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- inst wrapped(I) == bound(f(J)).                  % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
