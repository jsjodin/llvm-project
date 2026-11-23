% MANUAL: Insts modes and mode definitions
% There must not be more than one inst definition with the same name and
% arity in one module.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_duplicate_inst.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- inst yesish ---> yes(ground).
:- inst yesish ---> no.                             % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
