% MANUAL: User-defined equality and comparison
% A discriminated union whose definition is a single zero-arity
% constructor may not have a user-defined equality.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_single_constant.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type unit                                        % ERROR
    --->    unit
            where equality is unit_equals.

:- pred unit_equals(unit::in, unit::in) is semidet.
unit_equals(_, _).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
