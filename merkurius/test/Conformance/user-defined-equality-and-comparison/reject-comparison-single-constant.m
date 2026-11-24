% MANUAL: User-defined equality and comparison
% Nor a user-defined comparison.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_comparison_single_constant.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type unit                                        % ERROR
    --->    unit
            where comparison is unit_compare.

:- pred unit_compare(comparison_result::uo, unit::in, unit::in) is det.
unit_compare((=), _, _).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
